import os
import torch
import argparse
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_planner.model.diffusion_planner import Diffusion_Planner

from diffusion_planner.utils.train_utils import (
    set_seed,
    save_model,
    resume_model,
    apply_pretrained_arch_args,
    load_encoder_from_checkpoint,
    encoder_is_frozen,
    get_encoder_status_message,
    should_freeze_encoder,
    configure_encoder_for_epoch,
    build_ddp_kwargs,
    ddp_wrap_mode,
    get_checkpoint_epoch,
    merge_resume_training_args,
    build_training_optimizer,
)
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.bezier_utils import BezierStateNormalizer
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils import ddp

from diffusion_planner.train_epoch import train_epoch

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "diffusion-planner-training")', default="diffusion-planner-training")
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default=".")

    # Data
    parser.add_argument('--train_set', type=str, help='path to train data', default=None)
    parser.add_argument('--train_set_list', type=str, help='data list of train data', default=None)
    parser.add_argument(
        '--train_subset_size',
        type=int,
        default=200000,
        help='randomly sample this many npz files from train_set_list per run (<=0 = use all)',
    )

    parser.add_argument('--future_len', type=int, help='number of time point', default=80)
    parser.add_argument(
        '--trajectory_time_horizon',
        type=float,
        default=8.0,
        help='physical future horizon in seconds for Bezier sampling (nuPlan default 8s)',
    )
    parser.add_argument('--time_len', type=int, help='number of time point', default=21)

    parser.add_argument('--agent_state_dim', type=int, help='past state dim for agents', default=11)
    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    parser.add_argument('--static_objects_state_dim', type=int, help='state dim for static objects', default=10)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_state_dim', type=int, help='state dim for lane point', default=12)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_state_dim', type=int, help='state dim for route lane point', default=12)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    
    # DataLoader parameters
    parser.add_argument('--augment_prob', type=float, help='augmentation probability', default=0.5)
    parser.add_argument('--normalization_file_path', default='normalization.json', help='filepath of normalizaiton.json', type=str)
    parser.add_argument('--use_data_augment', default=True, type=boolean)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    
    # Training
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=800)
    parser.add_argument('--save_utd', type=int, help='save frequency', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 2048)', default=2048)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=0.0005)
    parser.add_argument('--warm_up_epoch', type=int, help='number of warm up', default=5)
    parser.add_argument('--encoder_drop_path_rate', type=float, help='encoder drop out rate', default=0.1)
    parser.add_argument('--decoder_drop_path_rate', type=float, help='decoder drop out rate', default=0.1)

    parser.add_argument('--alpha_planning_loss', type=float, help='coefficient of planning loss (default: 1.0)', default=1.0)
    parser.add_argument(
        '--alpha_bezier_waypoint_loss',
        type=float,
        default=0.2,
        help='weight for Bezier waypoint L2 aux loss (added to coeff L2 per agent, Bezier mode)',
    )

    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')

    parser.add_argument('--use_ema', default=True, type=boolean)

    # Model
    parser.add_argument('--encoder_depth', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_depth', type=int, help='number of decoding layers', default=3)
    parser.add_argument('--num_heads', type=int, help='number of multi-head', default=6)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=192)
    parser.add_argument('--diffusion_model_type', type=str, help='type of diffusion model [x_start, score]', choices=['score', 'x_start'], default='x_start')

    # decoder
    parser.add_argument('--predicted_neighbor_num', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--use_bezier', default=True, type=boolean, help='predict 6th-order Bezier coefficients instead of waypoints')
    parser.add_argument('--bezier_degree', type=int, default=6, help='Bezier polynomial degree (6 -> 7 control points)')
    parser.add_argument(
        '--bezier_debug_prob',
        type=float,
        default=0.03,
        help='probability per batch to log/plot GT vs predicted Bezier control points (rank 0 only, Bezier mode)',
    )
    parser.add_argument('--freeze_encoder', default=True, type=boolean, help='freeze encoder and train decoder only')
    parser.add_argument(
        '--encoder_freeze_epochs',
        type=int,
        default=400,
        help='0-based epoch count to freeze encoder (epochs 0..N-1 frozen; unfreeze from epoch N)',
    )
    parser.add_argument(
        '--encoder_init_path',
        type=str,
        default='/home/skt/Code/Diffusion-Planner-main/checkpoints/model.pth',
        help='pretrained checkpoint for encoder weights (file or directory)',
    )
    parser.add_argument(
        '--encoder_args_path',
        type=str,
        default='/home/skt/Code/Diffusion-Planner-main/checkpoints/args.json',
        help='args.json aligned with the pretrained encoder',
    )
    parser.add_argument(
        '--resume_model_path',
        type=str,
        help='optional Bezier-training run dir or .pth to resume decoder/optimizer (encoder still from encoder_init_path)',
        default=None,
    )

    parser.add_argument('--use_wandb', default=False, type=boolean)
    parser.add_argument('--notes', default='', type=str)

    # distributed training parameters
    parser.add_argument('--ddp', default=True, type=boolean, help='use ddp or not')
    parser.add_argument('--port', default='22323', type=str, help='port')

    args = parser.parse_args()
    merge_resume_training_args(args)

    # Align encoder architecture with the pretrained checkpoint before building the model.
    apply_pretrained_arch_args(args, args.encoder_args_path, rank=0)

    args.use_bezier = getattr(args, "use_bezier", True)
    args.bezier_degree = getattr(args, "bezier_degree", 6)
    args.freeze_encoder = getattr(args, "freeze_encoder", True)
    args.encoder_freeze_epochs = getattr(args, "encoder_freeze_epochs", 400)

    if args.use_bezier:
        args.bezier_state_normalizer = BezierStateNormalizer.from_json(args)
        args.state_normalizer = args.bezier_state_normalizer
    else:
        args.bezier_state_normalizer = None
        args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    
    return args

def model_training(args):

    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    if global_rank == 0:
        # Logging
        from datetime import datetime
        time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if args.resume_model_path is not None:
            resume_dir = args.resume_model_path
            if os.path.isfile(resume_dir):
                resume_dir = os.path.dirname(os.path.abspath(resume_dir))
            else:
                resume_dir = os.path.abspath(resume_dir)
            save_path = os.path.join(resume_dir, "")
            run_name = os.path.basename(resume_dir.rstrip(os.sep))
            args.name = run_name
            os.makedirs(save_path, exist_ok=True)
            print(f"Resume: continue checkpoints in {save_path}")
        elif args.use_bezier:
            run_name = f"Bezier_degree_{args.bezier_degree}_{time}"
            save_path = f"{args.save_dir}/training_log/{run_name}/"
            args.name = run_name
            os.makedirs(save_path, exist_ok=True)
        else:
            run_name = f"{args.name}_{time}"
            save_path = f"{args.save_dir}/training_log/{run_name}/"
            args.name = run_name
            os.makedirs(save_path, exist_ok=True)
        args.bezier_debug_dir = os.path.join(save_path, "bezier_debug")
        if args.use_bezier and args.bezier_debug_prob > 0:
            os.makedirs(args.bezier_debug_dir, exist_ok=True)
            log_path = os.path.join(args.bezier_debug_dir, "bezier_debug.log")
            print(
                f"Bezier debug: prob={args.bezier_debug_prob}, "
                f"dir={args.bezier_debug_dir}, log={log_path}"
            )

        print("------------- {} -------------".format(run_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))
        if args.resume_model_path is not None:
            print(f"Resume decoder from: {args.resume_model_path}")
        print(f"Encoder init from: {args.encoder_init_path}")
        if args.freeze_encoder and args.encoder_freeze_epochs > 0:
            print(
                f"Encoder schedule: frozen for 0-based epochs [0, {args.encoder_freeze_epochs - 1}] "
                f"(1-indexed epochs 1-{args.encoder_freeze_epochs}), "
                f"joint training from 1-indexed epoch {args.encoder_freeze_epochs + 1}"
            )
        print(f"Checkpoints save to: {save_path}")

        # Save args
        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer, BezierStateNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }

        from mmengine.fileio import dump
        dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    subset_size = args.train_subset_size if args.train_subset_size > 0 else None
    train_set = DiffusionPlannerData(
        args.train_set,
        args.train_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,
        subset_size=subset_size,
        subset_seed=args.seed,
    )
    train_sampler = DistributedSampler(train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size//ddp.get_world_size(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
   
    if global_rank == 0:
        if subset_size is not None and train_set.subset_num < train_set.total_num:
            print(
                f"Dataset: using random subset {train_set.subset_num} / {train_set.total_num} "
                f"(seed={args.seed})\n"
            )
        else:
            print("Dataset Prepared: {} train samples\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model (DDP wrap after encoder freeze state is configured)
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == 'cuda' else args.device)

    load_encoder_from_checkpoint(
        diffusion_planner,
        args.encoder_init_path,
        use_ema=True,
        rank=global_rank,
    )

    init_epoch = 0
    if args.resume_model_path is not None:
        init_epoch = get_checkpoint_epoch(args.resume_model_path)

    configure_encoder_for_epoch(
        diffusion_planner, None, init_epoch, args, rank=global_rank
    )

    model_ema = None

    def wrap_ddp(module, epoch_for_ddp: int):
        if not args.ddp:
            return module
        return DDP(module, **build_ddp_kwargs(module, epoch_for_ddp, args, [rank]))

    def sync_ddp_for_epoch(epoch_for_ddp: int):
        nonlocal diffusion_planner
        if not args.ddp:
            return
        mode = ddp_wrap_mode(epoch_for_ddp, args)
        if getattr(args, "_ddp_encoder_mode", None) == mode and hasattr(
            diffusion_planner, "module"
        ):
            return
        core = (
            diffusion_planner.module
            if hasattr(diffusion_planner, "module")
            else diffusion_planner
        )
        diffusion_planner = wrap_ddp(core, epoch_for_ddp)
        args._ddp_encoder_mode = mode
        if model_ema is not None:
            model_ema.module = diffusion_planner
        if global_rank == 0:
            if mode == "trainable":
                print(
                    f"DDP: full model trainable (find_unused_parameters=False), "
                    f"epoch {epoch_for_ddp + 1}"
                )
            elif mode == "frozen_ignored":
                n = sum(p.numel() for p in core.encoder.parameters())
                print(
                    f"DDP: encoder frozen ({n} params ignored), epoch {epoch_for_ddp + 1}"
                )
            else:
                print(
                    f"DDP: encoder frozen (find_unused_parameters=True, PyTorch<2.1), "
                    f"epoch {epoch_for_ddp + 1}"
                )

    sync_ddp_for_epoch(init_epoch)

    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

    model_core = ddp.get_model(diffusion_planner, args.ddp)
    if global_rank == 0:
        if should_freeze_encoder(init_epoch, args):
            print(
                f"Encoder frozen for epoch {init_epoch + 1} "
                f"(epoch < {args.encoder_freeze_epochs})."
            )
        elif args.freeze_encoder:
            print(f"Encoder trainable for epoch {init_epoch + 1} (epoch >= {args.encoder_freeze_epochs}).")

    trainable_params = [p for p in model_core.parameters() if p.requires_grad]
    if global_rank == 0:
        print("Model Params (total): {}".format(sum(p.numel() for p in model_core.parameters())))
        print("Trainable Params: {}".format(sum(p.numel() for p in trainable_params)))
        if args.use_bezier:
            print(
                "Bezier mode: degree={}, control points={}, coeff_dim={}, "
                "waypoint_aux_weight={}".format(
                    args.bezier_degree,
                    args.bezier_degree + 1,
                    (args.bezier_degree + 1) * 2,
                    args.alpha_bezier_waypoint_loss,
                )
            )

    resume_path = args.resume_model_path
    optimizer = build_training_optimizer(
        model_core, args, init_epoch, resume_path=resume_path
    )
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    wandb_id = None
    if resume_path is not None:
        skip_encoder_on_resume = should_freeze_encoder(init_epoch, args)
        if global_rank == 0:
            print(
                f"Resuming from {resume_path} "
                f"(checkpoint epoch {init_epoch}, "
                f"optimizer param_groups={len(optimizer.param_groups)}, "
                f"encoder {'frozen' if skip_encoder_on_resume else 'trainable'})"
            )
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            resume_path,
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            args.device,
            skip_encoder=skip_encoder_on_resume,
        )
        if skip_encoder_on_resume:
            load_encoder_from_checkpoint(
                model_core,
                args.encoder_init_path,
                use_ema=True,
                rank=global_rank,
            )
        configure_encoder_for_epoch(
            model_core, optimizer, init_epoch, args, rank=global_rank
        )
        sync_ddp_for_epoch(init_epoch)
        model_core = ddp.get_model(diffusion_planner, args.ddp)
    elif global_rank == 0:
        print("First Bezier training run: decoder randomly initialized.")

    # logger
    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank) 

    if args.ddp:
        torch.distributed.barrier()

    # begin training
    for epoch in range(init_epoch, train_epochs):
        configure_encoder_for_epoch(
            model_core, optimizer, epoch, args, rank=global_rank
        )
        sync_ddp_for_epoch(epoch)
        model_core = ddp.get_model(diffusion_planner, args.ddp)
        encoder_frozen = encoder_is_frozen(model_core)
        encoder_status = get_encoder_status_message(model_core, args)
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs} | [Encoder] {encoder_status}")

        train_loss, train_total_loss = train_epoch(
            train_loader, diffusion_planner, optimizer, args, model_ema, aug, epoch=epoch
        )
        


        if global_rank == 0:
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch+1)
            wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch+1)
            wandb_logger.log_metrics(
                {
                    "encoder/frozen": float(encoder_frozen),
                    "encoder/trainable": float(not encoder_frozen),
                },
                step=epoch + 1,
            )

            if (epoch+1) % args.save_utd == 0:
                # save model at the end of epoch
                save_model(
                    diffusion_planner,
                    optimizer,
                    scheduler,
                    save_path,
                    epoch,
                    train_total_loss,
                    wandb_logger.id,
                    model_ema.ema,
                    rank=global_rank,
                )
                print(f"Model saved in {save_path}\n")

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)

if __name__ == "__main__":

    args = get_args()
    
    # Run
    model_training(args)
