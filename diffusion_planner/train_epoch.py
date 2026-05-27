from tqdm import tqdm
import torch
from torch import nn

from diffusion_planner.utils.data_augmentation import StatePerturbation   
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation=None, epoch: int = 0):
    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    use_bezier = getattr(args, "use_bezier", False)
    debug_prob = float(getattr(args, "bezier_debug_prob", 0.0))
    debug_enabled = use_bezier and debug_prob > 0.0

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch_idx, batch in enumerate(data_epoch):
            '''
            data structure in batch: Tuple(Tensor) 

            ego_current_state,
            ego_future_gt,

            neighbor_agents_past,
            neighbors_future_gt,

            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,

            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,

            static_objects,

            '''

            # prepare data
            inputs = {
                'ego_current_state': batch[0].to(args.device),

                'neighbor_agents_past': batch[2].to(args.device),

                'lanes': batch[4].to(args.device),
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),

                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),

                'static_objects': batch[10].to(args.device)

            }

            ego_future = batch[1].to(args.device)
            neighbors_future = batch[3].to(args.device)
            # Normalize to ego-centric
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            if use_bezier:
                assert getattr(args, "bezier_state_normalizer", None) is not None, (
                    "use_bezier=True requires bezier_state_normalizer"
                )

            debug_context = None
            if debug_enabled:
                debug_context = {
                    "enabled": True,
                    "prob": debug_prob,
                    "rank": ddp.get_rank(),
                    "debug_dir": getattr(args, "bezier_debug_dir", None),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "future_len": args.future_len,
                    "alpha_planning_loss": args.alpha_planning_loss,
                }

            # Waypoint GT -> xy (then Bezier fit in loss); cos/sin kept for legacy waypoint mode only
            ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )

            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
            neighbors_future[mask] = 0.
            inputs = args.observation_normalizer(inputs)
                  
            # call the mdoel
            optimizer.zero_grad()
            loss = {}

            loss, _ = diffusion_loss_func(
                model,
                inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                args.state_normalizer,
                loss,
                args.diffusion_model_type,
                use_bezier=use_bezier,
                bezier_degree=getattr(args, "bezier_degree", 6),
                coeff_norm=getattr(args, "bezier_state_normalizer", None) if use_bezier else None,
                trajectory_time_horizon=float(getattr(args, "trajectory_time_horizon", 8.0)),
                alpha_bezier_waypoint_loss=float(
                    getattr(args, "alpha_bezier_waypoint_loss", 0.5)
                ),
                debug_context=debug_context,
            )

            loss['loss'] = loss['neighbor_prediction_loss'] + args.alpha_planning_loss * loss['ego_planning_loss']

            total_loss = loss['loss'].item()

            # loss backward
            loss['loss'].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()
            
            data_epoch.set_postfix(loss='{:.4f}'.format(total_loss))
            epoch_loss.append(
                {k: (v.detach().item() if torch.is_tensor(v) else v) for k, v in loss.items()}
            )

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
        
    return epoch_mean_loss, epoch_mean_loss['loss']
