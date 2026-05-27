import inspect
from typing import Optional

import torch
import random
import numpy as np
from mmengine import fileio
import io
import os
import json
from torch.nn.parallel import DistributedDataParallel as DDP

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def opendata(path):
    
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)

    return npz_data

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss

# Architecture keys that must match a pretrained encoder checkpoint.
PRETRAINED_ARCH_KEYS = (
    "device",
    "future_len",
    "trajectory_time_horizon",
    "time_len",
    "agent_state_dim",
    "agent_num",
    "static_objects_state_dim",
    "static_objects_num",
    "lane_len",
    "lane_state_dim",
    "lane_num",
    "route_len",
    "route_state_dim",
    "route_num",
    "encoder_drop_path_rate",
    "decoder_drop_path_rate",
    "encoder_depth",
    "decoder_depth",
    "num_heads",
    "hidden_dim",
    "diffusion_model_type",
    "predicted_neighbor_num",
)


def apply_pretrained_arch_args(args, encoder_args_path: str, rank: int = 0) -> None:
    """Align model architecture with the pretrained encoder args.json."""
    ckpt_args = openjson(encoder_args_path)
    for key in PRETRAINED_ARCH_KEYS:
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    if rank == 0:
        print(f"Applied pretrained architecture from {encoder_args_path}")


def _resolve_checkpoint_file(checkpoint_path: str) -> str:
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    if os.path.isdir(checkpoint_path):
        for name in ("model.pth", "latest.pth"):
            candidate = os.path.join(checkpoint_path, name)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")


def _extract_state_dict(ckpt_obj: dict, use_ema: bool = True) -> dict:
    if use_ema and "ema_state_dict" in ckpt_obj:
        return ckpt_obj["ema_state_dict"]
    if "model" in ckpt_obj:
        return ckpt_obj["model"]
    return ckpt_obj


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _extract_encoder_state_dict(state_dict: dict) -> dict:
    state_dict = _strip_module_prefix(state_dict)
    encoder_sd = {}
    for k, v in state_dict.items():
        if not k.startswith("encoder."):
            continue
        # Full model keys: encoder.encoder.* -> Diffusion_Planner_Encoder expects encoder.*
        if k.startswith("encoder.encoder."):
            k = k[len("encoder.") :]
        encoder_sd[k] = v
    return encoder_sd


def load_encoder_from_checkpoint(
    model,
    checkpoint_path: str,
    use_ema: bool = True,
    rank: int = 0,
) -> None:
    """
    Load only encoder weights from a pretrained checkpoint (e.g. checkpoints/model.pth).
    Decoder parameters are left unchanged (random init on first run).
    """
    ckpt_file = _resolve_checkpoint_file(checkpoint_path)
    ckpt_bytes = fileio.get(ckpt_file)
    with io.BytesIO(ckpt_bytes) as f:
        ckpt_obj = torch.load(f, map_location="cpu")

    if not isinstance(ckpt_obj, dict):
        raise ValueError(f"Unexpected checkpoint format: {ckpt_file}")

    state_dict = _extract_state_dict(ckpt_obj, use_ema=use_ema)
    encoder_sd = _extract_encoder_state_dict(state_dict)
    if len(encoder_sd) == 0:
        raise KeyError(f"No encoder weights found in {ckpt_file}")

    missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=True)
    if rank == 0:
        print(
            f"Loaded encoder from {ckpt_file} "
            f"({len(encoder_sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
        )


def freeze_encoder(model) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = True


def encoder_is_frozen(model) -> bool:
    return not any(p.requires_grad for p in model.encoder.parameters())


def should_freeze_encoder(epoch: int, args) -> bool:
    """Whether encoder should be frozen at the given 0-based training epoch."""
    if not getattr(args, "freeze_encoder", False):
        return False
    freeze_epochs = getattr(args, "encoder_freeze_epochs", 0)
    if freeze_epochs <= 0:
        return True
    return epoch < freeze_epochs


_DDP_SUPPORTS_IGNORED = "ignored_parameters" in inspect.signature(DDP.__init__).parameters


def build_ddp_kwargs(model, epoch: int, args, device_ids) -> dict:
    """
    DDP kwargs for staged encoder freeze.

    - Encoder frozen + PyTorch >=2.1: ``ignored_parameters`` only (no find_unused traversal).
    - Encoder frozen + PyTorch 2.0: ``find_unused_parameters=True`` (frozen weights skip grads).
    - Encoder trainable: ``find_unused_parameters=False`` (avoids extra graph walk; all
      params participate in Bezier decoder loss). Re-enable if a batch hits unused-param DDP errors.
    """
    kwargs: dict = {"device_ids": device_ids, "find_unused_parameters": False}
    if should_freeze_encoder(epoch, args):
        if _DDP_SUPPORTS_IGNORED:
            kwargs["ignored_parameters"] = list(model.encoder.parameters())
        else:
            kwargs["find_unused_parameters"] = True
    return kwargs


def ddp_wrap_mode(epoch: int, args) -> str:
    """Label for logging / DDP re-wrap decisions."""
    if not should_freeze_encoder(epoch, args):
        return "trainable"
    return "frozen_ignored" if _DDP_SUPPORTS_IGNORED else "frozen_find_unused"


def configure_encoder_for_epoch(
    model,
    optimizer,
    epoch: int,
    args,
    rank: int = 0,
) -> bool:
    """
    Set encoder freeze state from epoch vs encoder_freeze_epochs.
    When unfreezing, add encoder params to optimizer if missing.
    Returns True if encoder is frozen after this call.
    """
    if should_freeze_encoder(epoch, args):
        freeze_encoder(model)
        return True

    unfreeze_encoder(model)
    if optimizer is not None:
        added = add_encoder_params_to_optimizer(
            model, optimizer, getattr(args, "learning_rate", 0.0)
        )
        if rank == 0 and added > 0:
            print(
                f"Epoch {epoch + 1}: encoder unfrozen — joint training "
                f"({added} param tensors added to optimizer)."
            )
    return False


def get_encoder_status_message(model, args) -> str:
    """Human-readable encoder training status for logging."""
    frozen = encoder_is_frozen(model)
    trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.encoder.parameters())

    if not getattr(args, "freeze_encoder", False):
        return (
            f"encoder=trainable (freeze_encoder=False), "
            f"trainable {trainable_params}/{total_params} params"
        )

    freeze_epochs = getattr(args, "encoder_freeze_epochs", 0)
    if getattr(args, "freeze_encoder", False) and freeze_epochs > 0:
        if frozen:
            return (
                f"encoder=frozen"
            )
        return (
            f"encoder=unfrozen"
        )

    if frozen:
        return f"encoder=frozen"
    return (
        f"encoder=unfrozen"
    )


def add_encoder_params_to_optimizer(model, optimizer, lr: float) -> int:
    """Add encoder parameters to optimizer if not already present."""
    existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
    new_params = [p for p in model.encoder.parameters() if id(p) not in existing]
    if new_params:
        optimizer.add_param_group({"params": new_params, "lr": lr})
    return len(new_params)


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    return os.path.join(checkpoint_path, "latest.pth")


def _load_checkpoint_cpu(checkpoint_path: str) -> dict:
    ckpt_path = _resolve_checkpoint_path(checkpoint_path)
    ckpt_bytes = fileio.get(ckpt_path)
    with io.BytesIO(ckpt_bytes) as f:
        return torch.load(f, map_location="cpu")


def get_checkpoint_epoch(checkpoint_path: str) -> int:
    """Return saved epoch from a training checkpoint (0 if missing)."""
    ckpt = _load_checkpoint_cpu(checkpoint_path)
    return int(ckpt.get("epoch", 0))


def get_checkpoint_optimizer_num_groups(checkpoint_path: str) -> int:
    ckpt = _load_checkpoint_cpu(checkpoint_path)
    opt = ckpt.get("optimizer")
    if not isinstance(opt, dict):
        return 0
    return len(opt.get("param_groups", []))


def merge_resume_training_args(args) -> None:
    """Restore training hyperparameters from the run's args.json when resuming."""
    if not getattr(args, "resume_model_path", None):
        return
    resume_dir = args.resume_model_path
    if os.path.isfile(resume_dir):
        resume_dir = os.path.dirname(os.path.abspath(resume_dir))
    else:
        resume_dir = os.path.abspath(resume_dir)
    path = os.path.join(resume_dir, "args.json")
    if not os.path.isfile(path):
        return
    saved = openjson(path)
    for key in (
        "learning_rate",
        "warm_up_epoch",
        "batch_size",
        "alpha_planning_loss",
        "alpha_bezier_waypoint_loss",
        "augment_prob",
        "train_subset_size",
        "bezier_degree",
        "bezier_debug_prob",
    ):
        if key in saved:
            setattr(args, key, saved[key])


def build_training_optimizer(model, args, epoch: int, resume_path: Optional[str] = None):
    """
    AdamW with param groups matching staged encoder training.

    While encoder is frozen: decoder-only (1 group). After unfreeze: decoder + encoder
    (2 groups), matching checkpoints saved during joint training.
    """
    core = model.module if hasattr(model, "module") else model
    decoder_params = [p for p in core.decoder.parameters() if p.requires_grad]
    encoder_params = [p for p in core.encoder.parameters() if p.requires_grad]

    saved_groups = 0
    if resume_path:
        saved_groups = get_checkpoint_optimizer_num_groups(resume_path)

    lr = getattr(args, "learning_rate", 5e-4)
    if should_freeze_encoder(epoch, args):
        return torch.optim.AdamW([{"params": decoder_params, "lr": lr}])

    if saved_groups == 2 or (saved_groups == 0 and len(encoder_params) > 0):
        return torch.optim.AdamW(
            [
                {"params": decoder_params, "lr": lr},
                {"params": encoder_params, "lr": lr},
            ]
        )
    return torch.optim.AdamW([{"params": decoder_params + encoder_params, "lr": lr}])


def _unwrap_module(model):
    return model.module if hasattr(model, "module") else model


def _unwrap_model_state_dict(model) -> dict:
    """Full weights including frozen encoder (DDP-safe)."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def _sync_encoder_to_ema_state(model_state: dict, ema_state: dict) -> int:
    """
    Copy encoder.* tensors from model into EMA state so frozen encoder is
    preserved exactly in ema_state_dict (EMA may drift on unused grads otherwise).
    """
    count = 0
    for key, value in model_state.items():
        if key.startswith("encoder.") and key in ema_state:
            ema_state[key] = value.detach().clone()
            count += 1
    return count


def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema, rank: int = 0):
    """
    Save full model (encoder + decoder). Frozen encoder weights are included in
    model.state_dict(); encoder keys are synced into ema_state_dict before write.
    """
    model_state = _unwrap_model_state_dict(model)
    ema_state = ema.state_dict()
    encoder_synced = _sync_encoder_to_ema_state(model_state, ema_state)

    if rank == 0:
        encoder_keys = sum(1 for k in model_state if k.startswith("encoder."))
        decoder_keys = sum(1 for k in model_state if k.startswith("decoder."))
        print(
            f"Saving checkpoint: {encoder_keys} encoder + {decoder_keys} decoder tensors; "
            f"synced {encoder_synced} encoder tensors into EMA."
        )

    payload = {
        "epoch": epoch + 1,
        "model": model_state,
        "ema_state_dict": ema_state,
        "optimizer": optimizer.state_dict(),
        "schedule": scheduler.state_dict(),
        "loss": train_loss,
        "wandb_id": wandb_id,
    }

    with io.BytesIO() as f:
        torch.save(payload, f)
        fileio.put(f.getvalue(), f"{save_path}/model_epoch_{epoch + 1}_trainloss_{train_loss:.4f}.pth")
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")

def resume_model(path: str, model, optimizer, scheduler, ema, device, skip_encoder: bool = False):
    """
    Resume training checkpoint from a run directory (latest.pth) or a .pth file.
    When skip_encoder=True, only decoder / non-encoder weights are restored.
    """
    if os.path.isfile(path):
        ckpt_path = path
    else:
        ckpt_path = os.path.join(path, "latest.pth")

    ckpt = fileio.get(ckpt_path)
    with io.BytesIO(ckpt) as f:
        ckpt = torch.load(f, map_location=device)

    # load model
    try:
        state_dict = ckpt["model"]
    except (KeyError, TypeError):
        state_dict = ckpt

    if skip_encoder:
        if isinstance(state_dict, dict):
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if "encoder" not in k
            }
        print("Resume: skipping encoder weights (will reload from encoder_init_path)")

    target = _unwrap_module(model)
    strict = not skip_encoder
    try:
        missing, unexpected = target.load_state_dict(state_dict, strict=strict)
        print(
            f"Model load done from {ckpt_path} "
            f"(strict={strict}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
    except RuntimeError as e:
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        print(
            f"Model load (non-strict) from {ckpt_path}: {e}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    # load optimizer
    try:
        ckpt_groups = len(ckpt["optimizer"]["param_groups"])
        opt_groups = len(optimizer.param_groups)
        if ckpt_groups != opt_groups:
            raise ValueError(
                f"optimizer param_groups mismatch: checkpoint has {ckpt_groups}, "
                f"current optimizer has {opt_groups}"
            )
        optimizer.load_state_dict(ckpt["optimizer"])
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"Optimizer load done (param_groups={opt_groups}, lrs={lrs})")
    except Exception as e:
        print(f"WARNING: optimizer state not restored ({e}); using fresh AdamW state")

    # load schedule
    try:
        scheduler.load_state_dict(ckpt["schedule"])
        print("Schedule load done")
    except Exception as e:
        print(f"WARNING: scheduler state not restored ({e})")

    # load step
    try:
        init_epoch = int(ckpt["epoch"])
        print(f"Resume from checkpoint epoch {init_epoch} (next loop epoch index {init_epoch})")
    except Exception:
        init_epoch = 0
        print("WARNING: checkpoint has no epoch field; starting from epoch 0")

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(ckpt['ema_state_dict'])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema


