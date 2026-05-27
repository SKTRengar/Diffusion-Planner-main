from torch.optim.lr_scheduler import LinearLR


def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    """
    Linear warmup then constant LR.
    Uses a single LinearLR (no SequentialLR) to avoid deprecated step(epoch) calls.
    """
    assert epoch >= warm_up_epoch
    return LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=max(warm_up_epoch - 1, 1),
    )