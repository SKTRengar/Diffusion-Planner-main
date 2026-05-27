"""
Random training-time debug logs for Bezier control-point supervision.

Each trigger randomly picks ONE scenario clip inside the current batch (one batch
index, ego agent only) and plots/logs its GT fit vs model-predicted control points.
Does not overlay or aggregate the whole batch.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion_planner.utils.bezier_utils import (
    BezierStateNormalizer,
    bezier_degree_to_num_control_points,
    coeffs_to_control_points,
    fit_bezier_control_points,
    pin_bezier_start,
    bezier_trajectory_with_heading,
    sample_bezier_trajectory,
    split_flat_state,
)

_LOGGER: Optional[logging.Logger] = None
_LOGGER_DIR: Optional[str] = None


def _get_bezier_debug_logger(debug_dir: str) -> logging.Logger:
    """File + console logger under ``debug_dir/bezier_debug.log``."""
    global _LOGGER, _LOGGER_DIR
    if _LOGGER is not None and _LOGGER_DIR == debug_dir:
        return _LOGGER

    os.makedirs(debug_dir, exist_ok=True)
    log_path = os.path.join(debug_dir, "bezier_debug.log")

    logger = logging.getLogger("diffusion_planner.bezier_debug")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    _LOGGER = logger
    _LOGGER_DIR = debug_dir
    return logger


def _format_control_points(ctrl: torch.Tensor) -> str:
    lines = []
    for i in range(ctrl.shape[0]):
        x, y = ctrl[i, 0].item(), ctrl[i, 1].item()
        lines.append(f"  P{i}: ({x:.4f}, {y:.4f})")
    return "\n".join(lines)


def _plot_bezier_debug(
    save_path: str,
    gt_waypoints: torch.Tensor,
    gt_ctrl: torch.Tensor,
    gt_curve: torch.Tensor,
    pred_ctrl: torch.Tensor,
    pred_waypoints: torch.Tensor,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    gw = gt_waypoints.detach().cpu().numpy()
    gc = gt_curve.detach().cpu().numpy()
    pw = pred_waypoints.detach().cpu().numpy()
    gcp = gt_ctrl.detach().cpu().numpy()
    pcp = pred_ctrl.detach().cpu().numpy()
    n_pred = pw.shape[0]

    # Curves (underneath)
    ax.plot(gc[:, 0], gc[:, 1], "-", color="C0", linewidth=1.5, alpha=0.6, label="GT Bezier curve", zorder=2)
    ax.plot(pw[:, 0], pw[:, 1], "-", color="C1", linewidth=1.5, alpha=0.6, label="Pred Bezier curve", zorder=2)

    # Discrete waypoints (80 poses @ physical T) — same style as GT
    ax.scatter(
        gw[:, 0],
        gw[:, 1],
        s=28,
        c="C0",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
        label=f"GT waypoints ({gw.shape[0]})",
        zorder=4,
    )
    ax.scatter(
        pw[:, 0],
        pw[:, 1],
        s=42,
        c="C1",
        alpha=0.95,
        edgecolors="darkred",
        linewidths=0.6,
        marker="o",
        label=f"Pred waypoints ({n_pred})",
        zorder=6,
    )

    ax.scatter(gcp[:, 0], gcp[:, 1], c="C0", s=60, zorder=5)
    ax.scatter(pcp[:, 0], pcp[:, 1], c="C1", s=60, zorder=5, marker="x")
    for i, (x, y) in enumerate(gcp):
        ax.annotate(f"P{i}", (x, y), fontsize=8, color="C0")
    for i, (x, y) in enumerate(pcp):
        ax.annotate(f"P{i}", (x, y), fontsize=8, color="C1")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("x (ego)")
    ax.set_ylabel("y (ego)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def maybe_log_bezier_training_debug(
    *,
    enabled: bool,
    prob: float,
    rank: int,
    debug_dir: Optional[str],
    epoch: int,
    batch_idx: int,
    diffusion_t: torch.Tensor,
    ego_dpm_loss: torch.Tensor,
    loss: Dict[str, Any],
    all_gt: torch.Tensor,
    score: torch.Tensor,
    ego_future_xy: torch.Tensor,
    current_states: torch.Tensor,
    coeff_norm: BezierStateNormalizer,
    coeff_dim: int,
    bezier_degree: int,
    trajectory_time_horizon: float,
    future_len: int,
    model_type: str,
    alpha_planning_loss: float = 1.0,
) -> None:
    if not enabled or prob <= 0.0 or debug_dir is None:
        return
    if rank != 0:
        return
    if random.random() >= prob:
        return

    with torch.no_grad():
        batch_size = all_gt.shape[0]
        scenario_idx = random.randrange(batch_size)
        num_ctrl = bezier_degree_to_num_control_points(bezier_degree)

        # --- ego sample ---
        gt_xy = ego_future_xy[scenario_idx].detach()
        current_xy = current_states[scenario_idx, 0, :2].detach()

        gt_ctrl = fit_bezier_control_points(
            gt_xy.unsqueeze(0),
            degree=bezier_degree,
            time_horizon=trajectory_time_horizon,
            start_xy=current_xy.unsqueeze(0),
        )[0]
        gt_curve = sample_bezier_trajectory(
            gt_ctrl.unsqueeze(0),
            future_len,
            degree=bezier_degree,
            time_horizon=trajectory_time_horizon,
        )[0]

        _, pred_coeffs_norm = split_flat_state(score[scenario_idx, 0].unsqueeze(0), coeff_dim)
        pred_coeffs_phys = coeff_norm.inverse(pred_coeffs_norm)[0, 0]
        pred_ctrl = coeffs_to_control_points(pred_coeffs_phys.unsqueeze(0), num_ctrl)
        pred_ctrl = pin_bezier_start(pred_ctrl, current_xy.unsqueeze(0))[0]
        # Same decode path as inference: coeffs -> control points -> 80 poses at physical T
        pred_traj = bezier_trajectory_with_heading(
            pred_ctrl.unsqueeze(0),
            future_len,
            degree=bezier_degree,
            time_horizon=trajectory_time_horizon,
        )[0]
        pred_waypoints = pred_traj[:, :2]

        t_val = diffusion_t[scenario_idx].item()
        sample_ego_loss = ego_dpm_loss[scenario_idx, 0].item()

        def _loss_scalar(key: str) -> float:
            v = loss.get(key, 0.0)
            if torch.is_tensor(v):
                return float(v.detach().item())
            return float(v)

        batch_total = _loss_scalar("loss")
        batch_ego = _loss_scalar("ego_planning_loss")
        batch_neighbor = _loss_scalar("neighbor_prediction_loss")

        logger = _get_bezier_debug_logger(debug_dir)
        sep = "=" * 72
        logger.info(sep)
        logger.info("[Bezier debug]")
        logger.info(
            "epoch=%d, batch=%d, scenario=%d/%d (random), ego-only, diffusion_t=%.4f, model=%s",
            epoch,
            batch_idx,
            scenario_idx,
            batch_size - 1,
            t_val,
            model_type,
        )
        logger.info(
            "loss: batch_total=%.6f, batch_ego=%.6f, batch_neighbor=%.6f, selected_scenario_ego=%.6f",
            batch_total,
            batch_ego,
            batch_neighbor,
            sample_ego_loss,
        )
        logger.info(
            "selection: 1 random scenario in batch (index=%d / batch_size=%d), ego agent only",
            scenario_idx,
            batch_size,
        )
        logger.info(
            "start (current xy): (%.4f, %.4f)",
            current_xy[0].item(),
            current_xy[1].item(),
        )
        logger.info(
            "GT end (last waypoint): (%.4f, %.4f)",
            gt_xy[-1, 0].item(),
            gt_xy[-1, 1].item(),
        )
        logger.info(
            "Pred end (last waypoint @ T=%.1fs): (%.4f, %.4f)",
            trajectory_time_horizon,
            pred_waypoints[-1, 0].item(),
            pred_waypoints[-1, 1].item(),
        )
        logger.info(
            "Pred end (model Pn): (%.4f, %.4f)",
            pred_ctrl[-1, 0].item(),
            pred_ctrl[-1, 1].item(),
        )
        logger.info("GT fitted control points:\n%s", _format_control_points(gt_ctrl))
        logger.info("Model predicted control points:\n%s", _format_control_points(pred_ctrl))

        plot_name = f"epoch{epoch:04d}_batch{batch_idx:05d}_scenario{scenario_idx:03d}.png"
        plot_path = os.path.join(debug_dir, plot_name)
        _plot_bezier_debug(
            plot_path,
            gt_xy,
            gt_ctrl,
            gt_curve,
            pred_ctrl,
            pred_waypoints,
            title=f"epoch={epoch} batch={batch_idx} scenario={scenario_idx} t={t_val:.3f}",
        )
        logger.info("plot saved: %s", plot_path)
        logger.info("log file: %s", os.path.join(debug_dir, "bezier_debug.log"))
        logger.info(sep)
