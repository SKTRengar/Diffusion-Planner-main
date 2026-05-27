"""
Diffusion training losses.

Bezier mode (use_bezier=True):
  - Ego and all predicted neighbors share the same representation:
    current (x,y,cos,sin) + Bezier control-point coefficients.
  - GT (x,y) at physical times T are least-squares fit with P0 pinned to current (x,y).
  - Diffusion noise / x_start supervision applies only to the coeff dims for every agent.
  - Coefficient L2 (MSE) plus optional waypoint L2 on sampled (x,y) at physical times T.
"""

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer
from diffusion_planner.utils.bezier_utils import (
    BezierStateNormalizer,
    bezier_degree_to_num_control_points,
    build_flat_state,
    control_points_to_coeffs,
    coeffs_to_control_points,
    fit_bezier_control_points,
    pin_bezier_start,
    sample_bezier_trajectory,
    split_flat_state,
)
from diffusion_planner.utils.bezier_training_debug import maybe_log_bezier_training_debug

EGO_AGENT_INDEX = 0


def _neighbor_agent_invalid_mask(
    neighbors_current: torch.Tensor,
    neighbors_future: torch.Tensor,
) -> torch.Tensor:
    """
    Per-agent invalid mask [B, Pn]. True = empty current state or no valid future xy.
    """
    neighbor_current_invalid = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_future_invalid = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=(-1, -2)) == 0
    return neighbor_current_invalid | neighbor_future_invalid


def _fit_joint_bezier_gt(
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    ego_current: torch.Tensor,
    neighbors_current: torch.Tensor,
    bezier_degree: int,
    coeff_norm: BezierStateNormalizer,
    time_horizon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build joint diffusion targets for ego + neighbors in Bezier coefficient space.

    Returns:
        all_gt: [B, 1+Pn, 4+coeff_dim] — current pose + normalized Bezier coeffs per agent
        current_states: [B, 1+Pn, 4]
        neighbor_invalid: [B, Pn] — True where neighbor slot is empty (ego never masked)
    """
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
    joint_xy = torch.cat(
        [ego_future[:, None, :, :2], neighbors_future[..., :2]],
        dim=1,
    )  # [B, 1+Pn, T, 2]
    control_points = fit_bezier_control_points(
        joint_xy,
        degree=bezier_degree,
        time_horizon=time_horizon,
        start_xy=current_states[..., :2],
    )
    joint_coeffs = control_points_to_coeffs(control_points)  # [B, 1+Pn, coeff_dim]
    neighbor_invalid = _neighbor_agent_invalid_mask(neighbors_current, neighbors_future)

    joint_coeffs_norm = coeff_norm(joint_coeffs)
    joint_coeffs_norm[:, 1:][neighbor_invalid] = 0.0

    all_gt = build_flat_state(current_states, joint_coeffs_norm)
    return all_gt, current_states, neighbor_invalid


def _bezier_coeff_l2_loss(
    pred_flat: torch.Tensor,
    all_gt: torch.Tensor,
    coeff_dim: int,
    z: torch.Tensor,
    std: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    """Per-agent mean L2 (MSE) on Bezier coefficient dims, shape [B, P]."""
    _, future_gt = split_flat_state(all_gt, coeff_dim)
    _, pred_coeffs = split_flat_state(pred_flat, coeff_dim)

    if model_type == "score":
        return ((pred_coeffs * std + z) ** 2).mean(dim=-1)
    if model_type == "x_start":
        return ((pred_coeffs - future_gt) ** 2).mean(dim=-1)
    raise ValueError(f"Unknown model_type: {model_type}")


def _bezier_waypoint_l2_loss(
    pred_coeffs_norm: torch.Tensor,
    joint_gt_xy: torch.Tensor,
    current_states: torch.Tensor,
    coeff_norm: BezierStateNormalizer,
    bezier_degree: int,
    num_samples: int,
    trajectory_time_horizon: float,
) -> torch.Tensor:
    """
    Per-agent mean L2 on (x,y) sampled at physical times T.

    Args:
        pred_coeffs_norm: [B, P, coeff_dim] model x_start (normalized coeffs)
        joint_gt_xy: [B, P, T, 2] expert positions
    Returns:
        [B, P]
    """
    num_ctrl = bezier_degree_to_num_control_points(bezier_degree)
    pred_coeffs_phys = coeff_norm.inverse(pred_coeffs_norm)
    control_points = coeffs_to_control_points(pred_coeffs_phys, num_ctrl)
    control_points = pin_bezier_start(control_points, current_states[..., :2])
    pred_xy = sample_bezier_trajectory(
        control_points,
        num_samples,
        degree=bezier_degree,
        time_horizon=trajectory_time_horizon,
    )
    return ((pred_xy - joint_gt_xy) ** 2).mean(dim=(-2, -1))


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],
    futures: Tuple[torch.Tensor, torch.Tensor],
    norm: StateNormalizer,
    loss: Dict[str, Any],
    model_type: str,
    eps: float = 1e-3,
    use_bezier: bool = False,
    bezier_degree: int = 6,
    coeff_norm: BezierStateNormalizer = None,
    trajectory_time_horizon: float = 8.0,
    alpha_bezier_waypoint_loss: float = 0.5,
    debug_context: dict = None,
):
    ego_future, neighbors_future, neighbor_future_mask = futures

    B, Pn, T, _ = neighbors_future.shape
    ego_current = inputs["ego_current_state"][:, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]

    if use_bezier:
        assert coeff_norm is not None
        all_gt, current_states, neighbor_invalid = _fit_joint_bezier_gt(
            ego_future,
            neighbors_future,
            ego_current,
            neighbors_current,
            bezier_degree,
            coeff_norm,
            trajectory_time_horizon,
        )
        coeff_dim = all_gt.shape[-1] - 4
        P = all_gt.shape[1]
        assert P == 1 + Pn, f"expected 1 ego + {Pn} neighbors, got P={P}"

        t = torch.rand(B, device=all_gt.device) * (1 - eps) + eps
        z = torch.randn(B, P, coeff_dim, device=all_gt.device)

        future_gt = all_gt[..., 4:]
        mean, std = marginal_prob(future_gt, t)
        std = std.view(-1, *([1] * (len(future_gt.shape) - 1)))

        xT_future = mean + std * z
        xT = build_flat_state(current_states, xT_future)

        merged_inputs = {
            **inputs,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "use_bezier": True,
            "bezier_degree": bezier_degree,
        }

        _, decoder_output = model(merged_inputs)
        score = decoder_output["score"]
        assert score.shape[-1] == 4 + coeff_dim, (
            f"DiT must predict current(4)+Bezier coeffs({coeff_dim}) per agent, got {score.shape[-1]}"
        )

        coeff_l2 = _bezier_coeff_l2_loss(score, all_gt, coeff_dim, z, std, model_type)

        joint_gt_xy = torch.cat(
            [ego_future[:, None, :, :2], neighbors_future[..., :2]],
            dim=1,
        )
        _, pred_coeffs_norm = split_flat_state(score, coeff_dim)
        if model_type == "x_start":
            waypoint_l2 = _bezier_waypoint_l2_loss(
                pred_coeffs_norm,
                joint_gt_xy,
                current_states,
                coeff_norm,
                bezier_degree,
                T,
                trajectory_time_horizon,
            )
        else:
            waypoint_l2 = torch.zeros_like(coeff_l2)

        per_agent_loss = coeff_l2 + alpha_bezier_waypoint_loss * waypoint_l2

        neighbors_future_valid = ~neighbor_invalid
        masked_neighbor_coeff = coeff_l2[:, 1:][neighbors_future_valid]
        masked_neighbor_waypoint = waypoint_l2[:, 1:][neighbors_future_valid]
        masked_prediction_loss = per_agent_loss[:, 1:][neighbors_future_valid]

        if masked_prediction_loss.numel() > 0:
            loss["neighbor_bezier_coeff_loss"] = masked_neighbor_coeff.mean()
            loss["neighbor_bezier_waypoint_loss"] = masked_neighbor_waypoint.mean()
            loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
        else:
            zero = torch.tensor(0.0, device=coeff_l2.device)
            loss["neighbor_bezier_coeff_loss"] = zero
            loss["neighbor_bezier_waypoint_loss"] = zero
            loss["neighbor_prediction_loss"] = zero

        loss["ego_bezier_coeff_loss"] = coeff_l2[:, EGO_AGENT_INDEX].mean()
        loss["ego_bezier_waypoint_loss"] = waypoint_l2[:, EGO_AGENT_INDEX].mean()
        loss["ego_planning_loss"] = per_agent_loss[:, EGO_AGENT_INDEX].mean()
        assert not torch.isnan(per_agent_loss).sum(), "loss cannot be nan"

        if debug_context is not None:
            maybe_log_bezier_training_debug(
                enabled=debug_context.get("enabled", False),
                prob=float(debug_context.get("prob", 0.0)),
                rank=int(debug_context.get("rank", 0)),
                debug_dir=debug_context.get("debug_dir"),
                epoch=int(debug_context.get("epoch", 0)),
                batch_idx=int(debug_context.get("batch_idx", 0)),
                diffusion_t=t.detach(),
                ego_dpm_loss=per_agent_loss.detach(),
                loss=loss,
                all_gt=all_gt.detach(),
                score=score.detach(),
                ego_future_xy=ego_future[..., :2].detach(),
                current_states=current_states.detach(),
                coeff_norm=coeff_norm,
                coeff_dim=coeff_dim,
                bezier_degree=bezier_degree,
                trajectory_time_horizon=trajectory_time_horizon,
                future_len=int(debug_context.get("future_len", T)),
                model_type=model_type,
                alpha_planning_loss=float(debug_context.get("alpha_planning_loss", 1.0)),
            )

        return loss, decoder_output

    # --- legacy waypoint diffusion ---
    neighbors_future_valid = ~neighbor_future_mask
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1)
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps
    z = torch.randn_like(gt_future, device=gt_future.device)

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))

    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs)
    score = decoder_output["score"][:, :, 1:, :]

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z) ** 2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :]) ** 2, dim=-1)

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]
    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()
    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output
