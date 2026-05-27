"""
Bezier curve utilities for DiffusionPlanner.

Trajectory representation:
  - Plane curve: parametric Bezier B(u) = (x(u), y(u)), u in [0, 1].
  - Physical time T (seconds): expert / planner future samples at
    T_k = k * (time_horizon / num_samples), k = 1..num_samples
    (matches nuPlan transform_predictions_to_states).
  - u(T) = T / time_horizon.

Diffusion timestep t in [0, 1] is unrelated to physical time T.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from diffusion_planner.utils.train_utils import openjson


def bezier_degree_to_num_control_points(degree: int) -> int:
    return degree + 1


def bezier_num_control_points_to_coeff_dim(num_control_points: int) -> int:
    return num_control_points * 2


def trajectory_step_interval(time_horizon: float, num_samples: int) -> float:
    """Seconds between consecutive future poses (nuPlan: horizon / num_poses)."""
    return time_horizon / num_samples


def trajectory_physical_times(
    num_samples: int,
    time_horizon: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Physical timestamps T for each future pose [num_samples].
    Same as np.arange(0, time_horizon, step) + step  -> 0.1, 0.2, ..., 8.0 for 80@8s.
    """
    step = trajectory_step_interval(time_horizon, num_samples)
    return torch.arange(1, num_samples + 1, device=device, dtype=dtype) * step


def physical_time_to_curve_parameter(times: Tensor, time_horizon: float) -> Tensor:
    """Map physical time T [s] to Bezier parameter u = T / time_horizon."""
    return (times / time_horizon).clamp(0.0, 1.0)


def bernstein_basis(u: Tensor, degree: int) -> Tensor:
    """
    Args:
        u: [...] curve parameter in [0, 1]
        degree: Bezier degree (e.g. 6)
    Returns:
        [..., degree + 1]
    """
    n = degree
    u = u.clamp(0.0, 1.0)
    one_minus_u = 1.0 - u
    basis = []
    for i in range(n + 1):
        comb = math.comb(n, i)
        basis.append(comb * (one_minus_u ** (n - i)) * (u ** i))
    return torch.stack(basis, dim=-1)


def pin_bezier_start(control_points: Tensor, start_xy: Tensor) -> Tensor:
    """Pin P0 so B(u=0) = start_xy (current agent position)."""
    pinned = control_points.clone()
    pinned[..., 0, :] = start_xy
    return pinned


def fit_bezier_control_points(
    xy: Tensor,
    degree: int = 6,
    time_horizon: float = 8.0,
    start_xy: Optional[Tensor] = None,
) -> Tensor:
    """
    Least-squares fit of expert (x, y) at physical times T to a Bezier curve.

    When ``start_xy`` is provided, P0 is fixed to the current agent (x, y) and only
    P1..Pn are fitted.

    Args:
        xy: [..., T, 2] positions in ego-centric frame (T poses = num future samples)
        degree: Bezier degree (6 -> 7 control points)
        time_horizon: future horizon in seconds (default 8.0)
        start_xy: optional [..., 2] curve start (current agent xy)
    Returns:
        [..., degree + 1, 2] control points
    """
    assert xy.shape[-1] == 2
    num_ctrl = degree + 1
    num_samples = xy.shape[-2]

    times = trajectory_physical_times(
        num_samples, time_horizon, xy.device, xy.dtype
    )
    u = physical_time_to_curve_parameter(times, time_horizon)
    basis = bernstein_basis(u, degree)  # [T, num_ctrl]

    lead_shape = xy.shape[:-2]
    xy_flat = xy.reshape(-1, num_samples, 2)
    batch = xy_flat.shape[0]

    if start_xy is None:
        basis_b = basis.unsqueeze(0).expand(batch, -1, -1)
        solution = torch.linalg.lstsq(basis_b, xy_flat).solution
        return solution.reshape(*lead_shape, num_ctrl, 2)

    start_flat = start_xy.reshape(-1, 2)
    if num_ctrl <= 1:
        return start_flat.reshape(*lead_shape, num_ctrl, 2)

    fixed = torch.einsum("t,bi->bti", basis[:, 0], start_flat)
    target = xy_flat - fixed

    if num_ctrl == 2:
        inner = torch.linalg.lstsq(
            basis[:, 1:2].unsqueeze(0).expand(batch, -1, -1),
            target.contiguous(),
        ).solution
        control_points = torch.cat([start_flat[:, None, :], inner], dim=1)
        return control_points.reshape(*lead_shape, num_ctrl, 2)

    inner_basis = basis[:, 1:].unsqueeze(0).expand(batch, -1, -1).contiguous()
    inner = torch.linalg.lstsq(inner_basis, target.contiguous()).solution
    control_points = torch.cat([start_flat[:, None, :], inner], dim=1)
    return control_points.reshape(*lead_shape, num_ctrl, 2)


def control_points_to_coeffs(control_points: Tensor) -> Tensor:
    """[..., num_ctrl, 2] -> [..., num_ctrl * 2] with (x0,y0,x1,y1,...)."""
    return control_points.reshape(*control_points.shape[:-2], -1)


def coeffs_to_control_points(coeffs: Tensor, num_control_points: int) -> Tensor:
    """[..., coeff_dim] -> [..., num_ctrl, 2]."""
    return coeffs.reshape(*coeffs.shape[:-1], num_control_points, 2)


def sample_bezier_trajectory(
    control_points: Tensor,
    num_samples: int,
    degree: int = 6,
    time_horizon: float = 8.0,
) -> Tensor:
    """
    Sample (x, y) on the Bezier curve at physical times T (not diffusion time).

    Args:
        control_points: [..., num_ctrl, 2]
        num_samples: number of future poses (e.g. 80)
        time_horizon: future horizon in seconds (e.g. 8.0)
    Returns:
        [..., num_samples, 2]
    """
    device = control_points.device
    dtype = control_points.dtype
    times = trajectory_physical_times(num_samples, time_horizon, device, dtype)
    u = physical_time_to_curve_parameter(times, time_horizon)
    basis = bernstein_basis(u, degree)  # [S, num_ctrl]
    return torch.einsum("sn,...nd->...sd", basis, control_points)


def bezier_derivative_xy_wrt_physical_time(
    control_points: Tensor,
    num_samples: int,
    degree: int = 6,
    time_horizon: float = 8.0,
) -> Tensor:
    """
    d(x,y)/dT at each physical sample time.

    Returns:
        [..., num_samples, 2]
    """
    n = degree
    device = control_points.device
    dtype = control_points.dtype
    times = trajectory_physical_times(num_samples, time_horizon, device, dtype)
    u = physical_time_to_curve_parameter(times, time_horizon)

    delta = control_points[..., 1:, :] - control_points[..., :-1, :]
    deriv_ctrl = n * delta

    basis = bernstein_basis(u, n - 1)
    dxy_du = torch.einsum("sn,...nd->...sd", basis, deriv_ctrl)
    return dxy_du / time_horizon


def bezier_trajectory_with_heading(
    control_points: Tensor,
    num_samples: int,
    degree: int = 6,
    time_horizon: float = 8.0,
) -> Tensor:
    """
    Sample planner trajectory poses from predicted Bezier control points.

    Args:
        control_points: [..., num_ctrl, 2]
    Returns:
        [..., num_samples, 4] as (x, y, cos, sin) at physical times T
    """
    xy = sample_bezier_trajectory(control_points, num_samples, degree, time_horizon)
    dxy = bezier_derivative_xy_wrt_physical_time(
        control_points, num_samples, degree, time_horizon
    )
    heading = torch.atan2(dxy[..., 1], dxy[..., 0])
    return torch.cat(
        [xy, torch.cos(heading).unsqueeze(-1), torch.sin(heading).unsqueeze(-1)],
        dim=-1,
    )


def build_flat_state(current_states: Tensor, coeff_states: Tensor) -> Tensor:
    """
    current_states: [B, P, 4]
    coeff_states: [B, P, coeff_dim]
    Returns:
        [B, P, 4 + coeff_dim]
    """
    return torch.cat([current_states, coeff_states], dim=-1)


def split_flat_state(flat: Tensor, coeff_dim: int) -> Tuple[Tensor, Tensor]:
    return flat[..., :4], flat[..., 4 : 4 + coeff_dim]


class BezierStateNormalizer:
    """Normalize Bezier control-point coefficients (x,y interleaved)."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    @classmethod
    def from_json(cls, args) -> "BezierStateNormalizer":
        data = openjson(args.normalization_file_path)
        num_ctrl = bezier_degree_to_num_control_points(args.bezier_degree)
        coeff_dim = bezier_num_control_points_to_coeff_dim(num_ctrl)

        def _agent_coeff_stats(key: str):
            xy_mean = data[key]["mean"][:2]
            xy_std = data[key]["std"][:2]
            agent_mean = []
            agent_std = []
            for _ in range(num_ctrl):
                agent_mean.extend(xy_mean)
                agent_std.extend(xy_std)
            assert len(agent_mean) == coeff_dim
            return agent_mean, agent_std

        ego_mean, ego_std = _agent_coeff_stats("ego")
        nb_mean, nb_std = _agent_coeff_stats("neighbor")
        mean = [ego_mean] + [nb_mean] * args.predicted_neighbor_num
        std = [ego_std] + [nb_std] * args.predicted_neighbor_num
        return cls(mean, std)

    def __call__(self, coeffs: Tensor) -> Tensor:
        mean = self.mean.to(coeffs.device).view(1, -1, coeffs.shape[-1])
        std = self.std.to(coeffs.device).view(1, -1, coeffs.shape[-1])
        return (coeffs - mean) / std

    def inverse(self, coeffs: Tensor) -> Tensor:
        mean = self.mean.to(coeffs.device).view(1, -1, coeffs.shape[-1])
        std = self.std.to(coeffs.device).view(1, -1, coeffs.shape[-1])
        return coeffs * std + mean

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }
