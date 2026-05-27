import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.bezier_utils import (
    BezierStateNormalizer,
    bezier_degree_to_num_control_points,
    bezier_num_control_points_to_coeff_dim,
    bezier_trajectory_with_heading,
    build_flat_state,
    coeffs_to_control_points,
    pin_bezier_start,
    split_flat_state,
)
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._use_bezier = getattr(config, "use_bezier", False)
        self._bezier_degree = getattr(config, "bezier_degree", 6)
        self._trajectory_time_horizon = float(
            getattr(config, "trajectory_time_horizon", 8.0)
        )
        self._sde = VPSDE_linear()

        if self._use_bezier:
            self._num_control_points = bezier_degree_to_num_control_points(self._bezier_degree)
            self._coeff_dim = bezier_num_control_points_to_coeff_dim(self._num_control_points)
            output_dim = 4 + self._coeff_dim
            self._state_normalizer: BezierStateNormalizer = config.bezier_state_normalizer
        else:
            self._coeff_dim = None
            output_dim = (config.future_len + 1) * 4
            self._state_normalizer: StateNormalizer = config.state_normalizer

        self.dit = DiT(
            sde=self._sde,
            route_encoder=RouteEncoder(
                config.route_num,
                config.lane_len,
                drop_path_rate=config.encoder_drop_path_rate,
                hidden_dim=config.hidden_dim,
            ),
            depth=config.decoder_depth,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type,
        )

        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._guidance_fn = getattr(config, "guidance_fn", None)

    @property
    def sde(self):
        return self._sde

    def _pin_current_state(self, xt, current_states):
        B, P, _ = current_states.shape
        xt = xt.reshape(B, P, -1)
        xt[..., :4] = current_states
        return xt.reshape(B, P, -1)

    def _decode_bezier_prediction(self, flat_x0, current_states):
        _, coeff_norm = split_flat_state(flat_x0, self._coeff_dim)
        coeff_phys = self._state_normalizer.inverse(coeff_norm)
        control_points = coeffs_to_control_points(coeff_phys, self._num_control_points)
        control_points = pin_bezier_start(control_points, current_states[..., :2])
        return bezier_trajectory_with_heading(
            control_points,
            self._future_len,
            degree=self._bezier_degree,
            time_horizon=self._trajectory_time_horizon,
        )

    def forward(self, encoder_outputs, inputs):
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, : self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        ego_neighbor_encoding = encoder_outputs["encoding"]
        route_lanes = inputs["route_lanes"]

        if self.training:
            sampled_trajectories = inputs["sampled_trajectories"]
            diffusion_time = inputs["diffusion_time"]
            score = self.dit(
                sampled_trajectories,
                diffusion_time,
                ego_neighbor_encoding,
                route_lanes,
                neighbor_current_mask,
            )
            if self._use_bezier:
                # [B, P, 4+coeff_dim]: joint Bezier diffusion for ego (idx 0) + neighbors (idx 1..P-1)
                assert score.shape[-1] == 4 + self._coeff_dim
                return {"score": score}
            return {"score": score.reshape(B, P, -1, 4)}

        if self._use_bezier:
            noise = torch.randn(B, P, self._coeff_dim, device=current_states.device) * 0.5
            xT = build_flat_state(current_states, noise)

            def initial_state_constraint(xt, t, step):
                return self._pin_current_state(xt, current_states)

            x0 = dpm_sampler(
                self.dit,
                xT,
                other_model_params={
                    "cross_c": ego_neighbor_encoding,
                    "route_lanes": route_lanes,
                    "neighbor_current_mask": neighbor_current_mask,
                },
                dpm_solver_params={"correcting_xt_fn": initial_state_constraint},
                model_wrapper_params={
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit,
                        "model_condition": {
                            "cross_c": ego_neighbor_encoding,
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask,
                        },
                        "inputs": {
                            **inputs,
                            "use_bezier": True,
                            "bezier_degree": self._bezier_degree,
                            "trajectory_time_horizon": self._trajectory_time_horizon,
                        },
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer,
                        "future_len": self._future_len,
                        "trajectory_time_horizon": self._trajectory_time_horizon,
                    },
                    "guidance_scale": 0.5,
                    "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
                },
            )
            prediction = self._decode_bezier_prediction(x0, current_states)
            return {"prediction": prediction, "bezier_coeffs": split_flat_state(x0, self._coeff_dim)[1]}

        xT = torch.cat(
            [
                current_states[:, :, None],
                torch.randn(B, P, self._future_len, 4, device=current_states.device) * 0.5,
            ],
            dim=2,
        ).reshape(B, P, -1)

        def initial_state_constraint(xt, t, step):
            xt = xt.reshape(B, P, -1, 4)
            xt[:, :, 0, :] = current_states
            return xt.reshape(B, P, -1)

        x0 = dpm_sampler(
            self.dit,
            xT,
            other_model_params={
                "cross_c": ego_neighbor_encoding,
                "route_lanes": route_lanes,
                "neighbor_current_mask": neighbor_current_mask,
            },
            dpm_solver_params={"correcting_xt_fn": initial_state_constraint},
            model_wrapper_params={
                "classifier_fn": self._guidance_fn,
                "classifier_kwargs": {
                    "model": self.dit,
                    "model_condition": {
                        "cross_c": ego_neighbor_encoding,
                        "route_lanes": route_lanes,
                        "neighbor_current_mask": neighbor_current_mask,
                    },
                    "inputs": inputs,
                    "observation_normalizer": self._observation_normalizer,
                    "state_normalizer": self._state_normalizer,
                },
                "guidance_scale": 0.5,
                "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
            },
        )
        x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]
        return {"prediction": x0}


class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.0)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.0)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x

        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        super().__init__()

        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.0)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        B, P, _ = x.shape

        x = self.preproj(x)

        x_embedding = torch.cat(
            [self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)],
            dim=0,
        )
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)
        x = x + x_embedding

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask

        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        x = self.final_layer(x, y)

        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
