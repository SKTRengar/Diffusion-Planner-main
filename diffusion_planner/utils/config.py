import json
import os
import torch

from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.utils.bezier_utils import BezierStateNormalizer


def _resolve_normalization_file_path(args_file: str, args_dict: dict) -> str:
    path = args_dict.get("normalization_file_path")
    if path and os.path.isfile(path):
        return path

    args_dir = os.path.dirname(os.path.abspath(args_file))
    candidates = [
        os.path.join(args_dir, "normalization.json"),
        os.path.join(os.path.dirname(args_dir), "normalization.json"),
        os.path.join(os.path.dirname(os.path.dirname(args_dir)), "normalization.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    if path:
        return path
    return os.path.join(os.path.dirname(os.path.dirname(args_dir)), "normalization.json")


class Config:
    
    def __init__(
            self,
            args_file,
            guidance_fn
    ):
        self.args_file = args_file
        with open(args_file, 'r') as f:
            args_dict = json.load(f)
            
        for key, value in args_dict.items():
            setattr(self, key, value)

        self.normalization_file_path = _resolve_normalization_file_path(args_file, args_dict)

        self.use_bezier = getattr(self, "use_bezier", False)
        self.bezier_degree = getattr(self, "bezier_degree", 6)
        self.trajectory_time_horizon = float(getattr(self, "trajectory_time_horizon", 8.0))

        if self.use_bezier:
            saved_bezier = args_dict.get("bezier_state_normalizer")
            if isinstance(saved_bezier, dict) and "mean" in saved_bezier:
                self.bezier_state_normalizer = BezierStateNormalizer.from_dict(saved_bezier)
            else:
                self.bezier_state_normalizer = BezierStateNormalizer.from_json(self)
            self.state_normalizer = self.bezier_state_normalizer
        else:
            if isinstance(self.state_normalizer, dict):
                self.state_normalizer = StateNormalizer(
                    self.state_normalizer['mean'], self.state_normalizer['std']
                )
            else:
                self.state_normalizer = StateNormalizer.from_json(self)

        if isinstance(self.observation_normalizer, dict):
            first_val = next(iter(self.observation_normalizer.values()), None)
            if isinstance(first_val, dict) and "mean" in first_val:
                if isinstance(first_val["mean"], list):
                    self.observation_normalizer = ObservationNormalizer.from_json(self)
                else:
                    self.observation_normalizer = ObservationNormalizer({
                        k: {
                            'mean': torch.as_tensor(v['mean']),
                            'std': torch.as_tensor(v['std'])
                        } for k, v in self.observation_normalizer.items()
                    })
            else:
                self.observation_normalizer = ObservationNormalizer.from_json(self)
        
        self.guidance_fn = guidance_fn