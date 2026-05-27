import os

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson, opendata


def subsample_data_list(data_list, subset_size: int, seed: int):
    """
    Randomly sample subset_size entries without replacement.
    Uses a fixed seed so all DDP ranks see the same subset.
    """
    total = len(data_list)
    if subset_size is None or subset_size <= 0 or subset_size >= total:
        return list(data_list), total, total

    rng = np.random.RandomState(seed)
    indices = rng.choice(total, size=subset_size, replace=False)
    subset = [data_list[int(i)] for i in indices]
    return subset, subset_size, total


class DiffusionPlannerData(Dataset):
    def __init__(
        self,
        data_dir,
        data_list,
        past_neighbor_num,
        predicted_neighbor_num,
        future_len,
        subset_size=None,
        subset_seed=3407,
    ):
        self.data_dir = data_dir
        full_list = openjson(data_list)
        self.data_list, self.subset_num, self.total_num = subsample_data_list(
            full_list, subset_size, subset_seed
        )
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))

        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future']

        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]

        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects = data['static_objects']

        data = {
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects,
        }

        return tuple(data.values())