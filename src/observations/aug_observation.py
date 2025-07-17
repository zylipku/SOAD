from collections.abc import Sequence

import torch
from torch import nn

from .observation import Observation


class AugmentedObservation(nn.Module):
    def __init__(
        self,
        observation: Observation,  # no mask here
        aug_channel_idxs: Sequence[int],
        stride: int,
        spatial_mask: torch.Tensor,
    ) -> None:
        super().__init__()

        self.observation = observation
        self.aug_channel_idxs = aug_channel_idxs
        self.stride = stride
        self.spatial_mask = spatial_mask

    def state_to_obs(self, x: torch.Tensor) -> torch.Tensor:
        return self.observation.func(x[..., :: self.stride, :, :, :])[..., self.spatial_mask]

    def aug_get_obs(self, x_aug: torch.Tensor) -> torch.Tensor:
        return x_aug[..., :: self.stride, self.aug_channel_idxs, :, :][..., self.spatial_mask]

    def aug_set_obs(self, x_aug: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        y_of_x_aug = x_aug[..., :: self.stride, self.aug_channel_idxs, :, :]
        y_of_x_aug[..., self.spatial_mask] = values
        x_aug[..., :: self.stride, self.aug_channel_idxs, :, :] = y_of_x_aug
        return x_aug

    @classmethod
    def observations_from_config(cls, config) -> Sequence["AugmentedObservation"]:
        return [
            cls(observation=obs, aug_channel_idxs=channel_idxs, stride=stride, spatial_mask=spatial_mask)
            for obs, channel_idxs, stride, spatial_mask in zip(
                config.observations, config.obs_channel_idxs, config.time_strides, config.rand_masks, strict=False
            )
        ]
