import torch

from .observation import Observation


class Arctan(Observation):
    name: str = "arctan3x"

    def __init__(self, scale: float = 1.0, rand_mask=None, sigma: float = 0.0):
        super().__init__(sigma)

        self.scale = scale
        self.rand_mask = rand_mask

    def func(self, x: torch.Tensor) -> torch.Tensor:
        if self.rand_mask is None:
            return torch.arctan(self.scale * x)
        else:
            return torch.arctan(self.scale * x)[..., self.rand_mask]
