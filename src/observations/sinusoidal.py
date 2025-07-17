import torch

from .observation import Observation


class Sinusoidal(Observation):
    name: str = "sinusoidal"

    def __init__(self, freq: float = 1.0, rand_mask=None, sigma: float = 0.0):
        super().__init__(sigma)
        self.freq = freq
        self.rand_mask = rand_mask

    def func(self, x: torch.Tensor) -> torch.Tensor:
        if self.rand_mask is None:
            return 1.5 * torch.sin(self.freq * x)
        else:
            return 1.5 * torch.sin(self.freq * x)[..., self.rand_mask]
