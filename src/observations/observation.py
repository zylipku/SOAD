from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


class Observation(nn.Module, ABC):
    def __init__(self, sigma: float = 0.0):
        super().__init__()

        self.sigma = sigma

    @abstractmethod
    def func(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented

    def dfdx(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        return \nabla_x H(x) @ v

        """
        return NotImplemented

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.func(x)
        y = y + self.sigma * torch.randn_like(y)
        return ObservationOutput(operator=self, result=y)

    def log_likelihood(self, y: torch.Tensor, x: torch.Tensor):
        return -0.5 * torch.sum((y - self.func(x)) ** 2, dim=-1) / self.sigma**2


class SeqObservation(nn.Module, ABC):
    def __init__(self, observation: Observation, time_stride: int = 1, dim=-4) -> None:
        super().__init__()

        self.observation = observation
        self.time_stride = time_stride
        self.dim = dim

    def func(self, x: torch.Tensor) -> torch.Tensor:
        slices = [
            slice(None),
        ] * x.ndim
        slices[self.dim] = slice(None, None, self.time_stride)
        return self.observation.func(x[slices])


@dataclass
class ObservationOutput:
    operator: Observation
    result: torch.Tensor

    def __str__(self):
        return f"operator: {self.operator.__class__.__name__}" + f"\nresult.shape: {self.result.shape}"
