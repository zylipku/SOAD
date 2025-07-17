import math
from collections.abc import Sequence

import torch
from torch import nn


class VPSDEScheduler(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
    """

    def __init__(
        self,
        alpha_type: str = "cosine",
        # eps: nn.Module,
        # shape: torch.Size,
        eta: float = 1e-3,
    ):
        super().__init__()

        # self.eps = eps
        # self.shape = shape
        # self.dims = tuple(range(-len(shape), 0))
        self.alpha_type = alpha_type
        self.eta = eta

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.alpha_type == "linear":
            return 1 - (1 - self.eta) * t
        elif self.alpha_type == "cosine":
            return torch.cos(math.acos(math.sqrt(self.eta)) * t) ** 2
        elif self.alpha_type == "exponential":
            return torch.exp(math.log(self.eta) * t**2)
        else:
            raise ValueError

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta**2).sqrt()

    def sample_timesteps(self, *size: Sequence[int]) -> torch.Tensor:
        return torch.rand(size=size)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)

        t = t.reshape(t.shape + (1,) * (x.ndim - 1))

        x_noised = self.alpha(t) * x + self.sigma(t) * noise

        return x_noised, noise

    def sample(
        self,
        x1: torch.Tensor,
        # score_estimator: nn.Module = None,
        noise_estimator: nn.Module = None,
        score_likelihood: nn.Module = None,
        n_steps: int = 64,
        n_corrections: int = 0,
        tau: float = 1.0,
        **score_kwargs,
    ) -> torch.Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        # assert score_estimator or noise_estimator

        # def noise_func(xt: torch.Tensor, t: torch.Tensor, *kwargs) -> torch.Tensor:

        #     if score_estimator is None:
        #         noise = noise_estimator(xt, t, *kwargs)
        #     else:
        #         noise = -self.sigma(t) * score_estimator(xt, t, *kwargs)

        #     if score_likelihood:
        #         noise += - self.sigma(t) * score_likelihood(xt, t, *kwargs)

        #     return noise.detach()

        def sigma_s(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            sigma_sx = noise_estimator(x, t)
            if score_likelihood is None:
                return sigma_sx
            else:
                sigma_sy = -self.sigma(t) * score_likelihood(x, t)
                return sigma_sx + sigma_sy

        steps = torch.linspace(1.0, 0.0, n_steps + 1).to(x1.device)

        x = x1

        cached_sigma_s = None

        for t, prev_t in zip(steps[:-1], steps[1:], strict=False):
            # Predictor
            r = self.alpha(prev_t) / self.alpha(t)

            if cached_sigma_s is None:
                cached_sigma_s = sigma_s(x, t)

            x = r * x + (self.sigma(prev_t) - r * self.sigma(t)) * cached_sigma_s

            # Corrector
            cached_sigma_s = sigma_s(x, prev_t, **score_kwargs)
            for _ in range(n_corrections):
                z = torch.randn_like(x)
                noise = cached_sigma_s
                delta = tau / noise.square().mean(dim=tuple(range(1, noise.ndim)), keepdim=True)

                x = x - (delta * noise + torch.sqrt(2 * delta) * z) * self.sigma(prev_t)

        return x
