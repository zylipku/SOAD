import functools
import math
from collections.abc import Sequence

import jax
import numpy as np
import pyqg_jax
import torch
from einops import asnumpy
from jax import numpy as jnp
from tqdm.contrib import tzip

from ..diffusion.scheduler import VPSDEScheduler
from ..observations import AugmentedObservation
from .test_configs import TestCaseConfig


def QGforward(q: torch.Tensor) -> torch.Tensor:
    normalize_std = torch.tensor([9.7e-6, 1.19e-6])
    base_model = pyqg_jax.qg_model.QGModel(nx=256, precision=pyqg_jax.state.Precision.DOUBLE)
    dt = 15 * 60  # 15 min
    stepper = pyqg_jax.steppers.AB3Stepper(dt=dt)
    stepped_model = pyqg_jax.steppers.SteppedModel(base_model, stepper)
    dummy_state = base_model.create_initial_state(jax.random.key(0))
    init_state = dummy_state.update(q=(q * normalize_std[:, None, None]).cpu().numpy().astype(jnp.float64))
    init_state = stepped_model.initialize_stepper_state(init_state)

    @functools.partial(jax.jit, static_argnames=["num_steps"])
    def roll_out_state(state, num_steps):
        def loop_fn(carry, _x):
            current_state = carry
            next_state = stepped_model.step_model(current_state)
            # Note: we output the current state for ys
            # This includes the starting step in the trajectory
            return next_state, current_state

        _final_carry, traj_steps = jax.lax.scan(loop_fn, state, None, length=num_steps)
        return traj_steps

    final_state = roll_out_state(init_state, num_steps=24 * 60 * 60 // dt)

    out = np.array(final_state.state.q[-1])
    print(out.shape)
    return (torch.from_numpy(out) / normalize_std[:, None, None]).to(q)


class Assimilator:
    def __init__(
        self,
        config: TestCaseConfig,
        device: torch.device,
        has_background_prior: bool,
        unknown_background_noise: bool = False,
        observation_distribution: str = "gaussian",
        lognormal_scale: float = 1.0,
        use_fdc: bool = True,
    ) -> None:
        self.config = config
        self.device = device
        self.has_background_prior = has_background_prior
        self.unknown_background_noise = unknown_background_noise

        self.opHs = config.observations
        self.time_strides = config.time_strides
        self.aug_observations = AugmentedObservation.observations_from_config(config)
        self.model = config.model

        # assimilation config
        self.sigma_b = 0.1
        self.sigma_o = 0.1

        if observation_distribution.lower() in ["normal", "gaussian"]:
            self.obs_noise_shift = 0.0
            self.obs_noise_generator = torch.distributions.Normal(loc=0.0, scale=self.sigma_o)
        elif observation_distribution.lower() in ["laplace"]:
            self.obs_noise_shift = 0.0
            self.obs_noise_generator = torch.distributions.Laplace(
                loc=0.0,
                scale=self.sigma_o / math.sqrt(2),
            )
        elif observation_distribution.lower() in ["uniform"]:
            self.obs_noise_shift = 0.0
            self.obs_noise_generator = torch.distributions.Uniform(
                low=-self.sigma_o * math.sqrt(3),
                high=self.sigma_o * math.sqrt(3),
            )
        elif observation_distribution.lower() in ["lognormal", "log-normal"]:
            s = lognormal_scale
            self.obs_noise_shift = -self.sigma_o / math.expm1(s**2) ** 0.5
            self.obs_noise_generator = torch.distributions.LogNormal(
                loc=-(s**2) / 2 + math.log(-self.obs_noise_shift),
                scale=lognormal_scale,
            )
        else:
            raise ValueError(f"Unknown observation distribution: {observation_distribution}")
        print(f"Using {self.obs_noise_generator} as observation noise generator.")

        self.scheduler = VPSDEScheduler()
        self.use_fdc = use_fdc if "soad" in config.method else False

        testloader = config.datamodule.test_dataloader()
        x_ref: torch.Tensor = next(iter(testloader))[0]
        if self.has_background_prior:
            if self.unknown_background_noise:
                x_prev = x_ref[0, 0]
                x_ref = x_ref[:, 1:]
                x_b = QGforward(x_prev + torch.randn_like(x_prev) * 1e-2)
                x_b = x_b[None, ...]
            else:
                x_b = x_ref[..., 0, :, :, :].clone().detach()
                x_b += torch.randn_like(x_b) * self.sigma_b  # additional Gaussian
        else:
            x_b = None
        self.x_ref = x_ref
        self.x_b = x_b

        self.ass_output = None

        self.n_steps = 256
        self.gamma = 0.1
        self.n_corrections = 5
        self.tau = 0.25

    def move_to_cuda(self) -> None:
        for opH in self.aug_observations:
            opH = opH.to(self.device)
        self.model = self.model.to(self.device)
        self.x_ref = self.x_ref.to(self.device)
        self.x_b = None if self.x_b is None else self.x_b.to(self.device)

    def get_obs(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # x_ref.shape: (b, L, c, h, w)

        y_os = []
        x_ms = []
        for opH in self.aug_observations:
            y_o = opH.state_to_obs(self.x_ref)
            y_o += self.obs_noise_generator.sample(y_o.shape).to(y_o)  # additional Gaussian
            y_o += self.obs_noise_shift
            y_os.append(y_o.detach())

            if "soad" in self.config.method:
                x_m = torch.zeros(*self.x_ref.shape[:-3], self.model.channels, 256, 256).to(y_o) + torch.nan
                x_m = opH.aug_set_obs(x_m, y_o)
                x_ms.append(x_m.detach())

        return y_os, x_ms

    def run(self, enable_progress_bar=True) -> dict[str, np.ndarray]:
        self.move_to_cuda()
        y_os, x_ms = self.get_obs()
        x_b = self.x_b

        ass_inputs = {
            "x_ms": x_ms,
            "y_os": y_os,
            "x_b": x_b,
        }

        x_a = self._assimilate(ass_inputs, enable_progress_bar=enable_progress_bar)

        ass_out = {
            "x_r": asnumpy(self.x_ref),
            "x_ms": [asnumpy(x_m) for x_m in x_ms],
            "y_os": [asnumpy(y_o) for y_o in y_os],
            "x_b": x_b if x_b is None else asnumpy(x_b),
            "x_a": asnumpy(x_a),
        }
        self.ass_output = ass_out

        return ass_out

    def aug2obs(self, x_aug: torch.Tensor) -> Sequence[torch.Tensor]:
        y_os = [opH.aug_get_obs(x_aug) for opH in self.aug_observations]

        return y_os

    def _sda_score_likelihood(self, x: torch.Tensor, t: float, y_os: Sequence[torch.Tensor]) -> torch.Tensor:
        sigma = self.scheduler.sigma(t)
        mu = self.scheduler.mu(t)
        rt = sigma / mu

        with torch.enable_grad():
            xt = x.detach().requires_grad_(True)
            L = 0.0
            x_h = (x - sigma * self.model(xt, t)) / mu
            for y_o, opH in zip(y_os, self.aug_observations, strict=False):
                L = L + (mu * y_o - mu * opH.state_to_obs(x_h)).square().sum()
            (s,) = torch.autograd.grad(L, xt)

        coeff = (rt**2 / 2.0) / (self.sigma_o**2 + self.gamma * rt**2)
        coeff.clamp_max_(1.0 / s.abs().max())

        return coeff * s.detach()

    def _soad_score_likelihood(self, x: torch.Tensor, t: float, y_os: Sequence[torch.Tensor]) -> torch.Tensor:
        sigma = self.scheduler.sigma(t)
        mu = self.scheduler.mu(t)
        rt = sigma / mu

        with torch.enable_grad():
            xt = x.detach().requires_grad_(True)
            L = 0.0
            for y_o, xt2obs, model2obs in zip(y_os, self.aug2obs(xt), self.aug2obs(self.model(xt, t)), strict=False):
                L = L + (mu * y_o - xt2obs + sigma * model2obs).square().sum()
            (s,) = torch.autograd.grad(L, xt)

        sigma_x = 1.0
        coeff = (rt**2 / 2.0) / (self.sigma_o**2 + sigma_x**2 * rt**2 / (sigma_x**2 + rt**2))

        # regularization for coeff
        coeff.clamp_max_(1.0 / s.abs().max())
        # or
        # coeff = torch.arctan(.1 * coeff) / .1

        # gamma = .1
        # coeff = (rt**2 / 2.) / (self.sigma_o**2 + gamma * rt**2)
        # coeff = (sigma**2 / 2.) / (self.sigma_o**2 * mu**2 + gamma * sigma**2)

        return coeff * s.detach()

    def _fd_corrector(self, x: torch.Tensor, t: float, y_os: Sequence[torch.Tensor], x_b: torch.Tensor) -> torch.Tensor:
        sigma = self.scheduler.sigma(t)
        mu = self.scheduler.mu(t)
        rt = sigma / mu

        if rt > self.sigma_o:
            for y_o, opH in zip(y_os, self.aug_observations, strict=False):
                y_o_perturbed = y_o + torch.randn_like(y_o) * (rt**2 - self.sigma_o**2) ** 0.5
                x = opH.aug_set_obs(x_aug=x, values=mu * y_o_perturbed)

        if rt > self.sigma_b and self.has_background_prior:
            x_b_perturbed = x_b + torch.randn_like(x_b) * (rt**2 - self.sigma_b**2) ** 0.5
            x[..., 0, :2, :, :] = mu * x_b_perturbed

        return x

    def score_likelihood(self, x: torch.Tensor, t: float, y_os: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.config.method == "sda":
            return self._sda_score_likelihood(x, t, y_os)
        elif "soad" in self.config.method:
            return self._soad_score_likelihood(x, t, y_os)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def corrector(self, x: torch.Tensor, t: float, y_os: Sequence[torch.Tensor], x_b: torch.Tensor) -> torch.Tensor:
        return self._fd_corrector(x, t, y_os, x_b) if self.use_fdc else x

    def _assimilate(self, ass_inputs: dict[str, torch.Tensor], enable_progress_bar: bool = True) -> torch.Tensor:
        y_os = ass_inputs["y_os"]
        # x_ms = to_assimilate["x_ms"]
        x_b = ass_inputs["x_b"]

        noise_estimator = self.model

        def pi_theta(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            prior = self.scheduler.sigma(t) * noise_estimator(x, t)
            likelihood = self.score_likelihood(x, t, y_os)
            return (prior + likelihood).detach()

        with torch.no_grad():
            # x_ref.shape: (bs, L, c, H, W)
            bs, L, _, H, W = self.x_ref.shape
            x1 = torch.randn(bs, L, self.model.channels, H, W, device=self.device)

            steps = torch.linspace(1.0, 0.0, self.n_steps + 1, device=self.device)

            x = x1
            cached_pi = None

            zipper = tzip if enable_progress_bar else zip
            for t, prev_t in zipper(steps[:-1], steps[1:]):
                if torch.nonzero(torch.isnan(x)).shape[0] > 0:
                    print(f"NaN detected at step {t}")
                    break

                # Predictor
                r_mu = self.scheduler.alpha(prev_t) / self.scheduler.alpha(t)
                r_sigma = self.scheduler.sigma(prev_t) / self.scheduler.sigma(t)

                if cached_pi is None:
                    cached_pi = pi_theta(x, t)
                x = r_mu * x + (r_sigma - r_mu) * cached_pi

                x = self.corrector(x, prev_t, y_os, x_b)

                # Corrector
                cached_pi = pi_theta(x, prev_t)
                for _ in range(self.n_corrections):
                    z = torch.randn_like(x)
                    noise = cached_pi / self.scheduler.sigma(prev_t)
                    delta = self.tau / noise.square().mean(dim=tuple(range(1, noise.ndim)), keepdim=True)

                    x = x - (delta * noise + torch.sqrt(2 * delta) * z) * self.scheduler.sigma(prev_t)

                    x = self.corrector(x, prev_t, y_os, x_b)

        return x[..., :2, :, :]

    def evaluate(self, ass_output: dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        if ass_output is None:
            ass_output = self.ass_output

        x_r = ass_output["x_r"]
        x_a = ass_output["x_a"]
        x_r = x_r.reshape(-1, *x_r.shape[-3:])
        x_a = x_a.reshape(-1, *x_a.shape[-3:])

        x_r = x_r.reshape(x_r.shape[0], -1)
        x_a = x_a.reshape(x_a.shape[0], -1)
        rmse = ((x_a - x_r) ** 2).mean(axis=-1) ** 0.5

        return {"rmse": rmse}
