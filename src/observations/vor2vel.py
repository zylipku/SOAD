import torch
from einops import rearrange

from .observation import Observation


class Vor2Vel(Observation):
    name: str = "vor2vel"

    normalizer = torch.tensor([1.2e4, 1.6e4, 1.1e4, 1.6e4])

    def __init__(
        self,
        L: float = 1e6,
        W: float = None,
        nx: int = 256,
        ny: int = None,
        rd=15000.0,
        delta=0.25,
        H1=500,
        U1=0.025,
        U2=0.0,
        rand_mask=None,
        sigma: float = 0.0,
    ):
        super().__init__(sigma)

        self.rand_mask = rand_mask

        self.L = L
        self.W = W if W is not None else L
        self.nx = nx
        self.ny = nx if ny is None else ny

        self.rd = rd
        self.delta = delta
        self.H1 = H1
        self.U1 = U1
        self.U2 = U2

        self.F1 = rd**-2 / (1 + delta)
        self.F2 = delta * self.F1

        kk = torch.fft.rfftfreq(self.nx, d=(self.L / (2 * torch.pi * self.nx)))
        ll = torch.fft.fftfreq(self.ny, d=(self.W / (2 * torch.pi * self.ny)))

        k, l = torch.meshgrid(kk, ll, indexing="xy")  # noqa: E741
        self.wv2 = k**2 + l**2
        self.ik = 1j * kk
        self.il = 1j * ll

    def func(self, x: torch.Tensor) -> torch.Tensor:
        qh = self.q2qh(x)
        ph = self.qh2ph(qh)
        uv = self.ph2uv(ph, size=x.shape[-2:])
        uv = uv / self.normalizer[:, None, None].to(uv)
        if self.rand_mask is None:
            return uv
        else:
            return uv[..., self.rand_mask]

    def q2qh(self, q: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfftn(q, dim=(-2, -1))

    def qh2ph(self, qh: torch.Tensor) -> torch.Tensor:
        qh_shape = qh.shape
        qh = rearrange(qh, "... c h w -> ... (h w) c 1")
        coeff_mat = torch.empty(*qh.shape[:-2], 2, 2)

        coeff_mat[..., :, 0, 0] = -(self.wv2 + self.F1).reshape(-1)
        coeff_mat[..., :, 0, 1] = self.F1
        coeff_mat[..., :, 1, 0] = self.F2
        coeff_mat[..., :, 1, 1] = -(self.wv2 + self.F2).reshape(-1)

        ph_tail = torch.linalg.solve(coeff_mat[..., 1:, :, :].to(qh), qh[..., 1:, :, :])
        # (..., h*w-1, c, 1)
        ph = torch.nn.functional.pad(ph_tail, (0, 0, 0, 0, 1, 0), mode="constant", value=0)
        ph = rearrange(ph, "... (h w) c 1 -> ... c h w", h=qh_shape[-2], w=qh_shape[-1])

        return ph

    def ph2uv(self, ph: torch.Tensor, size=None) -> torch.Tensor:
        # Set ph to zero (skip, recompute fresh from sum below)
        # invert qh to find ph
        # calculate spectral velocities
        uh = -self.il[:, None].to(ph) * ph
        vh = self.ik[None, :].to(ph) * ph

        u = torch.fft.irfftn(uh, dim=(-2, -1), s=size)
        v = torch.fft.irfftn(vh, dim=(-2, -1), s=size)

        uv = torch.cat([u, v], -3)

        return uv
