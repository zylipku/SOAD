import abc

import torch
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint


class WithTimeEmbed(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Sequential5d(nn.Sequential, WithTimeEmbed):
    def forward(self, x: torch.Tensor, time_embed: torch.Tensor = None) -> torch.Tensor:
        for module in self:
            if isinstance(module, TemporalBlock):
                x = module(x)
            elif isinstance(module, WithTimeEmbed):
                assert time_embed is not None
                bx, L, *_ = x.shape
                bt, D = time_embed.shape

                assert bx == bt

                time_embed_flatten = repeat(time_embed, "b D -> (b L) D", L=L)
                x = rearrange(x, "b L ... -> (b L) ...")
                x = module(x, time_embed=time_embed_flatten)
                x = rearrange(x, "(b L) ... -> b L ...", b=bx, L=L)

            else:
                bx, L, *_ = x.shape
                x = rearrange(x, "b L ... -> (b L) ...")
                x = module(x)
                x = rearrange(x, "(b L) ... -> b L ...", b=bx, L=L)

        return x


class TemporalBlock(nn.Module):
    r"""Creates a temporal-wise residual block for raster sequences."""

    def __init__(self, channels: int, use_checkpoint: bool = False):
        super().__init__()

        self.channels = channels
        self.use_checkpoint = use_checkpoint

        self.conv = nn.Sequential(  # (-1, c, h, w)
            nn.Conv1d(channels, 4 * channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(4 * channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape = (-1, L, 1+npatches, embed_dim)
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, L, c, h, w = x.shape

        z = rearrange(x, "b L c h w -> (b h w) c L")
        z = self.conv(z)
        z = rearrange(z, "(b h w) c L -> b L c h w", b=b, h=h, w=w)

        return x + z
