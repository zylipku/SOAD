from collections.abc import Sequence

import torch
from torch import nn

from ..time_embedding import SDAMLPTimeEmbedding
from .seq2d import Sequential5d, TemporalBlock, WithTimeEmbed


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, dim: int = -1, **kwargs) -> None:
        super().__init__()

        self.normalized_shape = normalized_shape
        self.dim = dim

        self.norm = nn.LayerNorm(normalized_shape, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(self.dim, -1)
        x = self.norm(x)
        x = x.transpose(self.dim, -1)

        return x


class SpatialResidualBlock(WithTimeEmbed):
    def __init__(self, channels: int, time_embed_dim: int, **conv_kwargs) -> None:
        super().__init__()

        self.channels = channels

        self.time_proj = nn.Sequential(
            # (-1, context_dim)
            nn.Linear(time_embed_dim, channels),
            # (-1, c)
            nn.Unflatten(-1, (-1, 1, 1)),
            # (-1, c, 1, 1)
        )
        self.seq = nn.Sequential(
            # (-1, c, h, w)
            LayerNorm(channels, dim=-3),
            # (-1, c, h, w)
            nn.Conv2d(channels, channels << 2, **conv_kwargs),
            # (-1, 4c, h, w)
            nn.SiLU(),
            nn.Conv2d(channels << 2, channels, kernel_size=1),
            # (-1, c, h, w)
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        z = x + self.time_proj(time_embed)
        z = x + self.seq(z)

        return z


class ModulatedBlock(WithTimeEmbed):
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        n_blocks: int,
        **conv_kwargs,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [SpatialResidualBlock(channels, time_embed_dim, **conv_kwargs) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, time_embed=time_embed)

        return x


class UNetSDA(nn.Module):
    r"""Creates a U-Net with modulation.

    References:
        | U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        | https://arxiv.org/abs/1505.04597

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        mod_features: The number of modulation features.
        hidden_channels: The number of hidden channels.
        hidden_blocks: The number of hidden blocks at each depth.
        kernel_size: The size of the convolution kernels.
        stride: The stride of the downsampling convolutions.
        activation: The activation function constructor.
        spatial: The number of spatial dimensions. Can be either 1, 2 or 3.
        kwargs: Keyword arguments passed to :class:`nn.Conv2d`.
    """

    conv_dim = 2

    def __init__(
        self,
        channels: int,
        out_channels: int = None,
        time_embed_dim: int = 64,
        time_embed_method: str = "sdamlp",
        hidden_channels: Sequence[int] = (16, 32, 64),
        hidden_n_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: int = 5,
        padding_mode: str = "circular",
    ) -> None:
        """
        Args:
            x (torch.Tensor): `.shape = (-1, in_channels, ...)`
            context (torch.Tensor): `.shape = (-1, context_dim)`

        Returns:
            torch.Tensor: `.shape = (-1, out_channels, ...)`
        """
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels

        # time embedding
        self.time_embed_method = time_embed_method
        if time_embed_method == "sdamlp":
            self.time_embedder = SDAMLPTimeEmbedding(time_embed_dim)
        else:
            raise ValueError(f"Unknown time embedding method: {self.time_embed_method}")

        # conv. configurations
        kernel_size = 5
        conv_kwargs = {
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "padding_mode": padding_mode,
        }

        # Layers
        self.dns = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, n_blocks in enumerate(hidden_n_blocks):
            # conv-in
            self.dns.append(
                Sequential5d(
                    nn.Conv2d(self.channels, hidden_channels[i], **conv_kwargs),
                    ModulatedBlock(hidden_channels[i], time_embed_dim=time_embed_dim, n_blocks=n_blocks, **conv_kwargs),
                )
                if i == 0
                else Sequential5d(
                    nn.Conv2d(hidden_channels[i - 1], hidden_channels[i], stride=2, **conv_kwargs),
                    TemporalBlock(channels=hidden_channels[i]),
                    ModulatedBlock(hidden_channels[i], time_embed_dim=time_embed_dim, n_blocks=n_blocks, **conv_kwargs),
                )
            )

            self.ups.append(
                Sequential5d(
                    ModulatedBlock(hidden_channels[i], time_embed_dim=time_embed_dim, n_blocks=n_blocks, **conv_kwargs),
                    nn.Conv2d(hidden_channels[i], self.channels, **conv_kwargs),
                )
                if i == 0
                else Sequential5d(
                    ModulatedBlock(hidden_channels[i], time_embed_dim=time_embed_dim, n_blocks=n_blocks, **conv_kwargs),
                    TemporalBlock(channels=hidden_channels[i]),
                    LayerNorm(hidden_channels[i], dim=-3),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(hidden_channels[i], hidden_channels[i - 1], **conv_kwargs),
                )
            )

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): `.shape = (-1, in_channels, ...)`
            context (torch.Tensor): `.shape = (-1, context_dim)`

        Returns:
            torch.Tensor: `.shape = (-1, out_channels, ...)`
        """
        b, L, c, h, w = xt.shape

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.reshape(-1).to(xt)

        x = xt
        time_embed = self.time_embedder(t)
        # * (-1,) -> (-1, D)

        skips = []

        for layer in self.dns:
            x = layer(x, time_embed=time_embed)
            skips.append(x)

        skips.pop()

        for layer in reversed(self.ups):
            x = layer(x, time_embed=time_embed)

            if len(skips):
                x = x + skips.pop()

        return x
