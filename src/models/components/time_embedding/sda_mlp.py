import torch
from torch import nn

from .time_embedding import TimeEmbedding


class SDAMLPTimeEmbedding(TimeEmbedding):
    def __init__(self, embed_dim: int) -> None:
        """
        Args:
            t (torch.Tensor): `.shape = (bs,)`

        Returns:
            torch.Tensor: `.shape = (bs, emb_dim)`
        """
        super().__init__(embed_dim=embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(32, 256),
            nn.SiLU(),
            nn.Linear(256, self.embed_dim),
        )

        self.nfreq = 16
        self.freqs = nn.Parameter(torch.pi * torch.arange(1, self.nfreq + 1), requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): `.shape = (bs,)`

        Returns:
            torch.Tensor: `.shape = (bs, emb_dim)`
        """
        t = self.freqs * t[:, None]  # (-1, nfreq)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        embed = self.mlp(t)

        return embed
