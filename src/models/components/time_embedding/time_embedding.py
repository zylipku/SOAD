import abc

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        """
        Args:
            t (torch.Tensor): `.shape = (bs,)`

        Returns:
            torch.Tensor: `.shape = (bs, emb_dim)`
        """
        super().__init__()

        self.embed_dim = embed_dim

    @abc.abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): `.shape = (bs,)`

        Returns:
            torch.Tensor: `.shape = (bs, embed_dim)`
        """
        return NotImplemented
