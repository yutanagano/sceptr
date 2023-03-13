"""
Modules to embed various TCR tokens.
"""


import math
import torch
from torch import Tensor
from torch.nn import Module


class SinPositionEmbedding(Module):
    """
    Module to encode positional embeddings via a stacked sinusoidal function.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, sin_scale_factor: int = 30
    ) -> None:
        assert embedding_dim % 2 == 0
        self._embedding_dim = embedding_dim

        super().__init__()

        # 0th dim size is num_embeddings+1 to account for fact that 0 is null value (positions are 1-indexed)
        position_embedding = torch.zeros(num_embeddings + 1, embedding_dim)
        position_indices = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(
            -math.log(sin_scale_factor)
            * torch.arange(0, embedding_dim, 2)
            / embedding_dim
        )
        position_embedding[1:, 0::2] = torch.sin(position_indices * div_term)
        position_embedding[1:, 1::2] = torch.cos(position_indices * div_term)

        self.register_buffer("position_embedding", position_embedding)

    def forward(self, x: Tensor) -> Tensor:
        return self.position_embedding[x]


class SinPositionEmbeddingRelative(Module):
    """
    Stacked sinusoidal position embedding but for relative position (position
    here is a float value between 1 and 2 where 1 is the beginning and 2 is the
    end).
    """

    def __init__(self, embedding_dim: int, sin_scale_factor: int) -> None:
        assert embedding_dim % 2 == 0
        self._embedding_dim = embedding_dim

        super().__init__()

        phase_vector = torch.exp(
            math.log(math.pi)
            + math.log(sin_scale_factor) * torch.arange(embedding_dim / 2)
        )
        self.register_buffer("phase_vector", phase_vector)
        self._device = None

    def forward(self, x: Tensor) -> Tensor:
        if self._device is None:
            self._device = self.phase_vector.device

        padding = x[..., 0] == 0
        x = (x[..., 0] - 1) / (x[..., 1] - 1)

        x[torch.isnan(x)] = 0.5

        position_embedding = torch.zeros(
            *x.size(), self._embedding_dim, device=self._device
        )

        position_embedding[..., 0::2] = torch.sin(x.unsqueeze(-1) * self.phase_vector)
        position_embedding[..., 1::2] = torch.cos(x.unsqueeze(-1) * self.phase_vector)
        position_embedding[padding, ...] = 0

        return position_embedding


class SinPositionEmbeddingBiDirectional(Module):
    """
    Bidirectional positioning.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, sin_scale_factor: int = 30
    ) -> None:
        assert embedding_dim % 4 == 0
        self._embedding_dim = embedding_dim

        super().__init__()

        # 0th dim size is num_embeddings+1 to account for fact that 0 is null value (positions are 1-indexed)
        position_embedding = torch.zeros(num_embeddings + 1, int(embedding_dim / 2))
        position_indices = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(
            -math.log(sin_scale_factor)
            * torch.arange(0, embedding_dim / 2, 2)
            / embedding_dim
            / 2
        )
        position_embedding[1:, 0::2] = torch.sin(position_indices * div_term)
        position_embedding[1:, 1::2] = torch.cos(position_indices * div_term)

        self.register_buffer("position_embedding", position_embedding)

    def forward(self, x: Tensor) -> Tensor:
        forward_embs = self.position_embedding[x[..., 0]]
        backward_embs = self.position_embedding[x[..., 1] - x[..., 0] + 1]
        combined = torch.concat([forward_embs, backward_embs], dim=-1)
        combined[x[..., 0] == 0] = 0  # zero out padding

        return combined
