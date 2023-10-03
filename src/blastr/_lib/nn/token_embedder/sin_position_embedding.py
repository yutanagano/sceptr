import torch
from torch import Tensor
from torch.nn import Module


class SinPositionEmbedding(Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, sin_scale_factor: int = 30
    ) -> None:
        self._check_embedding_dim_is_even(embedding_dim)
        super().__init__()

        position_embedding_matrix = self._compute_position_embedding_matrix(
            num_embeddings, embedding_dim, sin_scale_factor
        )
        self.register_buffer("position_embedding_matrix", position_embedding_matrix)

    def forward(self, token_indices: Tensor) -> Tensor:
        return self.position_embedding_matrix[token_indices]

    def _check_embedding_dim_is_even(self, embedding_dim: int) -> None:
        if embedding_dim % 2 != 0:
            raise RuntimeError("embedding_dim must be even")

    def _compute_position_embedding_matrix(
        self, num_embeddings: int, embedding_dim: int, sin_scale_factor: int
    ) -> Tensor:
        num_embeddings_to_represent_including_padding = num_embeddings + 1
        position_embedding_matrix = torch.zeros(
            num_embeddings_to_represent_including_padding, embedding_dim
        )

        position_index_column_vector = torch.arange(0, num_embeddings).unsqueeze(1)
        wavelength_scaling_row_vector = self._compute_wavelength_scaling_row_vector(
            embedding_dim, sin_scale_factor
        )

        position_embedding_matrix[1:, 0::2] = torch.sin(
            position_index_column_vector / wavelength_scaling_row_vector
        )
        position_embedding_matrix[1:, 1::2] = torch.cos(
            position_index_column_vector / wavelength_scaling_row_vector
        )

        return position_embedding_matrix

    def _compute_wavelength_scaling_row_vector(
        self, embedding_dim: int, sin_scale_factor: int
    ) -> Tensor:
        scaling_strength_for_every_pair_of_embedding_dims = (
            torch.arange(0, embedding_dim, 2) / embedding_dim
        )
        scaling_factor_for_every_pair_of_embedding_dims = torch.pow(
            sin_scale_factor, scaling_strength_for_every_pair_of_embedding_dims
        )

        return scaling_factor_for_every_pair_of_embedding_dims
