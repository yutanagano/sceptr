from torch import FloatTensor, LongTensor
from torch.nn import Module

from src.nn.data.tokeniser.token_indices import DefaultTokenIndex


class SimpleRelativePositionEmbedding(Module):
    def forward(self, position_indices: LongTensor) -> FloatTensor:
        """
        Input tensor should have shape (..., 2) with first of the final two dimensions encoding token position, and the final dimension encoding compartment length.
        """
        null_mask = position_indices[..., 0] == DefaultTokenIndex.NULL
        zero_indexed_token_positions = position_indices[..., 0] - 1
        compartment_length_minus_one = position_indices[..., 1] - 1

        relative_token_positions = (
            zero_indexed_token_positions / compartment_length_minus_one
        )

        RELATIVE_POSITION_IF_ONLY_ONE_TOKEN_IN_COMPARTMENT = 0.5
        relative_token_positions[
            relative_token_positions.isnan()
        ] = RELATIVE_POSITION_IF_ONLY_ONE_TOKEN_IN_COMPARTMENT
        relative_token_positions[null_mask] = 0

        relative_token_positions = relative_token_positions.unsqueeze(dim=-1)

        return relative_token_positions
