import torch
from torch import FloatTensor, LongTensor

from blastr._lib.nn.data.tokeniser.token_indices import AminoAcidTokenIndex
from blastr._lib.nn.token_embedder import TokenEmbedder
from blastr._lib.nn.token_embedder.simple_relative_position_embedding import (
    SimpleRelativePositionEmbedding,
)
from blastr._lib.nn.token_embedder.one_hot_token_index_embedding import (
    OneHotTokenIndexEmbedding,
)


class BetaCdr3SimpleEmbedder(TokenEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._token_embedding = OneHotTokenIndexEmbedding(AminoAcidTokenIndex)
        self._position_embedding = SimpleRelativePositionEmbedding()

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self._token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self._position_embedding.forward(tokenised_tcrs[:, :, 1:3])
        all_components_stacked = torch.concatenate(
            [token_component, position_component], dim=-1
        )
        return all_components_stacked