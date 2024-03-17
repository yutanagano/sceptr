import torch
from torch import FloatTensor, LongTensor

from sceptr._lib.nn.data.tokeniser.token_indices import (
    CdrCompartmentIndex,
)
from sceptr._lib.nn.token_embedder import TokenEmbedder
from sceptr._lib.nn.token_embedder.simple_relative_position_embedding import (
    SimpleRelativePositionEmbedding,
)
from sceptr._lib.nn.token_embedder.one_hot_token_index_embedding import OneHotTokenIndexEmbedding
from sceptr._lib.nn.token_embedder.blosum_embedding import (
    BlosumEmbedding,
)


class CdrBlosumEmbedder(TokenEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._blosum_embedding = BlosumEmbedding()
        self._position_embedding = SimpleRelativePositionEmbedding()
        self._compartment_embedding = OneHotTokenIndexEmbedding(CdrCompartmentIndex)

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self._blosum_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self._position_embedding.forward(tokenised_tcrs[:, :, 1:3])
        compartment_component = self._compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )
        all_components_stacked = torch.concatenate(
            [token_component, position_component, compartment_component], dim=-1
        )
        return all_components_stacked
