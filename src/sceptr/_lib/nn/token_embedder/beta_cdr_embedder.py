import math
from torch import Tensor
from torch.nn import Embedding

from sceptr._lib.nn.data.tokeniser.token_indices import (
    DefaultTokenIndex,
    AminoAcidTokenIndex,
    BetaCdrCompartmentIndex,
)
from sceptr._lib.nn.token_embedder.token_embedder import TokenEmbedder
from sceptr._lib.nn.token_embedder.sin_position_embedding import SinPositionEmbedding


MAX_PLAUSIBLE_CDR_LENGTH = 100
VOCABULARY_SIZE = len(AminoAcidTokenIndex)
NUMBER_OF_COMPARTMENTS = len(BetaCdrCompartmentIndex)


class BetaCdrEmbedder(TokenEmbedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim

        self.token_embedding = Embedding(
            num_embeddings=VOCABULARY_SIZE,
            embedding_dim=embedding_dim,
            padding_idx=DefaultTokenIndex.NULL,
        )
        self.position_embedding = SinPositionEmbedding(
            num_embeddings=MAX_PLAUSIBLE_CDR_LENGTH, embedding_dim=embedding_dim
        )
        self.compartment_embedding = Embedding(
            num_embeddings=VOCABULARY_SIZE,
            embedding_dim=embedding_dim,
            padding_idx=DefaultTokenIndex.NULL,
        )

    def forward(self, tokenised_tcrs: Tensor) -> Tensor:
        token_component = self.token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self.position_embedding.forward(tokenised_tcrs[:, :, 1])
        compartment_component = self.compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )

        all_components_summed = (
            token_component + position_component + compartment_component
        )

        return all_components_summed * math.sqrt(self._embedding_dim)
