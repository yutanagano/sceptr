'''
CDR123 embedding modules.
'''


import math
from .sinpos import (
    SinPositionEmbedding,
    SinPositionEmbeddingBiDirectional
)
from torch import Tensor
from torch.nn import Embedding, Module


class BCDREmbedding(Module):
    '''
    CDR embedder for beta-chain only models.

    Compatible tokenisers: BCDRTokeniser
    '''


    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.token_embedding = Embedding(
            num_embeddings=23, # <pad> + <mask> + <cls> + 20 amino acids
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.position_embedding = SinPositionEmbedding(
            num_embeddings=100,
            embedding_dim=embedding_dim
        )
        self.compartment_embedding = Embedding(
            num_embeddings=4, # <pad>, CDR1, CDR2, CDR3
            embedding_dim=embedding_dim,
            padding_idx=0
        )
    

    def forward(self, x: Tensor) -> Tensor:
        return \
            (
                self.token_embedding(x[:,:,0]) +
                self.position_embedding(x[:,:,1]) +
                self.compartment_embedding(x[:,:,3])
            ) * math.sqrt(self.embedding_dim)
    

class BCDREmbeddingBDPos(BCDREmbedding):
    '''
    CDR embedder for beta only models with bidirectional position embedding.

    Compatible tokenisers: BCDRTokeniser
    '''

    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)

        self.position_embedding = SinPositionEmbeddingBiDirectional(
            num_embeddings=100,
            embedding_dim=embedding_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        return \
            (
                self.token_embedding(x[:,:,0]) +
                self.position_embedding(x[:,:,1:3]) +
                self.compartment_embedding(x[:,:,3])
            ) * math.sqrt(self.embedding_dim)