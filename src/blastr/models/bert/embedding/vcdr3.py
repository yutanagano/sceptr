'''
VCDR3 embedding modules.
'''


import math
from .sinpos import SinPositionEmbedding
from torch import Tensor
from torch.nn import Embedding, Module


class BVCDR3Embedding(Module):
    '''
    VCDR3 embedder for beta-chain only models.

    Compatible tokenisers: BVCDR3Tokeniser
    '''
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.token_embedding = Embedding(
            num_embeddings=71, # <pad> + <mask> + <cls> + 20 amino acids + 48 V genes
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.position_embedding = SinPositionEmbedding(
            num_embeddings=100,
            embedding_dim=embedding_dim
        )
        self.compartment_embedding = Embedding(
            num_embeddings=3, # <pad>, V gene, CDR3
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