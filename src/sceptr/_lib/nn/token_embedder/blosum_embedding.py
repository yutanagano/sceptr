import blosum
import itertools
from sceptr._lib.nn.data.tokeniser.token_indices import DefaultTokenIndex
from sceptr._lib.schema import AminoAcid
import torch
from torch import FloatTensor, LongTensor
from torch.nn import Module, Embedding


class BlosumEmbedding(Module):
    def __init__(self) -> None:
        super().__init__()

        self._special_token_embeddings = Embedding(
            num_embeddings=len(DefaultTokenIndex),
            embedding_dim=len(AminoAcid),
            padding_idx=DefaultTokenIndex.NULL
        )
        self._register_blosum_embeddings()

    def _register_blosum_embeddings(self) -> None:
        blosum_matrix = blosum.BLOSUM(62)

        null_embedding = torch.zeros(size=(len(DefaultTokenIndex),len(AminoAcid)))
        aa_embeddings = torch.zeros(size=(len(AminoAcid),len(AminoAcid)))
        
        for row, column in itertools.product(AminoAcid, repeat=2):
            blosum_score = blosum_matrix[row.name][column.name]
            aa_embeddings[row.value,column.value] = blosum_score

        blosum_embeddings = torch.concatenate(
            [null_embedding, aa_embeddings], dim=0
        )
        blosum_embeddings_normalised = blosum_embeddings / blosum_embeddings.abs().max()

        self.register_buffer("_blosum_embeddings", blosum_embeddings_normalised)

    def forward(self, token_indices: LongTensor) -> FloatTensor:
        special_token_mask = token_indices < len(DefaultTokenIndex)
        token_indices_aa_masked_out = token_indices * special_token_mask

        special_token_embeddings = self._special_token_embeddings.forward(token_indices_aa_masked_out)
        aa_blosum_embeddings = self._blosum_embeddings[token_indices]

        return special_token_embeddings + aa_blosum_embeddings
