from enum import IntEnum
import torch
from torch import FloatTensor, LongTensor
from torch.nn import Module


class OneHotTokenIndexEmbedding(Module):
    def __init__(self, token_index: IntEnum) -> None:
        super().__init__()
        self._register_token_embeddings(token_index)

    def _register_token_embeddings(self, token_index: IntEnum) -> FloatTensor:
        num_tokens = len(token_index)
        num_tokens_excluding_null = num_tokens - 1

        null_embedding = torch.zeros((1, num_tokens_excluding_null))
        non_null_token_embeddings = torch.eye(num_tokens_excluding_null)
        token_embeddings = torch.concatenate(
            [null_embedding, non_null_token_embeddings], dim=0
        )

        self.register_buffer("_token_embeddings", token_embeddings)

    def forward(self, token_indices: LongTensor) -> FloatTensor:
        return self._token_embeddings[token_indices]
