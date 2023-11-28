from abc import ABC, abstractmethod
from torch import Tensor

from sceptr._lib.nn.self_attention_stack import SelfAttentionStack


class VectorRepresentationDelegate(ABC):
    @abstractmethod
    def __init__(self, self_attention_stack: SelfAttentionStack) -> None:
        pass

    @abstractmethod
    def get_vector_representations_of(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        pass
