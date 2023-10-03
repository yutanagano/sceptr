from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class TokenEmbedder(ABC, Module):
    @abstractmethod
    def forward(self, tokenised_tcrs: Tensor) -> Tensor:
        pass
