"""
Model wrappers to export default forward procedure for a particular training.
"""


from src.models.embedder import _MLMEmbedder
from torch import Tensor
from torch.nn import Module


class MLMModelWrapper(Module):
    def __init__(self, embedder: _MLMEmbedder) -> None:
        super().__init__()
        self.embedder = embedder

    def forward(self, masked: Tensor) -> Tensor:
        return self.embedder.mlm(masked)


class CLModelWrapper(Module):
    def __init__(self, embedder: _MLMEmbedder) -> None:
        super().__init__()
        self.embedder = embedder

    def forward(self, x: Tensor, x_prime: Tensor, masked: Tensor) -> tuple:
        z = self.embedder.embed(x)
        z_prime = self.embedder.embed(x_prime)
        mlm_logits = self.embedder.mlm(masked)

        return z, z_prime, mlm_logits
