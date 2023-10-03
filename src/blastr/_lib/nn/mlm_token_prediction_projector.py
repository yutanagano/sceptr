from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Linear, Module


NUM_AMINO_ACIDS = 20


class MlmTokenPredictionProjector(ABC, Module):
    @abstractmethod
    def forward(self, token_embeddings: Tensor) -> Tensor:
        pass


class AminoAcidTokenProjector(MlmTokenPredictionProjector):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.projector = Linear(in_features=d_model, out_features=NUM_AMINO_ACIDS)

    def forward(self, token_embeddings: Tensor) -> Tensor:
        return self.projector.forward(token_embeddings)
