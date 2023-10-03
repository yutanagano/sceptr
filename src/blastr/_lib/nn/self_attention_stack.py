from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Linear, Module, TransformerEncoder, TransformerEncoderLayer


class SelfAttentionStack(ABC, Module):
    @property
    @abstractmethod
    def d_model(self) -> int:
        pass

    @abstractmethod
    def forward(self, token_embeddings: Tensor, padding_mask: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_token_embeddings_at_penultimate_layer(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        pass


class SelfAttentionStackWithBuiltins(SelfAttentionStack):
    d_model: int = None

    def __init__(
        self, num_layers: int, d_model: int, nhead: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self._num_layers_in_stack = num_layers

        self_attention_block = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self._self_attention_stack = TransformerEncoder(
            encoder_layer=self_attention_block, num_layers=num_layers
        )

    def forward(self, token_embeddings: Tensor, padding_mask: Tensor) -> Tensor:
        return self._self_attention_stack.forward(
            src=token_embeddings, src_key_padding_mask=padding_mask
        )

    def get_token_embeddings_at_penultimate_layer(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        penultimate_layer_index = self._num_layers_in_stack - 1

        for layer in self._self_attention_stack.layers[:penultimate_layer_index]:
            token_embeddings = layer.forward(
                src=token_embeddings, src_key_padding_mask=padding_mask
            )

        return token_embeddings


class SelfAttentionStackWithInitialProjection(SelfAttentionStack):
    d_model: int = None

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self._initial_projector = Linear(
            in_features=embedding_dim, out_features=d_model, bias=False
        )
        self._standard_stack = SelfAttentionStackWithBuiltins(
            num_layers=num_layers, d_model=d_model, nhead=nhead, dropout=dropout
        )

    def forward(self, token_embeddings: Tensor, padding_mask: Tensor) -> Tensor:
        projected_embeddings = self._initial_projector.forward(token_embeddings)
        return self._standard_stack.forward(projected_embeddings, padding_mask)

    def get_token_embeddings_at_penultimate_layer(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        projected_embeddings = self._initial_projector.forward(token_embeddings)
        return self._standard_stack.get_token_embeddings_at_penultimate_layer(
            projected_embeddings, padding_mask
        )
