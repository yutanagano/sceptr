"""
BERT templates.
"""


from ..embedder import _MLMEmbedder
import torch
from torch import Tensor
from torch.nn.functional import normalize
from typing import Tuple


def masked_average_pool(x: Tensor, padding_mask: Tensor) -> Tensor:
    """
    Given a tensor x, representing a batch of length-padded token embedding
    sequences, and a corresponding padding mask tensor, compute the average
    pooled vector for every sequence in the batch with the padding taken into
    account.

    :param x: Tensor representing a batch of length-padded token embedding
        sequences, where indices of tokens are marked with 1s and indices of
        paddings are marked with 0s (size B,L,E)
    :type x: torch.Tensor
    :param padding_mask: Tensor representing the corresponding padding mask
        (size B,L)
    :type padding_mask: torch.Tensor
    :return: Average pool of token embeddings per sequence (size B,E)
    """
    # Reverse the boolean values of the mask to mark where the tokens are, as
    # opposed to where the tokens are not. Then, resize padding mask to make it
    # broadcastable with token embeddings
    padding_mask = padding_mask.logical_not().unsqueeze(-1)

    # Compute averages of token embeddings per sequence, ignoring padding tokens
    token_embeddings_masked = x * padding_mask
    token_embeddings_summed = token_embeddings_masked.sum(1)
    token_embeddings_averaged = token_embeddings_summed / padding_mask.sum(1)

    return token_embeddings_averaged


class _BERTBase(_MLMEmbedder):
    """
    BERT base template.
    """

    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_layer = num_encoder_layers - 1

        self._name = name

        self._num_layers = num_encoder_layers
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder_stack = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        padding_mask = x[:, :, 0] == 0

        # Create an embedding of the input tensor, and run through BERT stack
        x_emb = self.embedder(x)
        out = self.encoder_stack(src=x_emb, src_key_padding_mask=padding_mask)

        return out, padding_mask

    def embed(self, x: Tensor) -> Tensor:
        """
        Get the average-pooled, l2-normalised embedding vectors of a given
        layer.
        """
        # Run the input partially through the BERT stack
        padding_mask = x[:, :, 0] == 0
        x_emb = self.embedder(x)
        for layer in self.encoder_stack.layers[: self.embed_layer]:
            x_emb = layer(src=x_emb, src_key_padding_mask=padding_mask)

        # Compute the masked average pool
        emb_mask = padding_mask.clone()
        emb_mask[:, 0] = 1
        x_emb = masked_average_pool(x_emb, padding_mask)

        # l2 norm and return
        return normalize(x_emb, p=2, dim=1)

    def mlm(self, x: Tensor) -> Tensor:
        return self.generator(self.forward(x)[0])


class _BERTClsEmbedBase(_BERTBase):
    """
    Base class for BERT models that use the <cls> token to embed input
    sequences instead of taking the average pool of a particular layer.
    """

    def embed(self, x: Tensor) -> Tensor:
        """
        Get the l2-normalised <cls> embeddings of the final layer.
        """
        x_emb = self.forward(x)[0]
        x_emb = x_emb[:, 0, :]
        return normalize(x_emb, p=2, dim=1)
