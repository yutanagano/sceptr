"""
CDR3BERT classes

Compatible tokenisers: BVCDR3Tokeniser
"""


from .bert import _BERTBase, _BERTClsEmbedBase
from .embedding.vcdr3 import BVCDR3Embedding
import torch


class _VCDR3BERTBase(_BERTBase):
    """
    VCDR3BERT base class.
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
        super().__init__(
            name, num_encoder_layers, d_model, nhead, dim_feedforward, dropout
        )

        self.generator = torch.nn.Linear(d_model, 68)


class BVCDR3BERT(_VCDR3BERTBase):
    """
    VCDR3BERT model for beta-chain only data.
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
        super().__init__(
            name, num_encoder_layers, d_model, nhead, dim_feedforward, dropout
        )

        self.embedder = BVCDR3Embedding(embedding_dim=d_model)


class BVCDR3ClsBERT(_BERTClsEmbedBase, BVCDR3BERT):
    """
    BVCDR3BERT model which uses the <cls> token to embed.
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
        super().__init__(
            name, num_encoder_layers, d_model, nhead, dim_feedforward, dropout
        )
