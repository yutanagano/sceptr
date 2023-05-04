"""
CDRBERT classes

Compatible tokenisers: BCDRTokeniser
"""


from .bert import _BERTBase, _BERTClsEmbedBase
from .embedding.cdr import BCDREmbedding, CDREmbedding, BCDREmbeddingBDPos
import torch


class _CDRBERTBase(_BERTBase):
    """
    CDRBERT base class.
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

        self.generator = torch.nn.Linear(d_model, 20)


class BCDRBERT(_CDRBERTBase):
    """
    CDRBERT model for beta-chain only data.
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

        self.embedder = BCDREmbedding(embedding_dim=d_model)


class CDRBERT(_CDRBERTBase):
    """
    CDRBERT model for paired-chain data.
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

        self.embedder = CDREmbedding(embedding_dim=d_model)


class BCDRBERTBDPos(_CDRBERTBase):
    """
    CDRBERT model for beta-chain only data with bidirectional position embeddings.
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

        self.embedder = BCDREmbeddingBDPos(embedding_dim=d_model)


class BCDRClsBERT(_BERTClsEmbedBase, BCDRBERT):
    """
    BCDRBERT model which uses the <cls> token to embed.
    """


class CDRClsBERT(_BERTClsEmbedBase, CDRBERT):
    """
    CDRBERT model which uses the <cls> token to embed.
    """


class BCDRClsBERTBDPos(_BERTClsEmbedBase, BCDRBERTBDPos):
    """
    BCDRBERTBDPos model which uses the <cls> token to embed.
    """
