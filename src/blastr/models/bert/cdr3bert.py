"""
CDR3BERT classes

Compatible tokenisers: ABCDR3Tokeniser, BCDR3Tokeniser
"""


from .bert import _BERTBase, _BERTClsEmbedBase
from .embedding.cdr3 import *
import torch


class _CDR3BERTBase(_BERTBase):
    """
    CDR3BERT base class.
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


class BCDR3BERTMPos(_CDR3BERTBase):
    """
    CDR3BERT model that only gets amino acid information.
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

        self.embedder = CDR3Embedding_a(embedding_dim=d_model)


class BCDR3BERT(_CDR3BERTBase):
    """
    CDR3BERT model that gets amino acid and positional information.
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

        self.embedder = CDR3Embedding_ap(embedding_dim=d_model)


class BCDR3BERTRPos(_CDR3BERTBase):
    """
    CDR3BERT model that gets amino acid and relative positional information.
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

        self.embedder = CDR3Embedding_ar(embedding_dim=d_model)


class BCDR3BERTBDPos(_CDR3BERTBase):
    """
    CDR3BERT model that gets amino acid and bidirectional positional
    information.
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

        self.embedder = CDR3Embedding_ab(embedding_dim=d_model)


class CDR3BERTMPos(_CDR3BERTBase):
    """
    CDR3BERT model that gets amino acid and chain information.
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

        self.embedder = CDR3Embedding_ac(embedding_dim=d_model)


class CDR3BERT(_CDR3BERTBase):
    """
    CDR3BERT model that gets amino acid, chain, and residue position
    information.
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

        self.embedder = CDR3Embedding_apc(embedding_dim=d_model)


class BCDR3ClsBERT(_BERTClsEmbedBase, BCDR3BERT):
    """
    CDR3BERT_ap model which uses the <cls> token to embed.
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


class BCDR3ClsBERTBDPos(_BERTClsEmbedBase, BCDR3BERTBDPos):
    """
    CDR3BERT_ab model which uses the <cls> token to embed.
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


class CDR3ClsBERT(_BERTClsEmbedBase, CDR3BERT):
    """
    CDR3BERT_acp model which uses the <cls> token to embed.
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
