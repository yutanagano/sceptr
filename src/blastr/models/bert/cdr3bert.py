'''
CDR3BERT classes

Compatible tokenisers: ABCDR3Tokeniser, BCDR3Tokeniser
'''


from .bert import _BERTBase, _BERTClsEmbedBase
from .embedding.cdr3 import *
import torch


class _CDR3BERTBase(_BERTBase):
    '''
    CDR3BERT base class.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.generator = torch.nn.Linear(d_model, 20)


class CDR3BERT_a(_CDR3BERTBase):
    '''
    CDR3BERT model that only gets amino acid information.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = CDR3Embedding_a(embedding_dim=d_model)


class CDR3BERT_ap(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and positional information.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = CDR3Embedding_ap(embedding_dim=d_model)


class CDR3BERT_ar(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and relative positional information.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = CDR3Embedding_ar(embedding_dim=d_model)


class CDR3BERT_ab(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and bidirectional positional
    information.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = CDR3Embedding_ab(embedding_dim=d_model)


class CDR3BERT_ac(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and chain information.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = CDR3Embedding_ac(embedding_dim=d_model)


class CDR3BERT_apc(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid, chain, and residue position
    information.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = CDR3Embedding_apc(embedding_dim=d_model)


class CDR3ClsBERT_ap(_BERTClsEmbedBase, CDR3BERT_ap):
    '''
    CDR3BERT_ap model which uses the <cls> token to embed.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )


class CDR3ClsBERT_ab(_BERTClsEmbedBase, CDR3BERT_ab):
    '''
    CDR3BERT_ab model which uses the <cls> token to embed.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )


class CDR3ClsBERT_apc(_BERTClsEmbedBase, CDR3BERT_apc):
    '''
    CDR3BERT_acp model which uses the <cls> token to embed.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )