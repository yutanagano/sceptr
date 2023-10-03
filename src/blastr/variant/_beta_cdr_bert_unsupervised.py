from blastr._lib.blastr import Blastr
from blastr.variant._load_variant import load_variant
from blastr._model_saves import beta_cdr_bert_unsupervised_model_save


def beta_cdr_bert_unsupervised() -> Blastr:
    return load_variant(
        beta_cdr_bert_unsupervised_model_save.config,
        beta_cdr_bert_unsupervised_model_save.state_dict,
    )
