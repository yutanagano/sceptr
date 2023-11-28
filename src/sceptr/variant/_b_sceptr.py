from sceptr._lib.sceptr import Sceptr
from sceptr.variant._load_variant import load_variant
from sceptr._model_saves import b_sceptr_model_save


def b_sceptr() -> Sceptr:
    return load_variant(
        b_sceptr_model_save.config,
        b_sceptr_model_save.state_dict,
    )
