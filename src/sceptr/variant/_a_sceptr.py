from sceptr._lib.sceptr import Sceptr
from sceptr.variant._load_variant import load_variant
from sceptr._model_saves import a_sceptr_model_save


def a_sceptr() -> Sceptr:
    return load_variant(
        a_sceptr_model_save.config,
        a_sceptr_model_save.state_dict,
    )
