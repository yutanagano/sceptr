from sceptr._lib.sceptr import Sceptr
from sceptr.variant._load_variant import load_variant
from sceptr._model_saves import ab_sceptr_blosum_model_save


def ab_sceptr_blosum() -> Sceptr:
    return load_variant(
        ab_sceptr_blosum_model_save.config,
        ab_sceptr_blosum_model_save.state_dict,
    )
