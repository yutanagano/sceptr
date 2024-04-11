from sceptr._model_saves import load_variant


def sceptr():
    return load_variant("SCEPTR")


def sceptr_mlm_only():
    return load_variant("SCEPTR_MLM_only")


def sceptr_classic():
    return load_variant("SCEPTR_classic")


def sceptr_classic_mlm_only():
    return load_variant("SCEPTR_classic_MLM_only")


def sceptr_cdr3_only():
    return load_variant("SCEPTR_CDR3_only")


def sceptr_cdr3_only_mlm_only():
    return load_variant("SCEPTR_CDR3_only_MLM_only")


def sceptr_large():
    return load_variant("SCEPTR_large")


def sceptr_blosum():
    return load_variant("SCEPTR_BLOSUM")


def sceptr_finetuned():
    return load_variant("SCEPTR_finetuned")


def a_sceptr():
    return load_variant("A_SCEPTR")


def b_sceptr():
    return load_variant("B_SCEPTR")