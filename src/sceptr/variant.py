from sceptr._model_saves import load_variant


def a_sceptr():
    return load_variant("A_SCEPTR")


def b_sceptr():
    return load_variant("B_SCEPTR")


def ab_sceptr():
    return load_variant("AB_SCEPTR")


def ab_sceptr_large():
    return load_variant("AB_SCEPTR_large")


def ab_sceptr_blosum():
    return load_variant("AB_SCEPTR_BLOSUM")


def ab_sceptr_xlarge_cdr3_only_mlm_only():
    return load_variant("AB_SCEPTR_Large_CDR3_only_MLM_only")


def ab_sceptr_cdr3_only():
    return load_variant("AB_SCEPTR_CDR3_only")


def ab_sceptr_cdr3_only_mlm_only():
    return load_variant("AB_SCEPTR_CDR3_only_MLM_only")