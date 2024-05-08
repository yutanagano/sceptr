from sceptr._model_saves import load_variant


def default():
    return load_variant("SCEPTR")


def mlm_only():
    return load_variant("SCEPTR_MLM_only")


def classic():
    return load_variant("SCEPTR_classic")


def classic_mlm_only():
    return load_variant("SCEPTR_classic_MLM_only")


def cdr3_only():
    return load_variant("SCEPTR_CDR3_only")


def cdr3_only_mlm_only():
    return load_variant("SCEPTR_CDR3_only_MLM_only")


def large():
    return load_variant("SCEPTR_large")


def small():
    return load_variant("SCEPTR_small")


def tiny():
    return load_variant("SCEPTR_tiny")


def blosum():
    return load_variant("SCEPTR_BLOSUM")


def average_pooling():
    return load_variant("SCEPTR_average_pooling")


def shuffled_data():
    return load_variant("SCEPTR_shuffled_data")


def synthetic_data():
    return load_variant("SCEPTR_synthetic_data")


def dropout_noise_only():
    return load_variant("SCEPTR_dropout_noise_only")


def finetuned():
    return load_variant("SCEPTR_finetuned")


def a_sceptr():
    return load_variant("A_SCEPTR")


def b_sceptr():
    return load_variant("B_SCEPTR")