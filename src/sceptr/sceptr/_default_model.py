from sceptr.variant import ab_sceptr
from numpy import ndarray
from pandas import DataFrame


default_model = ab_sceptr()


def calc_cdist_matrix(anchor_tcrs: DataFrame, comparison_tcrs: DataFrame) -> ndarray:
    return default_model.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)


def calc_pdist_vector(tcrs: DataFrame) -> ndarray:
    return default_model.calc_pdist_vector(tcrs)


def calc_vector_representations(tcrs: DataFrame) -> ndarray:
    return default_model.calc_vector_representations(tcrs)
