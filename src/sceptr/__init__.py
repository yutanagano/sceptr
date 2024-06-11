"""
SCEPTR is a small, fast, and performant TCR representation model for alignment-free TCR analysis.
The root module provides easy access to SCEPTR through a functional API which uses the default model.
"""

from sceptr import variant
from sceptr.model import Sceptr
import sys
from numpy import ndarray
from pandas import DataFrame


__version__ = "1.0.1"


def calc_cdist_matrix(anchors: DataFrame, comparisons: DataFrame) -> ndarray:
    """
    Generate a cdist matrix between two collections of TCRs.

    Parameters
    ----------
    anchors : DataFrame
        DataFrame in the :ref:`prescribed format <data_format>`.
    comparisons : DataFrame
        DataFrame in the :ref:`prescribed format <data_format>`.

    Returns
    -------
    ndarray
        A 2D numpy ndarray representing a cdist matrix between TCRs from `anchors` and `comparisons`.
        The returned array will have shape :math:`(X, Y)` where :math:`X` is the number of TCRs in `anchors` and :math:`Y` is the number of TCRs in `comparisons`.
    """
    return get_default_model().calc_cdist_matrix(anchors, comparisons)


def calc_pdist_vector(instances: DataFrame) -> ndarray:
    r"""
    Generate a pdist vector of distances between each pair of TCRs in the input data.

    Parameters
    ----------
    instances : DataFrame
        DataFrame in the :ref:`prescribed format <data_format>`.

    Returns
    -------
    ndarray
        A 1D numpy ndarray representing a pdist vector of distances between each pair of TCRs in `instances`.
        The returned array will have shape :math:`(\frac{1}{2}N(N-1),)`, where :math:`N` is the number of TCRs in `instances`.
    """
    return get_default_model().calc_pdist_vector(instances)


def calc_vector_representations(instances: DataFrame) -> ndarray:
    """
    Map a table of TCRs provided as a pandas DataFrame in the above format to their corresponding vector representations.

    Parameters
    ----------
    instances : DataFrame
        DataFrame in the :ref:`prescribed format <data_format>`.

    Returns
    -------
    ndarray
        A 2D numpy ndarray object where every row vector corresponds to a row in `instances`.
        The returned array will have shape :math:`(N, 64)` where :math:`N` is the number of TCRs in `instances`.
    """
    return get_default_model().calc_vector_representations(instances)


def get_default_model() -> Sceptr:
    if "_DEFAULT_MODEL" not in dir(sys.modules[__name__]):
        setattr(sys.modules[__name__], "_DEFAULT_MODEL", variant.default())
    return _DEFAULT_MODEL
