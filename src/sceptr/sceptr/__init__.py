"""
This submodule provides easy access to SCEPTR's functional API.
The functional API uses the default SCEPTR model.
"""


from sceptr import variant
from numpy import ndarray
from pandas import DataFrame


default_model = variant.default()


def calc_cdist_matrix(anchors: DataFrame, comparisons: DataFrame) -> ndarray:
    """
    Generate a cdist matrix between two collections of TCRs.

    Parameters
    ----------
    anchors : DataFrame
        DataFrame in the prescribed format, representing TCRs from collection A.
    comparisons : DataFrame
        DataFrame in the prescribed format, representing TCRs from collection B.

    Returns
    -------
    ndarray
        A 2D numpy ndarray representing a cdist matrix between TCRs from `anchors` and `comparisons`.
        The returned array will have shape (X, Y) where X is the number of TCRs in `anchors` and Y is the number of TCRs in `comparisons`.
    """
    return default_model.calc_cdist_matrix(anchors, comparisons)


def calc_pdist_vector(instances: DataFrame) -> ndarray:
    """
    Generate a pdist set of distances between each pair of TCRs in the input data.

    Parameters
    ----------
    instances : DataFrame
        DataFrame in the prescribed format.

    Returns
    -------
    ndarray
        A 2D numpy ndarray representing a pdist vector of distances between each pair of TCRs in `instances`.
        The returned array will have shape (1/2 * N * (N-1),), where N is the number of TCRs in `instances`.
    """
    return default_model.calc_pdist_vector(instances)


def calc_vector_representations(instances: DataFrame) -> ndarray:
    """
    Map a table of TCRs provided as a pandas DataFrame in the above format to a set of vector representations.

    Parameters
    ----------
    instances : DataFrame
        DataFrame in the presribed format.

    Returns
    -------
    ndarray
        A 2D numpy ndarray object where every row vector corresponds to a row in `instances`.
        The returned array will have shape (N, D) where N is the number of TCRs in `instances` and D is the dimensionality of the SCEPTR model.
    """
    return default_model.calc_vector_representations(instances)
