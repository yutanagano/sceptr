import sceptr
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_data():
    df = pd.read_csv("tests/mock_data.csv")
    return df


def test_embed(dummy_data):
    result = sceptr.calc_vector_representations(dummy_data)

    assert type(result) == np.ndarray
    assert len(result.shape) == 2
    assert result.shape[0] == 3


def test_cdist(dummy_data):
    result = sceptr.calc_cdist_matrix(dummy_data, dummy_data)

    assert type(result) == np.ndarray
    assert result.shape == (3, 3)


def test_pdist(dummy_data):
    result = sceptr.calc_pdist_vector(dummy_data)

    assert type(result) == np.ndarray
    assert result.shape == (3,)
