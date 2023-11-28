from sceptr import variant
from sceptr._lib.sceptr import Sceptr
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_data():
    df = pd.read_csv("tests/mock_data.csv")
    return df


@pytest.fixture
def model():
    return variant.ab_sceptr()


def test_load_variant(model):
    assert isinstance(model, Sceptr)


def test_embed(model, dummy_data):
    result = model.calc_vector_representations(dummy_data)

    assert type(result) == np.ndarray
    assert len(result.shape) == 2
    assert result.shape[0] == 3


def test_cdist(model, dummy_data):
    result = model.calc_cdist_matrix(dummy_data, dummy_data)

    assert type(result) == np.ndarray
    assert result.shape == (3, 3)


def test_pdist(model, dummy_data):
    result = model.calc_pdist_vector(dummy_data)

    assert type(result) == np.ndarray
    assert result.shape == (3,)
