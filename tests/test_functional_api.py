import sceptr
from sceptr.model import ResidueRepresentations
import numpy as np
import pandas as pd
import pytest


sceptr.disable_hardware_acceleration()


@pytest.fixture
def dummy_data():
    df = pd.read_csv("tests/mock_data.csv")
    return df


def test_embed(dummy_data):
    result = sceptr.calc_vector_representations(dummy_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 64)


def test_residue_embed(dummy_data):
    result = sceptr.calc_residue_representations(dummy_data)

    assert isinstance(result, ResidueRepresentations)
    assert result.representation_array.shape == (3, 48, 64)
    assert result.compartment_mask.shape == (3, 48)


def test_cdist(dummy_data):
    result = sceptr.calc_cdist_matrix(dummy_data, dummy_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


def test_pdist(dummy_data):
    result = sceptr.calc_pdist_vector(dummy_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_enable_hardware_acceleration():
    sceptr.enable_hardware_acceleration()
    assert sceptr._USE_HARDWARE_ACCELERATION
    sceptr.disable_hardware_acceleration()


def test_disable_hardware_acceleration():
    sceptr.disable_hardware_acceleration()
    assert not sceptr._USE_HARDWARE_ACCELERATION
