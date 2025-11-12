import sceptr
import numpy as np
import pandas as pd
import pytest


sceptr.disable_hardware_acceleration()


@pytest.fixture
def dummy_data():
    sceptr.setup("musmusculus")
    yield pd.read_csv("tests/mock_data_musmusculus.csv")
    sceptr.setup("homosapiens")


def test_musmusculus_support(dummy_data):
    result = sceptr.calc_vector_representations(dummy_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 64)
