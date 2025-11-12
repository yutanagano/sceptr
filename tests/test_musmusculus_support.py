import sceptr
import pandas as pd
import pytest


sceptr.disable_hardware_acceleration()


@pytest.fixture
def dummy_data_homosapiens():
    df = pd.read_csv("tests/mock_data.csv")
    return df


def test_musmusculus_support(dummy_data_homosapiens):
    _ = sceptr.calc_vector_representations(dummy_data_homosapiens)
