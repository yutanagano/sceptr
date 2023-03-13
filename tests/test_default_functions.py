import blastr
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_data():
    df = pd.DataFrame(
        data=[
            ["TRBV13*01", "CASSYLPGQGDHYSNQPQHF", "TRBJ1-5*01"],
            ["TRBV28*01", "CASSLGQSGANVLTF", "TRBJ2-6*01"],
            ["TRBV4-1*01", "CASSDWGSQNTLYF", "TRBJ2-4*01"],
        ],
        columns=["TRBV", "CDR3B", "TRBJ"],
    )
    return df


class TestEmbed:
    def test_embed(self, dummy_data):
        result = blastr.embed(data=dummy_data)

        assert type(result) == np.ndarray
        assert len(result.shape) == 2
        assert result.shape[0] == 3


class TestCDist:
    def test_cdist(self, dummy_data):
        result = blastr.cdist(data_a=dummy_data, data_b=dummy_data)

        assert type(result) == np.ndarray
        assert result.shape == (3, 3)


class TestPDist:
    def test_pdist(self, dummy_data):
        result = blastr.pdist(data=dummy_data)

        assert type(result) == np.ndarray
        assert result.shape == (3,)
