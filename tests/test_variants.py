from sceptr import variant
from sceptr.model import Sceptr
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_data():
    df = pd.read_csv("tests/mock_data.csv")
    return df


@pytest.mark.parametrize(
    "model",
    (
        variant.ab_sceptr(),
        variant.ab_sceptr_large(),
        variant.ab_sceptr_blosum(),
        variant.ab_sceptr_mlm_only(),
        variant.ab_sceptr_cdr3_only(),
        variant.ab_sceptr_cdr3_only_mlm_only(),
        variant.ab_sceptr_xlarge_cdr3_only_mlm_only(),
        variant.a_sceptr(),
        variant.b_sceptr(),
    ),
)
class TestVariant:
    def test_load_variant(self, model):
        assert isinstance(model, Sceptr)

    def test_embed(self, model, dummy_data):
        result = model.calc_vector_representations(dummy_data)

        assert type(result) == np.ndarray
        assert len(result.shape) == 2
        assert result.shape[0] == 3

    def test_cdist(self, model, dummy_data):
        result = model.calc_cdist_matrix(dummy_data, dummy_data)

        assert type(result) == np.ndarray
        assert result.shape == (3, 3)

    def test_pdist(self, model, dummy_data):
        result = model.calc_pdist_vector(dummy_data)

        assert type(result) == np.ndarray
        assert result.shape == (3,)
