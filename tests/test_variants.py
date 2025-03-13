import logging
import numpy as np
import pandas as pd
import pytest
import sceptr
from sceptr import variant
from sceptr.model import Sceptr, ResidueRepresentations


sceptr.disable_hardware_acceleration()


@pytest.fixture
def dummy_data():
    df = pd.read_csv("tests/mock_data.csv")
    return df


@pytest.fixture
def default_model():
    return variant.default()


@pytest.mark.parametrize(
    "model",
    (
        variant.default(),
        variant.mlm_only(),
        variant.left_aligned(),
        variant.cdr3_only(),
        variant.cdr3_only_mlm_only(),
        variant.large(),
        variant.small(),
        variant.tiny(),
        variant.blosum(),
        variant.average_pooling(),
        variant.shuffled_data(),
        variant.synthetic_data(),
        variant.dropout_noise_only(),
        variant.finetuned(),
        variant.a_sceptr(),
        variant.b_sceptr(),
    ),
)
class TestVariant:
    def test_load_variant(self, model):
        assert isinstance(model, Sceptr)

    def test_enable_hardware_acceleration(self, model, caplog):
        caplog.set_level(logging.DEBUG)
        model.enable_hardware_acceleration()
        assert "enable_hardware_acceleration" in caplog.text
        assert model.name in caplog.text
        model.disable_hardware_acceleration()

    def test_disable_hardware_acceleration(self, model, caplog):
        caplog.set_level(logging.DEBUG)
        model.disable_hardware_acceleration()
        assert "disable_hardware_acceleration" in caplog.text
        assert model.name in caplog.text

    def test_embed(self, model, dummy_data):
        result = model.calc_vector_representations(dummy_data)

        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2
        assert result.shape[0] == 3

    def test_cdist(self, model, dummy_data):
        result = model.calc_cdist_matrix(dummy_data, dummy_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    def test_pdist(self, model, dummy_data):
        result = model.calc_pdist_vector(dummy_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_residue_representations(self, model, dummy_data):
        if model.name in (
            "SCEPTR",
            "SCEPTR (MLM only)",
            "SCEPTR (left-aligned)",
            "SCEPTR (small)",
            "SCEPTR (BLOSUM)",
            "SCEPTR (average-pooling)",
            "SCEPTR (finetuned)",
        ):
            result = model.calc_residue_representations(dummy_data)

            assert isinstance(result, ResidueRepresentations)
            assert len(result.representation_array.shape) == 3
            assert result.representation_array.shape[:2] == (3, 48)
            assert result.compartment_mask.shape == (3, 48)

        if model.name in (
            "SCEPTR (CDR3 only)",
            "A SCEPTR",
        ):
            with pytest.raises(NotImplementedError):
                model.calc_residue_representations(dummy_data)


def test_set_batch_size(default_model):
    assert default_model._batch_size == 512
    default_model.set_batch_size(128)
    assert default_model._batch_size == 128


def test_set_batch_size_type_error(default_model):
    with pytest.raises(TypeError):
        default_model.set_batch_size("128")
