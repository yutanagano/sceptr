import numpy as np
import pytest
from sceptr.model import ResidueRepresentations


def test_repr(res_reps):
    assert res_reps.__repr__() == "ResidueRepresentations[num_tcrs: 3, rep_dim: 64]"


@pytest.fixture
def res_reps() -> ResidueRepresentations:
    rep_array = np.zeros((3, 10, 64))
    comp_mask = np.zeros_like(rep_array, dtype=int)
    return ResidueRepresentations(rep_array, comp_mask)
