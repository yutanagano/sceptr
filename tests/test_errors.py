import pandas as pd
import pytest
import sceptr


def test_bad_v():
    df = pd.read_csv("tests/bad_trav.csv")
    with pytest.raises(ValueError, match="Bad TRAV symbol at index 2"):
        sceptr.calc_vector_representations(df)


def test_bad_cdr3():
    df = pd.read_csv("tests/bad_cdr3a.csv")
    with pytest.raises(ValueError, match="Bad CDR3A sequence at index 2"):
        sceptr.calc_vector_representations(df)
