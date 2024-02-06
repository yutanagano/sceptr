import pandas as pd
from pandas import DataFrame, Series
from sceptr._lib import schema
from sceptr._lib.schema import Tcr, TcrPmhcPair


def generate_tcr_pmhc_series(data: DataFrame) -> Series:
    tcr_pmhc_series = data.apply(_generate_tcr_pmhc_pair_from_row, axis="columns")
    return tcr_pmhc_series


def generate_tcr_series(data: DataFrame) -> Series:
    tcr_series = data.apply(_generate_tcr_from_row, axis="columns")
    return tcr_series


def _generate_tcr_pmhc_pair_from_row(row: Series) -> TcrPmhcPair:
    tcr = _generate_tcr_from_row(row)

    epitope = _get_value_if_not_na_else_none(row.Epitope)
    mhc_a = _get_value_if_not_na_else_none(row.MHCA)
    mhc_b = _get_value_if_not_na_else_none(row.MHCB)
    pmhc = schema.make_pmhc_from_components(
        epitope_sequence=epitope, mhc_a_symbol=mhc_a, mhc_b_symbol=mhc_b
    )

    return TcrPmhcPair(tcr, pmhc)


def _generate_tcr_from_row(row: Series) -> Tcr:
    trav = _get_value_if_not_na_else_none(row.TRAV)
    trbv = _get_value_if_not_na_else_none(row.TRBV)
    junction_a = _get_value_if_not_na_else_none(row.CDR3A)
    junction_b = _get_value_if_not_na_else_none(row.CDR3B)
    return schema.make_tcr_from_components(
        trav_symbol=trav,
        junction_a_sequence=junction_a,
        trbv_symbol=trbv,
        junction_b_sequence=junction_b,
    )


def _get_value_if_not_na_else_none(value) -> any:
    if pd.isna(value):
        return None
    return value
