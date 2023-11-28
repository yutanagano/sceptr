import pandas as pd
from pandas import DataFrame, Series
from torch.utils.data import Dataset

from sceptr._lib.nn.data import schema
from sceptr._lib.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class TcrDataset(Dataset):
    def __init__(self, data: DataFrame):
        super().__init__()
        self._tcr_pmhc_series = self._generate_tcr_pmhc_series_from(data)

    def __len__(self) -> int:
        return len(self._tcr_pmhc_series)

    def __getitem__(self, index: int) -> TcrPmhcPair:
        return self._tcr_pmhc_series.iloc[index]

    def _generate_tcr_pmhc_series_from(self, data: DataFrame) -> Series:
        tcr_series = data.apply(self._generate_tcr_pmhc_pair_from_row, axis="columns")
        return tcr_series

    def _generate_tcr_pmhc_pair_from_row(self, row: Series) -> TcrPmhcPair:
        trav = self._get_value_if_not_na_else_none(row.TRAV)
        trbv = self._get_value_if_not_na_else_none(row.TRBV)
        junction_a = self._get_value_if_not_na_else_none(row.CDR3A)
        junction_b = self._get_value_if_not_na_else_none(row.CDR3B)
        tcr = schema.make_tcr_from_components(
            trav_symbol=trav,
            junction_a_sequence=junction_a,
            trbv_symbol=trbv,
            junction_b_sequence=junction_b,
        )

        epitope = self._get_value_if_not_na_else_none(row.Epitope)
        mhc_a = self._get_value_if_not_na_else_none(row.MHCA)
        mhc_b = self._get_value_if_not_na_else_none(row.MHCB)
        pmhc = schema.make_pmhc_from_components(
            epitope_sequence=epitope, mhc_a_symbol=mhc_a, mhc_b_symbol=mhc_b
        )

        return TcrPmhcPair(tcr, pmhc)

    def _get_value_if_not_na_else_none(self, value) -> any:
        if pd.isna(value):
            return None

        return value
