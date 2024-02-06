from pandas import DataFrame
from sceptr._lib import schema
from sceptr._lib.schema import TcrPmhcPair
from torch.utils.data import Dataset


class TcrDataset(Dataset):
    def __init__(self, data: DataFrame):
        super().__init__()
        self._tcr_pmhc_series = schema.generate_tcr_pmhc_series(data)

    def __len__(self) -> int:
        return len(self._tcr_pmhc_series)

    def __getitem__(self, index: int) -> TcrPmhcPair:
        return self._tcr_pmhc_series.iloc[index]
