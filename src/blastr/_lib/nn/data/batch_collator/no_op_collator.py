from typing import Iterable, Tuple
from torch import Tensor
from blastr._lib.nn.data.batch_collator import BatchCollator
from blastr._lib.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class NoOpCollator(BatchCollator):
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[Tensor]:
        return tcr_pmhc_pairs
