from abc import ABC, abstractmethod
from torch import Tensor
from typing import Iterable, Tuple

from sceptr._lib.nn.data.tokeniser.tokeniser import Tokeniser
from sceptr._lib.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class BatchCollator(ABC):
    def __init__(self, tokeniser: Tokeniser) -> None:
        self._tokeniser = tokeniser

    @abstractmethod
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[Tensor]:
        pass
