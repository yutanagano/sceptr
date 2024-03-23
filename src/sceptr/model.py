import torch
from torch import FloatTensor
from torch.nn import utils
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from libtcrlm.bert import Bert
from libtcrlm.tokeniser import Tokeniser
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex
from libtcrlm import schema


BATCH_SIZE = 512


class Sceptr:
    name: str = None
    distance_bins = np.linspace(0, 2, num=21)

    def __init__(
        self, name: str, tokeniser: Tokeniser, bert: Bert, device: torch.device
    ) -> None:
        self.name = name
        self._tokeniser = tokeniser
        self._bert = bert.eval()
        self._device = device

    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        torch_representations = self._calc_torch_representations(instances)
        return torch_representations.cpu().numpy()

    @torch.no_grad()
    def _calc_torch_representations(self, instances: DataFrame) -> FloatTensor:
        instances = instances.copy()

        for col in ("TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"):
            if col not in instances:
                instances[col] = None

        tcrs = schema.generate_tcr_series(instances)

        representations = []
        for idx in range(0, len(tcrs), BATCH_SIZE):
            batch = tcrs.iloc[idx : idx + BATCH_SIZE]
            tokenised_batch = [self._tokeniser.tokenise(tcr) for tcr in batch]
            padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised_batch,
                batch_first=True,
                padding_value=DefaultTokenIndex.NULL,
            )
            batch_representation = self._bert.get_vector_representations_of(
                padded_batch.to(self._device)
            )
            representations.append(batch_representation)

        return torch.concatenate(representations, dim=0)

    def calc_cdist_matrix(self, anchors: DataFrame, comparisons: DataFrame) -> ndarray:
        anchor_representations = self._calc_torch_representations(anchors)
        comparison_representations = self._calc_torch_representations(comparisons)
        cdist_matrix = torch.cdist(
            anchor_representations, comparison_representations, p=2
        )
        return cdist_matrix.cpu().numpy()

    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        representations = self._calc_torch_representations(instances)
        pdist_vector = torch.pdist(representations, p=2)
        return pdist_vector.cpu().numpy()
