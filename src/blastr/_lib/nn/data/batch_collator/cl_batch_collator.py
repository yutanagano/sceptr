import numpy as np
from typing import Iterable, Tuple, List
import torch
from torch import BoolTensor, LongTensor, Tensor

from blastr._lib.nn.data.batch_collator import MlmBatchCollator
from blastr._lib.nn.data.tokeniser.token_indices import DefaultTokenIndex
from blastr._lib.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class ClBatchCollator(MlmBatchCollator):
    PROPORTION_OF_TOKENS_TO_CENSOR = 0.2

    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[Tensor]:
        double_view_batch = self._generate_double_view_batch(tcr_pmhc_pairs)
        double_view_positives_mask = self._generate_double_view_positives_mask(
            tcr_pmhc_pairs
        )
        masked_tcrs_padded, mlm_targets_padded = super().collate_fn(tcr_pmhc_pairs)

        return (
            double_view_batch,
            double_view_positives_mask,
            masked_tcrs_padded,
            mlm_targets_padded,
        )

    def _generate_double_view_batch(
        self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]
    ) -> LongTensor:
        tokenised_tcrs = [self._tokeniser.tokenise(pair.tcr) for pair in tcr_pmhc_pairs]
        tokenised_tcrs_view_1 = self._randomly_censor(tokenised_tcrs)
        tokenised_tcrs_view_2 = self._randomly_censor(tokenised_tcrs)
        double_view_batch = self._pad_tokenised_sequences(
            tokenised_tcrs_view_1 + tokenised_tcrs_view_2
        )
        return double_view_batch

    def _randomly_censor(self, tokenised_tcrs: LongTensor) -> LongTensor:
        indices_of_tokens_to_drop = [
            self._choose_random_subset_of_indices(
                tcr, self.PROPORTION_OF_TOKENS_TO_CENSOR
            )
            for tcr in tokenised_tcrs
        ]
        censored_tcrs = [
            self._censor_tcr(tcr, indices_to_drop)
            for tcr, indices_to_drop in zip(tokenised_tcrs, indices_of_tokens_to_drop)
        ]
        return censored_tcrs

    def _censor_tcr(
        self, tokenised_tcr: LongTensor, indices_to_drop: List[int]
    ) -> LongTensor:
        censored_tcr = tokenised_tcr.clone()
        TOKEN_ID_DIM = 0

        for idx_to_drop in indices_to_drop:
            censored_tcr[idx_to_drop, TOKEN_ID_DIM] = DefaultTokenIndex.NULL

        return censored_tcr

    def _generate_double_view_positives_mask(
        self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]
    ) -> BoolTensor:
        num_samples_in_single_view = len(tcr_pmhc_pairs)
        single_view_identities = np.eye(num_samples_in_single_view)

        pmhc_array = np.array([pair.pmhc for pair in tcr_pmhc_pairs])
        single_view_pmhc_matches = (
            pmhc_array[:, np.newaxis] == pmhc_array[np.newaxis, :]
        )

        single_view_positives_including_identities = np.logical_or(
            single_view_identities, single_view_pmhc_matches
        )

        double_view_identities = np.eye(2 * num_samples_in_single_view)
        double_view_positives_including_identities = np.tile(
            single_view_positives_including_identities, reps=(2, 2)
        )
        double_view_positives = np.logical_and(
            double_view_positives_including_identities,
            np.logical_not(double_view_identities),
        )

        return torch.tensor(double_view_positives, dtype=torch.bool)
