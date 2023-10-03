import math
import random
import torch
from torch import LongTensor
from torch.nn import utils
from typing import Iterable, List, Tuple

from blastr._lib.nn.data.batch_collator import BatchCollator
from blastr._lib.nn.data.tokeniser.token_indices import DefaultTokenIndex
from blastr._lib.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class MlmBatchCollator(BatchCollator):
    PROPORTION_OF_TOKENS_TO_MASK = 0.15
    PROBABILITY_MASKING_RESULTS_IN_RANDOM_AA = 0.1
    PROBABILITY_MASKING_RESULTS_IN_NO_OP = 0.1

    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[LongTensor]:
        tokenised_tcrs = [self._tokeniser.tokenise(pair.tcr) for pair in tcr_pmhc_pairs]

        indices_of_tokens_to_mask = [
            self._choose_random_subset_of_indices(
                tcr, self.PROPORTION_OF_TOKENS_TO_MASK
            )
            for tcr in tokenised_tcrs
        ]

        masked_tcrs = [
            self._mask_tcr(tcr, indices_to_mask)
            for tcr, indices_to_mask in zip(tokenised_tcrs, indices_of_tokens_to_mask)
        ]
        mlm_targets = [
            self._generate_mlm_targets(tcr, indices_to_mask)
            for tcr, indices_to_mask in zip(tokenised_tcrs, indices_of_tokens_to_mask)
        ]

        masked_tcrs_padded = self._pad_tokenised_sequences(masked_tcrs)
        mlm_targets_padded = self._pad_tokenised_sequences(mlm_targets)

        return masked_tcrs_padded, mlm_targets_padded

    def _choose_random_subset_of_indices(
        self, tokenised_tcr: LongTensor, proportion: float
    ) -> List[int]:
        number_of_tokens = self._get_num_tokens_in(tokenised_tcr)
        indices_of_tokens_to_mask = self._randomly_pick_token_indices_from(
            number_of_tokens, proportion
        )
        return indices_of_tokens_to_mask

    def _get_num_tokens_in(self, tokenised_tcr: LongTensor) -> int:
        num_tokens_including_cls = len(tokenised_tcr)
        num_tokens_ignoring_cls = num_tokens_including_cls - 1
        return num_tokens_ignoring_cls

    def _randomly_pick_token_indices_from(
        self, num_tokens: int, proportion: float
    ) -> List[int]:
        num_indices_to_mask = math.ceil(num_tokens * proportion)

        FIRST_TOKEN_IDX_AFTER_CLS = 1
        indices_to_mask = random.sample(
            range(FIRST_TOKEN_IDX_AFTER_CLS, FIRST_TOKEN_IDX_AFTER_CLS + num_tokens),
            k=num_indices_to_mask,
        )

        return indices_to_mask

    def _mask_tcr(
        self, tokenised_tcr: LongTensor, indices_to_mask: List[int]
    ) -> LongTensor:
        masked_tcr = tokenised_tcr.clone()

        for idx_to_mask in indices_to_mask:
            r = random.random()
            replace_token_at_idx_with_random_aa = (
                r < self.PROBABILITY_MASKING_RESULTS_IN_RANDOM_AA
            )
            leave_token_at_idx_untouched = (
                1 - r < self.PROBABILITY_MASKING_RESULTS_IN_NO_OP
            )
            TOKEN_ID_DIM = 0

            if replace_token_at_idx_with_random_aa:
                current_token_at_idx = tokenised_tcr[idx_to_mask, TOKEN_ID_DIM]
                tokens_that_could_replace_it = (
                    set(self._tokeniser.token_vocabulary_index)
                    - set(DefaultTokenIndex)
                    - {current_token_at_idx}
                )
                masked_tcr[idx_to_mask, TOKEN_ID_DIM] = random.choice(
                    sorted(tokens_that_could_replace_it)
                )
                continue

            elif leave_token_at_idx_untouched:
                continue

            else:
                masked_tcr[idx_to_mask, TOKEN_ID_DIM] = DefaultTokenIndex.MASK

        return masked_tcr

    def _generate_mlm_targets(
        self, tokenised_tcr: LongTensor, indices_to_mask: List[int]
    ) -> LongTensor:
        TOKEN_ID_DIM = 0
        mlm_targets = torch.full_like(
            tokenised_tcr[:, TOKEN_ID_DIM], fill_value=DefaultTokenIndex.NULL
        )

        for idx in indices_to_mask:
            mlm_targets[idx] = tokenised_tcr[idx, TOKEN_ID_DIM]

        return mlm_targets

    def _pad_tokenised_sequences(
        self, tokenised_sequences: Iterable[LongTensor]
    ) -> LongTensor:
        return utils.rnn.pad_sequence(
            sequences=tokenised_sequences,
            batch_first=True,
            padding_value=DefaultTokenIndex.NULL,
        )
