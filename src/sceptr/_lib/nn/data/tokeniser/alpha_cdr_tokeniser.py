# TODO: Merge with beta tokeniser


import torch
from torch import Tensor
from typing import List, Optional, Tuple

from sceptr._lib.nn.data.tokeniser.tokeniser import Tokeniser
from sceptr._lib.nn.data.tokeniser.token_indices import (
    AminoAcidTokenIndex,
    SingleChainCdrCompartmentIndex,
)
from sceptr._lib.schema import Tcr


class AlphaCdrTokeniser(Tokeniser):
    """
    Tokenise TCR in terms of its alpha chain CDRs 1 2 and 3.

    Dim 0: token index
    Dim 1: topen position
    Dim 2: CDR length
    Dim 3: CDR index
    """

    token_vocabulary_index = AminoAcidTokenIndex

    def tokenise(self, tcr: Tcr) -> Tensor:
        initial_cls_vector = (
            AminoAcidTokenIndex.CLS,
            0,
            0,
            SingleChainCdrCompartmentIndex.NULL,
        )

        cdr1b = self._convert_to_numerical_form(
            tcr.cdr1a_sequence, SingleChainCdrCompartmentIndex.CDR1
        )
        cdr2b = self._convert_to_numerical_form(
            tcr.cdr2a_sequence, SingleChainCdrCompartmentIndex.CDR2
        )
        cdr3b = self._convert_to_numerical_form(
            tcr.junction_a_sequence, SingleChainCdrCompartmentIndex.CDR3
        )

        all_cdrs_tokenised = [initial_cls_vector] + cdr1b + cdr2b + cdr3b

        number_of_tokens_other_than_initial_cls = len(all_cdrs_tokenised) - 1
        if number_of_tokens_other_than_initial_cls == 0:
            raise RuntimeError(f"tcr {tcr} does not contain any TRA information")

        return torch.tensor(all_cdrs_tokenised, dtype=torch.long)

    def _convert_to_numerical_form(
        self, aa_sequence: Optional[str], cdr_index: SingleChainCdrCompartmentIndex
    ) -> List[Tuple[int]]:
        if aa_sequence is None:
            return []

        token_indices = [AminoAcidTokenIndex[aa] for aa in aa_sequence]
        token_positions = [idx for idx, _ in enumerate(aa_sequence, start=1)]
        cdr_length = [len(aa_sequence) for _ in aa_sequence]
        compartment_index = [cdr_index for _ in aa_sequence]

        iterator_over_token_vectors = zip(
            token_indices, token_positions, cdr_length, compartment_index
        )

        return list(iterator_over_token_vectors)
