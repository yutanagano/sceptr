import torch
from torch import Tensor
from typing import List, Optional, Tuple

from sceptr._lib.nn.data.tokeniser.tokeniser import Tokeniser
from sceptr._lib.nn.data.tokeniser.token_indices import AminoAcidTokenIndex, Cdr3CompartmentIndex
from sceptr._lib.schema import Tcr


class Cdr3Tokeniser(Tokeniser):
    """
    Tokenise TCR in terms of its alpha and beta chain CDR3s.

    Dim 0: token index
    Dim 1: token position
    Dim 2: CDR length
    Dim 3: CDR index
    """

    token_vocabulary_index = AminoAcidTokenIndex

    def tokenise(self, tcr: Tcr) -> Tensor:
        initial_cls_vector = (AminoAcidTokenIndex.CLS, 0, 0, Cdr3CompartmentIndex.NULL)

        cdr3a = self._convert_to_numerical_form(
            tcr.junction_a_sequence, Cdr3CompartmentIndex.CDR3A
        )
        cdr3b = self._convert_to_numerical_form(
            tcr.junction_b_sequence, Cdr3CompartmentIndex.CDR3B
        )

        all_cdrs_tokenised = (
            [initial_cls_vector] + cdr3a + cdr3b
        )

        number_of_tokens_other_than_initial_cls = len(all_cdrs_tokenised) - 1
        if number_of_tokens_other_than_initial_cls == 0:
            raise RuntimeError(f"tcr {tcr} does not contain any TCR information")

        return torch.tensor(all_cdrs_tokenised, dtype=torch.long)

    def _convert_to_numerical_form(
        self, aa_sequence: Optional[str], cdr_index: Cdr3CompartmentIndex
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
