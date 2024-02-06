import torch
from torch import Tensor
from typing import List, Optional, Tuple

from sceptr._lib.nn.data.tokeniser.tokeniser import Tokeniser
from sceptr._lib.nn.data.tokeniser.token_indices import AminoAcidTokenIndex, DefaultTokenIndex
from sceptr._lib.schema import Tcr


class BetaCdr3Tokeniser(Tokeniser):
    """
    Tokenise TCR in terms of its beta chain CDR3.

    Dim 0: token index
    Dim 1: topen position
    Dim 2: CDR length
    """

    token_vocabulary_index = AminoAcidTokenIndex

    def tokenise(self, tcr: Tcr) -> Tensor:
        initial_cls_vector = (
            AminoAcidTokenIndex.CLS,
            DefaultTokenIndex.NULL,
            DefaultTokenIndex.NULL,
        )
        cdr3b = self._convert_to_numerical_form(tcr.junction_b_sequence)

        tokenised = [initial_cls_vector] + cdr3b

        number_of_tokens_other_than_initial_cls = len(tokenised) - 1
        if number_of_tokens_other_than_initial_cls == 0:
            raise RuntimeError(f"tcr {tcr} does not contain beta junction information")

        return torch.tensor(tokenised, dtype=torch.long)

    def _convert_to_numerical_form(
        self, aa_sequence: Optional[str]
    ) -> List[Tuple[int]]:
        if aa_sequence is None:
            return []

        token_indices = [AminoAcidTokenIndex[aa] for aa in aa_sequence]
        token_positions = [idx for idx, _ in enumerate(aa_sequence, start=1)]
        cdr_length = [len(aa_sequence) for _ in aa_sequence]

        iterator_over_token_vectors = zip(token_indices, token_positions, cdr_length)

        return list(iterator_over_token_vectors)
