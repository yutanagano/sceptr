from typing import Optional


class MhcGene:
    def __init__(self, symbol: Optional[str]) -> None:
        self.symbol = symbol

    def could_be_equal_to(self, other: "MhcGene") -> bool:
        if self.symbol is None or other.symbol is None:
            return True

        self_is_subset_of_other = other.symbol in self.symbol
        other_is_subset_of_self = self.symbol in other.symbol
        return self_is_subset_of_other or other_is_subset_of_self

    def __repr__(self) -> str:
        if self.symbol is None:
            return "?"

        return self.symbol


class Pmhc:
    def __init__(
        self, epitope_sequence: Optional[str], mhc_a: MhcGene, mhc_b: MhcGene
    ) -> None:
        self.epitope_sequence = epitope_sequence
        self.mhc_a = mhc_a
        self.mhc_b = mhc_b

    def __eq__(self, __value: object) -> bool:
        if self.epitope_sequence is None:
            return False

        same_epitope = self.epitope_sequence == __value.epitope_sequence
        maybe_same_mhc_a = self.mhc_a.could_be_equal_to(__value.mhc_a)
        maybe_same_mhc_b = self.mhc_b.could_be_equal_to(__value.mhc_b)
        return same_epitope and maybe_same_mhc_a and maybe_same_mhc_b

    def __repr__(self) -> str:
        epitope_representation = (
            "?" if self.epitope_sequence is None else self.epitope_sequence
        )
        return f"{epitope_representation}/{self.mhc_a}/{self.mhc_b}"


def make_pmhc_from_components(
    epitope_sequence: Optional[str],
    mhc_a_symbol: Optional[str],
    mhc_b_symbol: Optional[str],
) -> Pmhc:
    mhc_a = MhcGene(mhc_a_symbol)
    mhc_b = MhcGene(mhc_b_symbol)
    return Pmhc(epitope_sequence=epitope_sequence, mhc_a=mhc_a, mhc_b=mhc_b)
