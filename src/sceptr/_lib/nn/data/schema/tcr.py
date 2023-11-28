from enum import Enum
import re
from tidytcells import tcr
from typing import Optional, Union


def get_v_gene_indices(gene_symbol):
    match = re.match(r"TR[AB]V([0-9]+)(-([0-9]+))?", gene_symbol)

    group_num = int(match.group(1))
    sub_num_if_any = 0 if match.group(3) is None else int(match.group(3))

    return (group_num, sub_num_if_any)


functional_travs = tcr.query(contains="TRAV", functionality="F", precision="gene")
functional_trbvs = tcr.query(contains="TRBV", functionality="F", precision="gene")


TravGene = Enum(
    "TravGene", sorted(functional_travs, key=get_v_gene_indices), module=__name__
)
TrbvGene = Enum(
    "TrbvGene", sorted(functional_trbvs, key=get_v_gene_indices), module=__name__
)


class Tcrv:
    def __init__(
        self, gene: Union[TravGene, TrbvGene, None], allele_num: Optional[int]
    ) -> None:
        self.gene = gene
        self.allele_num = allele_num

        if not self._gene_is_unknown() and self._allele_is_unknown():
            self._assume_first_allele()

    @property
    def cdr1_sequence(self) -> Optional[str]:
        if self._gene_is_unknown():
            return None

        allele_symbol = self.__repr__()
        cdr1 = tcr.get_aa_sequence(allele_symbol)["CDR1-IMGT"]
        return cdr1

    @property
    def cdr2_sequence(self) -> Optional[str]:
        if self._gene_is_unknown():
            return None

        allele_symbol = self.__repr__()
        cdr2 = tcr.get_aa_sequence(allele_symbol)["CDR2-IMGT"]
        return cdr2

    def __eq__(self, __value: object) -> bool:
        return self.gene == __value.gene and self.allele_num == __value.allele_num

    def __repr__(self) -> str:
        if self._gene_is_unknown():
            return "?"

        return f"{self.gene.name}*{self.allele_num:02d}"

    def _gene_is_unknown(self) -> bool:
        return self.gene is None

    def _allele_is_unknown(self) -> bool:
        return self.allele_num is None

    def _assume_first_allele(self) -> None:
        self.allele_num = 1


class Tcr:
    def __init__(
        self,
        trav: Tcrv,
        junction_a_sequence: Optional[str],
        trbv: Tcrv,
        junction_b_sequence: Optional[str],
    ) -> None:
        self._trav = trav
        self.junction_a_sequence = junction_a_sequence

        self._trbv = trbv
        self.junction_b_sequence = junction_b_sequence

    @property
    def cdr1a_sequence(self) -> Optional[str]:
        return self._trav.cdr1_sequence

    @property
    def cdr2a_sequence(self) -> Optional[str]:
        return self._trav.cdr2_sequence

    @property
    def cdr1b_sequence(self) -> Optional[str]:
        return self._trbv.cdr1_sequence

    @property
    def cdr2b_sequence(self) -> Optional[str]:
        return self._trbv.cdr2_sequence

    def __eq__(self, __value: object) -> bool:
        return (
            self._trav == __value._trav
            and self.junction_a_sequence == __value.junction_a_sequence
            and self._trbv == __value._trbv
            and self.junction_b_sequence == __value.junction_b_sequence
        )

    def __repr__(self) -> str:
        junction_a_repr = self._represent_junction_sequence(self.junction_a_sequence)
        junction_b_repr = self._represent_junction_sequence(self.junction_b_sequence)

        return (
            f"Tra({self._trav}/{junction_a_repr})/Trb({self._trbv}/{junction_b_repr})"
        )

    def _represent_junction_sequence(self, sequence: Optional[str]) -> str:
        if sequence is None:
            return "?"
        else:
            return sequence


def make_tcr_from_components(
    trav_symbol: Optional[str],
    junction_a_sequence: Optional[str],
    trbv_symbol: Optional[str],
    junction_b_sequence: Optional[str],
) -> Tcr:
    trav = _get_trav_from_symbol(trav_symbol)
    trbv = _get_trbv_from_symbol(trbv_symbol)
    return Tcr(
        trav=trav,
        junction_a_sequence=junction_a_sequence,
        trbv=trbv,
        junction_b_sequence=junction_b_sequence,
    )


def _get_trav_from_symbol(symbol: Optional[str]) -> Tcrv:
    if symbol is None:
        return Tcrv(gene=None, allele_num=None)

    gene = _get_trav_gene_object_from_symbol(symbol)
    allele_num = _get_allele_number_from_symbol(symbol)
    return Tcrv(gene=gene, allele_num=allele_num)


def _get_trbv_from_symbol(symbol: Optional[str]) -> Tcrv:
    if symbol is None:
        return Tcrv(gene=None, allele_num=None)

    gene = _get_trbv_gene_object_from_symbol(symbol)
    allele_num = _get_allele_number_from_symbol(symbol)
    return Tcrv(gene=gene, allele_num=allele_num)


def _get_trav_gene_object_from_symbol(symbol: str) -> TravGene:
    str_representing_gene = symbol.split("*")[0]
    return TravGene[str_representing_gene]


def _get_trbv_gene_object_from_symbol(symbol: str) -> TrbvGene:
    str_representing_gene = symbol.split("*")[0]
    return TrbvGene[str_representing_gene]


def _get_allele_number_from_symbol(symbol: str) -> int:
    split_at_asterisk = symbol.split("*")
    has_allele_number = len(split_at_asterisk) == 2

    if not has_allele_number:
        return None

    str_representing_allele_number = split_at_asterisk[1]

    return int(str_representing_allele_number)
