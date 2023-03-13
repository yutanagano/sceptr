'''
Custom tokeniser classes for TCR tokenisation.

Because of how this codebase modularises tokenisation, the vocabulary can
change depending on the setting. To keep some consistency, there is one index
that is reserved in all cases for padding values:

0: reserved for <pad>

In addition, in the case of token indices (as opposed to indices that may
describe chain, position, etc.), there are two more reserved indices:

1: reserved for <mask>
2: reserved for <cls>
'''


from abc import ABC, abstractmethod
from pandas import isna, notna, Series
import random
from ..resources import *
import torch
from torch import Tensor


class _Tokeniser(ABC):
    '''
    Abstract base class for tokenisers.
    '''

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        '''
        Return this tokeniser's vocabulary size.
        '''


    @abstractmethod
    def tokenise(self, tcr: Series, noising: bool = False) -> Tensor:
        '''
        Given a pandas Series containing information on a particular TCR,
        generate a tokenisation of it. If noising, then randomly drop out
        segments of the TCR representation.
        '''


class _AATokeniser(_Tokeniser):
    '''
    Base class for tokenisers focusing on AA-level tokenisation.
    '''


    def __init__(self) -> None:
        self._aa_to_index = dict()

        for i, aa in enumerate(AMINO_ACIDS):
            self._aa_to_index[aa] = 3+i # offset for reserved tokens


    @property
    def vocab_size(self) -> int:
        return 20


class CDR3Tokeniser(_AATokeniser):
    '''
    Basic tokeniser which will tokenise a TCR in terms of its alpha and beta
    chain CDR3 amino acid sequences.
    '''


    def tokenise(self, tcr: Series, noising: bool = False) -> Tensor:
        '''
        Tokenise a TCR in terms of its alpha and/or beta chain CDR3 amino acid
        sequences.

        :return:
            Tensor with CDR3A/B residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
            Dim 3 - chain ID
        '''

        # NOTE: noising not yet implemented

        cdr3a = tcr.loc['CDR3A']
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0,0]]

        if notna(cdr3a):
            cdr3a_size = len(cdr3a)
            for i, aa in enumerate(cdr3a):
                tokenised.append([self._aa_to_index[aa], i+1, cdr3a_size, 1])

        if notna(cdr3b):
            cdr3b_size = len(cdr3b)
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, cdr3b_size, 2])

        if len(tokenised) == 1:
            raise ValueError(f'No CDR3 data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)


class BCDR3Tokeniser(_AATokeniser):
    '''
    Basic tokeniser which will tokenise a TCR in terms of its beta chain CDR3.
    '''

    def __init__(self, p_drop_aa: float) -> None:
        super().__init__()
        self._p_drop_aa = p_drop_aa


    def tokenise(self, tcr: Series, noising: bool = False) -> Tensor:
        '''
        Tokenise a TCR in terms of its beta chain CDR3 amino acid sequences.

        :return:
            Tensor with CDR3B AA residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
        '''
    
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0]]

        if isna(cdr3b):
            raise ValueError(f'CDR3B data missing from row {tcr.name}')

        cdr3b_size = len(cdr3b)
        for i, aa in enumerate(cdr3b):
            # If noising, randomly drop some AAs
            if noising and random.random() < self._p_drop_aa:
                continue

            tokenised.append([self._aa_to_index[aa], i+1, cdr3b_size])

        return torch.tensor(tokenised, dtype=torch.long)
    

class BVCDR3Tokeniser(_Tokeniser):
    '''
    Tokeniser which takes the beta chain V gene and CDR3 sequence.
    '''

    def __init__(self) -> None:
        self._aa_to_index = dict()
        self._v_to_index = dict()


        for i, aa in enumerate(AMINO_ACIDS):
            self._aa_to_index[aa] = 3+i # offset for reserved tokens

        for i, trbv in enumerate(FUNCTIONAL_TRBVS):
            self._v_to_index[trbv] = 3+20+i # offset for reserved tokens and amino acid tokens


    @property
    def vocab_size(self) -> int:
        return 20 + 48 #aas + trbvs


    def tokenise(self, tcr: Series) -> Tensor:
        '''
        Tokenise a TCR in terms of its beta chain V gene and CDR3 amino acid
        sequence.

        :return:
            Tensor with beta V gene and CDR3 AA residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
            Dim 3 - compartment ID
        '''

        trbv = None if isna(tcr.loc['TRBV']) else tcr.loc['TRBV'].split('*')[0]
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0,0]]

        if notna(trbv):
            tokenised.append([self._v_to_index[trbv], 0, 0, 1])

        if notna(cdr3b):
            cdr3b_size = len(cdr3b)
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, cdr3b_size, 2])
        
        if len(tokenised) == 1:
            raise ValueError(f'No TCRB data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)


class BCDRTokeniser(_AATokeniser):
    '''
    Tokeniser that takes the beta V gene and CDR3, and represents the chain as
    the set of CDRs 1, 2 and 3.
    '''


    def __init__(
        self,
        p_drop_cdr: float,
        p_drop_aa: float
    ) -> None:
        super().__init__()
        self._p_drop_cdr = p_drop_cdr
        self._p_drop_aa = p_drop_aa


    def tokenise(self, tcr: Series, noising: bool = False) -> Tensor:
        '''
        Tokenise a TCR in terms of the amino acid sequences of its 3 CDRs.

        Amino acids get mapped as in the following: 'A' -> 3, 'C' -> 4, ...
        'Y' -> 22.

        :return:
            Tensor with beta CDR AA residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
            Dim 3 - compartment ID
        '''

        # Get the CDRs
        trbv = tcr.loc['TRBV']
        cdr3b = tcr.loc['CDR3B']

        cdr1b = None if isna(trbv) else V_CDRS[trbv]['CDR1-IMGT']
        cdr2b = None if isna(trbv) else V_CDRS[trbv]['CDR2-IMGT']

        if not (cdr1b or cdr2b or cdr3b):
            raise ValueError(f'No TCRB data found in row {tcr.name}.')

        # If noising, potentially drop some CDRs
        include_cdr1 = include_cdr2 = include_cdr3 = True

        if noising:
            include_cdr1 = random.random() < self._p_drop_cdr
            include_cdr2 = random.random() < self._p_drop_cdr
            include_cdr3 = random.random() < self._p_drop_cdr

            # Make sure filter doesn't cause the whole TCR to be censored
            if not (
                (cdr1b and include_cdr1) or 
                (cdr2b and include_cdr2) or 
                (cdr3b and include_cdr3)
            ):
                include_cdr1 = include_cdr2 = include_cdr3 = True

        tokenised = [[2,0,0,0]]

        if (cdr1b and include_cdr1):
            tokenised.extend(self._tokenise_cdr(cdr1b, 1, noising))

        if (cdr2b and include_cdr2):
            tokenised.extend(self._tokenise_cdr(cdr2b, 2, noising))

        if (cdr3b and include_cdr3):
            tokenised.extend(self._tokenise_cdr(cdr3b, 3, noising))

        return torch.tensor(tokenised, dtype=torch.long)
    

    def _tokenise_cdr(self, cdr: str, cmpt_idx: int, noising: bool) -> list:
        cdr_size = len(cdr)

        tokenised = []
        for i, aa in enumerate(cdr):
            # If noising, randomly drop some AAs
            if noising and random.random() < self._p_drop_aa:
                continue

            tokenised.append([self._aa_to_index[aa], i+1, cdr_size, cmpt_idx])

        # Make sure filter doesn't cause the whole CDR to be censored
        if not tokenised:
            for i, aa in enumerate(cdr):
                tokenised.append([self._aa_to_index[aa], i+1, cdr_size, cmpt_idx])

        return tokenised