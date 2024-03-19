from enum import IntEnum


class DefaultTokenIndex(IntEnum):
    NULL = 0
    MASK = 1
    CLS = 2


class AminoAcidTokenIndex(IntEnum):
    NULL = DefaultTokenIndex.NULL
    MASK = DefaultTokenIndex.MASK
    CLS = DefaultTokenIndex.CLS
    A = 3
    C = 4
    D = 5
    E = 6
    F = 7
    G = 8
    H = 9
    I = 10
    K = 11
    L = 12
    M = 13
    N = 14
    P = 15
    Q = 16
    R = 17
    S = 18
    T = 19
    V = 20
    W = 21
    Y = 22


class Cdr3CompartmentIndex(IntEnum):
    NULL = DefaultTokenIndex.NULL
    CDR3A = 1
    CDR3B = 2


class SingleChainCdrCompartmentIndex(IntEnum):
    NULL = DefaultTokenIndex.NULL
    CDR1 = 1
    CDR2 = 2
    CDR3 = 3


class CdrCompartmentIndex(IntEnum):
    NULL = DefaultTokenIndex.NULL
    CDR1A = 1
    CDR2A = 2
    CDR3A = 3
    CDR1B = 4
    CDR2B = 5
    CDR3B = 6
