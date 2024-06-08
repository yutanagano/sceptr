"""
For the curious/power users, several model variants are available to load and use through this submodule.
The submodule exposes functions, each named after a particular variant, which when called will return a :py:class:`~sceptr.model.Sceptr` instance corresponding to the selected model variant.
:py:class:`~sceptr.model.Sceptr` instances expose the same methods as in the functional API: namely :py:func:`~sceptr.model.Sceptr.calc_pdist_vector`, :py:func:`~sceptr.model.Sceptr.calc_cdist_matrix`, and :py:func:`~sceptr.model.Sceptr.calc_vector_representations`.
Each of their function signatures are equivalent to the functional API, so you can just plug and play!
"""

from sceptr._model_saves import load_variant


def default():
    """
    Load the default variant of SCEPTR.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the default model state loaded.
        Calling methods on this :py:class:`~sceptr.model.Sceptr` instance is equivalent to using the functional API.
    """
    return load_variant("SCEPTR")


def mlm_only():
    """
    Load the MLM-only variant of SCEPTR.
    This is the default model trained without autocontrastive learning.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the MLM-only model state loaded.
    """
    return load_variant("SCEPTR_MLM_only")


def left_aligned():
    """
    Load the left-aligned variant of SCEPTR.
    This is similar to the default model, but with learnable token embeddings and a sinusoidal position information embedding method more similar to the original NLP BERT/transformer models.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the left-aligned model state loaded.
    """
    return load_variant("SCEPTR_left_aligned")


def cdr3_only():
    """
    Load the CDR3-only variant of SCEPTR.
    This variant only uses the CDR3 loops as input.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the CDR3-only model state loaded.
    """
    return load_variant("SCEPTR_CDR3_only")


def cdr3_only_mlm_only():
    """
    Load the CDR3-only MLM-only variant of SCEPTR.
    This variant only uses CDR3 loops as input, and did not receive autocontrastive learning.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the CDR3-only MLM-only model state loaded.
    """
    return load_variant("SCEPTR_CDR3_only_MLM_only")


def large():
    """
    Load the large variant of SCEPTR.
    This is a larger variant with model dimensionality 128 instead of 64.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the large model state loaded.
    """
    return load_variant("SCEPTR_large")


def small():
    """
    Load the small variant of SCEPTR.
    This is a smaller variant with model dimensioanlity 32 instead of 64.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the small model state loaded.
    """
    return load_variant("SCEPTR_small")


def tiny():
    """
    Load the tiny variant of SCEPTR.
    This is a tiny variant with model dimensionality 16 instead of 64.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the tiny model state loaded.
    """
    return load_variant("SCEPTR_tiny")


def blosum():
    """
    Load the BLOSUM variant of SCEPTR.
    The embedder module of this variant uses columns/rows of the BLOSUM62 substitution matrix to generate the initial vector embeddings of the amino acid residue tokens, instead of the one-hot solution used by the default.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the BLOSUM model state loaded.
    """
    return load_variant("SCEPTR_BLOSUM")


def average_pooling():
    """
    Load the average-pooling variant of SCEPTR.
    This variant uses the average-pooling method to generate TCR embeddings, instead of the `<cls>` pooling solution used by the default.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the average-pooling model state loaded.
    """
    return load_variant("SCEPTR_average_pooling")


def shuffled_data():
    """
    Load the shuffled-data variant of SCEPTR.
    This variant was trained on the same Tanno et al. dataset as the default, but with alpha-beta chain pairing randomised.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the shuffled-data model state loaded.
    """
    return load_variant("SCEPTR_shuffled_data")


def synthetic_data():
    """
    Load the synthetic-data variant of SCEPTR.
    This variant was trained on a size-matched set of synthetic TCR sequences generated by OLGA, a probabilistic model of VDJ recombination.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the synthetic-data model state loaded.
    """
    return load_variant("SCEPTR_synthetic_data")


def dropout_noise_only():
    """
    Load the dropout noise only variant of SCEPTR.
    This variant did not receive any residue/chain dropping noising signal during autocontrastive learning, and instead relied entirely on the model's internal dropout noise.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the dropout noise only model state loaded.
    """
    return load_variant("SCEPTR_dropout_noise_only")


def finetuned():
    """
    Load the fine-tuned variant of SCEPTR.
    This variant is fine-tuned using supervised contrastive learning for six pMHCs with peptides GILGFVFTL, NLVPMVATV, SPRWYFYYL, TFEYVSQPFLMDLE, TTDPSFLGRY and YLQPRTFLL (from `VDJdb <https://vdjdb.cdr3.net/>`_)

    .. note ::
        This model is fine-tuned explicity for discriminating between the pMHC specificities listed above.
        While it does indeed become much better than the default model at that specific task, its general performance is otherwise *demonstrably worse* compared to the default.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the fine-tuned model state loaded.
    """
    return load_variant("SCEPTR_finetuned")


def a_sceptr():
    """
    Load the alpha chain-only variant of SCEPTR.
    This variant has the same architecture as the default, but is specifically trained only with the alpha chain in distribution.

    .. note ::
        Because this model is trained only with the alpha chain in distribution, we expect it to perform slightly better than the default in settings where strictly only the alpha chains are available.
        However, in most cases where a mix of alpha, beta, and paired-chain data are available, we recommend using the default for ease of use.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the alpha chain-only model state loaded.
    """
    return load_variant("A_SCEPTR")


def b_sceptr():
    """
    Load the beta chain-only variant of SCEPTR.
    This variant has the same architecture as the default, but is specifically trained only with the beta chain in distribution.

    .. note ::
        Because this model is trained only with the beta chain in distribution, we expect it to perform slightly better than the default in settings where strictly only the beta chains are available.
        However, in most cases where a mix of alpha, beta, and paired-chain data are available, we recommend using the default for ease of use.

    Returns
    -------
    :py:class:`~sceptr.model.Sceptr`
        A :py:class:`~sceptr.model.Sceptr` instance with the beta chain-only model state loaded.
    """
    return load_variant("B_SCEPTR")
