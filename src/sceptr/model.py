import torch
from torch import FloatTensor
from torch.nn import utils
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from libtcrlm.bert import Bert
from libtcrlm.tokeniser import Tokeniser
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex
from libtcrlm import schema


BATCH_SIZE = 512


class ResidueRepresentations:
    """
    An object containing information necessary to interpret and operate on residue-level representations from the SCEPTR family of models.
    Instances of this class can be obtained via the :py:func:`sceptr.calc_residue_representations` function and a method of the same name on the :py:class:`~sceptr.model.Sceptr` class.

    This feature is implemented for the curious users who would like to tinker around and examine what kind of information SCEPTR focuses on at the individual amino acid residue level, and do so without completely hacking into the source code of :py:mod:`sceptr`.
    For some examples of how to use instances of this class to make useful examinations of SCEPTR's residue-level embeddings, please refer to the "Examples" section below.

    Attributes
    ----------
    representation_array : ndarray
        A numpy float ndarray containing the residue-level representation data.
        The array is of shape :math:`(N, M, D)` where :math:`N` is the number of TCRs in the original input, :math:`M` is the maximum number of residues amongst the input TCRs when put into its tokenised form, and :math:`D` is the dimensionality of the model variant that produced the result.

    compartment_mask : ndarray
        A numpy integer array containing information on which indices in the `representation_array` correspond to tokens that come from each CDR loop of the input TCRs.
        The array is of shape :math:`(N, M)` where :math:`N` is the number of TCRs in the original input, and :math:`M` is the maximum number of residues amongst the input TCRs when put into its tokenised form.
        Entries in `compartment_mask` have the following values:

        +------------------------------+------------------+
        | If residue at index is from: | Entry has value: |
        +==============================+==================+
        | None (padding token)         | 0                |
        +------------------------------+------------------+
        | CDR1A                        | 1                |
        +------------------------------+------------------+
        | CDR2A                        | 2                |
        +------------------------------+------------------+
        | CDR3A                        | 3                |
        +------------------------------+------------------+
        | CDR1B                        | 4                |
        +------------------------------+------------------+
        | CDR2B                        | 5                |
        +------------------------------+------------------+
        | CDR3B                        | 6                |
        +------------------------------+------------------+

        Within each CDR loop compartment, residues are ordered from C- to N-terminal from left to right.

    Examples
    --------
    As an example, let's see how one could get the residue-level representations for the beta-chain CDR3 amino acid sequences of all input TCR sequences.
    Say we have some DataFrame ``tcrs`` that contains the sequence data for four TCRs.

    >>> from pandas import DataFrame
    >>> tcrs = DataFrame(
    ... 	data = {
    ... 		"TRAV": ["TRAV38-1*01", "TRAV3*01", "TRAV13-2*01", "TRAV38-2/DV8*01"],
    ... 		"CDR3A": ["CAHRSAGGGTSYGKLTF", "CAVDNARLMF", "CAERIRKGQVLTGGGNKLTF", "CAYRSAGGGTSYGKLTF"],
    ... 		"TRBV": ["TRBV2*01", "TRBV25-1*01", "TRBV9*01", "TRBV2*01"],
    ... 		"CDR3B": ["CASSEFQGDNEQFF", "CASSDGSFNEQFF", "CASSVGDLLTGELFF", "CASSPGTGGNEQYF"],
    ... 	},
    ... 	index = [0,1,2,3]
    ... )
    >>> print(tcrs)
                  TRAV                 CDR3A         TRBV            CDR3B
    0      TRAV38-1*01     CAHRSAGGGTSYGKLTF     TRBV2*01   CASSEFQGDNEQFF
    1         TRAV3*01            CAVDNARLMF  TRBV25-1*01    CASSDGSFNEQFF
    2      TRAV13-2*01  CAERIRKGQVLTGGGNKLTF     TRBV9*01  CASSVGDLLTGELFF
    3  TRAV38-2/DV8*01     CAYRSAGGGTSYGKLTF     TRBV2*01   CASSPGTGGNEQYF

    We can get the residue-level representations for those TCRs like so:

    >>> import sceptr
    >>> res_reps = sceptr.calc_residue_representations(tcrs)

    Now, we can iterate through the residue-level representation subarray corresponding to each TCR, and filter out/obtain the representations for the beta chain CDR3 sequence.

    >>> cdr3b_reps = []
    >>> for reps, mask in zip(res_reps.representation_array, res_reps.compartment_mask):
    ...     # reps.shape == (M, D)
    ...     # mask.shape == (M,)
    ...     cdr3b_rep = reps[mask == 6] # collect only the residue representations for the beta CDR3 sequence
    ...     cdr3b_reps.append(cdr3b_rep)

    Now we have a list containing four numpy ndarrays, each of which is a matrix whose row vectors are representations of individual CDR3B amino acid residues.

    >>> type(cdr3b_reps[0])
    numpy.ndarray
    >>> cdr3b_reps[0].shape
    (14, 64)

    Note that the zeroth element of the shape tuple above is 14 because the CDR3B sequence of the first TCR in ``tcrs`` is 14 residues long, and the first element of the shape tuple is 64 because the model dimensionality of the default SCEPTR variant is 64.
    """
    representation_array: ndarray
    compartment_mask: ndarray

    def __init__(self, representation_array: ndarray, compartment_mask: ndarray) -> None:
        self.representation_array = representation_array
        self.compartment_mask = compartment_mask


class Sceptr:
    """
    Loads a trained state of a SCEPTR (variant) and provides an easy interface for generating TCR representations and making inferences from them.
    Instances can be obtained through the :py:mod:`sceptr.variant` submodule.

    Attributes
    ----------
    name : str
        The name of the model variant.
    """

    name: str = None
    distance_bins = np.linspace(0, 2, num=21)

    def __init__(
        self, name: str, tokeniser: Tokeniser, bert: Bert, device: torch.device
    ) -> None:
        self.name = name
        self._tokeniser = tokeniser
        self._bert = bert.eval()
        self._device = device

    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        """
        Map TCRs to their corresponding vector representations.

        Parameters
        ----------
        instances : DataFrame
            DataFrame in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        ndarray
            A 2D numpy ndarray object where every row vector corresponds to a row in `instances`.
            The returned array will have shape :math:`(N, D)` where :math:`N` is the number of TCRs in `instances` and :math:`D` is the dimensionality of the current model variant.
        """
        torch_representations = self._calc_torch_representations(instances)
        return torch_representations.cpu().numpy()

    @torch.no_grad()
    def calc_residue_representations(self, instances: DataFrame) -> ResidueRepresentations:
        """
        Given multiple TCRs, map each TCR to a set of amino acid residue-level representations.
        The residue-level representations are taken from the output of the penultimate self-attention layer, and are the same ones used by the :py:func:`~sceptr.variant.average_pooling` variant when generating TCR receptor-level representations.

        .. note ::
            This method is currently only supported on SCEPTR model variants such as the default one that 1) use both the alpha and beta chains, and 2) take into account all three CDR loops from each chain.

        Parameters
        ----------
        instances : DataFrame
            DataFrame in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        :py:class:`~sceptr.model.ResidueRepresentations`
            For details on how to interpret/use this output, please refer to the documentation for :py:class:`~sceptr.model.ResidueRepresentations`.
        """
        instances = instances.copy()

        for col in ("TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"):
            if col not in instances:
                instances[col] = None

        tcrs = schema.generate_tcr_series(instances)

        residue_reps_collection = []
        compartment_masks_collection = []

        for idx in range(0, len(tcrs), BATCH_SIZE):
            batch = tcrs.iloc[idx : idx + BATCH_SIZE]
            tokenised_batch = [self._tokeniser.tokenise(tcr) for tcr in batch]
            padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised_batch,
                batch_first=True,
                padding_value=DefaultTokenIndex.NULL,
            ).to(self._device)

            raw_token_embeddings = self._bert._embed(padded_batch)
            padding_mask = self._bert._get_padding_mask(padded_batch)

            residue_reps = self._bert._self_attention_stack.get_token_embeddings_at_penultimate_layer(raw_token_embeddings, padding_mask)
            residue_reps = residue_reps[:, 1:, :]

            compartment_masks = padded_batch[:, 1:, 3]

            residue_reps_collection.append(residue_reps)
            compartment_masks_collection.append(compartment_masks)

        residue_reps_combined = torch.concatenate(residue_reps_collection, dim=0).cpu().numpy()
        compartment_masks_combined = torch.concatenate(compartment_masks_collection, dim=0).cpu().numpy()

        return ResidueRepresentations(residue_reps_combined, compartment_masks_combined)

    @torch.no_grad()
    def _calc_torch_representations(self, instances: DataFrame) -> FloatTensor:
        instances = instances.copy()

        for col in ("TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"):
            if col not in instances:
                instances[col] = None

        tcrs = schema.generate_tcr_series(instances)

        representations = []
        for idx in range(0, len(tcrs), BATCH_SIZE):
            batch = tcrs.iloc[idx : idx + BATCH_SIZE]
            tokenised_batch = [self._tokeniser.tokenise(tcr) for tcr in batch]
            padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised_batch,
                batch_first=True,
                padding_value=DefaultTokenIndex.NULL,
            )
            batch_representation = self._bert.get_vector_representations_of(
                padded_batch.to(self._device)
            )
            representations.append(batch_representation)

        return torch.concatenate(representations, dim=0)

    def calc_cdist_matrix(self, anchors: DataFrame, comparisons: DataFrame) -> ndarray:
        """
        Generate a cdist matrix between two collections of TCRs.

        Parameters
        ----------
        anchors : DataFrame
            DataFrame in the :ref:`prescribed format <data_format>`.
        comparisons : DataFrame
            DataFrame in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        ndarray
            A 2D numpy ndarray representing a cdist matrix between TCRs from `anchors` and `comparisons`.
            The returned array will have shape :math:`(X, Y)` where :math:`X` is the number of TCRs in `anchors` and :math:`Y` is the number of TCRs in `comparisons`.
        """
        anchor_representations = self._calc_torch_representations(anchors)
        comparison_representations = self._calc_torch_representations(comparisons)
        cdist_matrix = torch.cdist(
            anchor_representations, comparison_representations, p=2
        )
        return cdist_matrix.cpu().numpy()

    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        r"""
        Generate a pdist vector of distances between each pair of TCRs in the input data.

        Parameters
        ----------
        instances : DataFrame
            DataFrame in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        ndarray
            A 1D numpy ndarray representing a pdist vector of distances between each pair of TCRs in `instances`.
            The returned array will have shape :math:`(\frac{1}{2}N(N-1),)`, where :math:`N` is the number of TCRs in `instances`.
        """
        representations = self._calc_torch_representations(instances)
        pdist_vector = torch.pdist(representations, p=2)
        return pdist_vector.cpu().numpy()
