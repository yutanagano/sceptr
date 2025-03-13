from libtcrlm.bert import Bert
from libtcrlm.tokeniser import Tokeniser, CdrTokeniser
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex
from libtcrlm import schema
import logging
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
import torch
from torch import FloatTensor
from torch.nn import utils


BATCH_SIZE_DEFAULT = 512


logger = logging.getLogger(__name__)


class ResidueRepresentations:
    """
    An object containing information necessary to interpret and operate on residue-level representations from the SCEPTR family of models.
    Instances of this class can be obtained via the :py:func:`sceptr.calc_residue_representations` function and a method of the same name on the :py:class:`~sceptr.model.Sceptr` class.

    This feature is implemented to give power-users easy access to model internals to tinker around and examine what kind of information SCEPTR focuses on at the individual amino acid residue level.
    The "Examples" section below illustrates how to use instances of this class to examine SCEPTR's residue-level embeddings.

    Attributes
    ----------
    representation_array : NDArray[numpy.float32]
        A numpy float array containing the residue-level representation data.
        The array is of shape :math:`(N, M, D)` where :math:`N` is the number of TCRs in the original input, :math:`M` is the maximum number of residues among the input TCRs when put into its tokenised form, and :math:`D` is the dimensionality of the model variant that produced the result.

    compartment_mask : NDArray[numpy.int64]
        A numpy integer array mapping residue indices in the `representation_array` to corresponding CDR loops of the input TCRs.
        The array is of shape :math:`(N, M)` where :math:`N` is the number of TCRs in the original input, and :math:`M` is the maximum number of residues among the input TCRs when put into its tokenised form.
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
    In the following we show how to extract the residue-level representations for the beta-chain CDR3 amino acid sequences of all input TCR sequences.
    To start with, we define a DataFrame ``tcrs`` that contains the sequence data for four TCRs.

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
    >>> print(res_reps)
    ResidueRepresentations[num_tcrs: 4, rep_dim: 64]

    Now, we can iterate through the residue-level representation subarray corresponding to each TCR, and filter out/obtain the representations for the beta chain CDR3 sequence.

    >>> cdr3b_reps = []
    >>> for reps, mask in zip(res_reps.representation_array, res_reps.compartment_mask):
    ...     cdr3b_rep = reps[mask == 6] # collect only the residue representations for the beta CDR3 sequence
    ...     cdr3b_reps.append(cdr3b_rep)

    Now we have a list containing four numpy ndarrays, each of which is a matrix whose row vectors are representations of individual CDR3B amino acid residues.

    >>> type(cdr3b_reps[0])
    <class 'numpy.ndarray'>
    >>> cdr3b_reps[0].shape
    (14, 64)

    Note that the zeroth element of the shape tuple above is 14 because the CDR3B sequence of the first TCR in ``tcrs`` is 14 residues long, and the first element of the shape tuple is 64 because the model dimensionality of the default SCEPTR variant is 64.
    """

    representation_array: NDArray[np.float32]
    compartment_mask: NDArray[np.int64]

    def __init__(
        self,
        representation_array: NDArray[np.float32],
        compartment_mask: NDArray[np.int64],
    ) -> None:
        self.representation_array = representation_array
        self.compartment_mask = compartment_mask

    def __repr__(self) -> str:
        return f"ResidueRepresentations[num_tcrs: {self.representation_array.shape[0]}, rep_dim: {self.representation_array.shape[2]}]"


class Sceptr:
    """
    Loads a trained state of a SCEPTR model and provides an easy interface for generating TCR representations and making inferences from them.
    Instances can be obtained through the :py:mod:`sceptr.variant` submodule.

    Attributes
    ----------
    name : str
        The name of the model variant.
    """

    name: str = None
    distance_bins = np.linspace(0, 2, num=21)

    def __init__(self, name: str, tokeniser: Tokeniser, bert: Bert) -> None:
        self.name = name
        self._tokeniser = tokeniser
        self._bert = bert.eval()
        self._device = torch.device("cpu")
        self._batch_size = BATCH_SIZE_DEFAULT

    def enable_hardware_acceleration(self) -> None:
        """
        Move this `Sceptr` instance and its computations to a hardware-accelerated device, if available (e.g. CUDA-enabled GPU).
        For toggling the package-level setting, see :py:func:`sceptr.enable_hardware_acceleration`.
        """
        self._device = _get_hardware_accelerated_device()
        self._bert.to(self._device)
        logger.debug(
            f"enable_hardware_acceleration called on {self} ({self.name}), setting device to {self._device}"
        )

    def disable_hardware_acceleration(self) -> None:
        """
        Move this `Sceptr` instance and its computations to the CPU.
        For toggling the package-level setting, see :py:func:`sceptr.disable_hardware_acceleration`.
        """
        self._device = torch.device("cpu")
        self._bert.to(self._device)
        logger.debug(
            f"disable_hardware_acceleration called on {self} ({self.name}), setting device to cpu"
        )

    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the batch size used when generating TCR vector representations.
        That is, how many representations are computed at a time on the CPU / GPU.
        By default, the batch size is set to 512.
        """
        if not isinstance(batch_size, int):
            raise TypeError(f"The batch size must be an int. Got {type(batch_size)}.")

        self._batch_size = batch_size

    def calc_vector_representations(self, instances: DataFrame) -> NDArray[np.float32]:
        """
        Map TCRs to their corresponding vector representations.

        Parameters
        ----------
        instances : DataFrame
            DataFrame specifying the input TCRs.
            It must be in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        NDArray[numpy.float32]
            A 2D numpy ndarray object where every row vector corresponds to a row in `instances`.
            The returned array will have shape :math:`(N, D)` where :math:`N` is the number of TCRs in `instances` and :math:`D` is the dimensionality of the current model variant.
        """
        torch_representations = self._calc_torch_representations(instances)
        return torch_representations.cpu().numpy()

    @torch.no_grad()
    def calc_residue_representations(
        self, instances: DataFrame
    ) -> ResidueRepresentations:
        """
        Map each TCR to a set of amino acid residue-level representations.
        The residue-level representations are the output of the penultimate self-attention layer, as also used by the :py:func:`~sceptr.variant.average_pooling` variant when generating TCR receptor-level representations.

        .. note ::
            This method is currently only supported on SCEPTR model variants such as the default one that 1) use both the alpha and beta chains, and 2) take into account all three CDR loops from each chain.

        Parameters
        ----------
        instances : DataFrame
            DataFrame specifying the input TCRs.
            It must be in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        :py:class:`~sceptr.model.ResidueRepresentations`
            An array of representation vectors for each amino acid residue in the tokenised forms of the input TCRs.
            For details on how to interpret/use this output, please refer to the documentation for :py:class:`~sceptr.model.ResidueRepresentations`.
        """
        if not isinstance(self._tokeniser, CdrTokeniser):
            raise NotImplementedError(
                "The calc_residue_representations method is currently only supported on SCEPTR model variants that 1) use both the alpha and beta chains, and 2) take into account all three CDR loops from each chain."
            )

        instances = instances.copy()

        for col in ("TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"):
            if col not in instances:
                instances[col] = None

        tcrs = schema.generate_tcr_series(instances)

        residue_reps_collection = []
        compartment_masks_collection = []

        for idx in range(0, len(tcrs), self._batch_size):
            batch = tcrs.iloc[idx : idx + self._batch_size]
            tokenised_batch = [self._tokeniser.tokenise(tcr) for tcr in batch]
            padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised_batch,
                batch_first=True,
                padding_value=DefaultTokenIndex.NULL,
            ).to(self._device)

            raw_token_embeddings = self._bert._embed(padded_batch)
            padding_mask = self._bert._get_padding_mask(padded_batch)

            residue_reps = self._bert._self_attention_stack.get_token_embeddings_at_penultimate_layer(
                raw_token_embeddings, padding_mask
            )
            residue_reps = residue_reps[:, 1:, :]

            compartment_masks = padded_batch[:, 1:, 3]

            residue_reps_collection.append(residue_reps)
            compartment_masks_collection.append(compartment_masks)

        residue_reps_combined = (
            torch.concatenate(residue_reps_collection, dim=0).cpu().numpy()
        )
        compartment_masks_combined = (
            torch.concatenate(compartment_masks_collection, dim=0).cpu().numpy()
        )

        return ResidueRepresentations(residue_reps_combined, compartment_masks_combined)

    @torch.no_grad()
    def _calc_torch_representations(self, instances: DataFrame) -> FloatTensor:
        instances = instances.copy()

        for col in ("TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"):
            if col not in instances:
                instances[col] = None

        tcrs = schema.generate_tcr_series(instances)

        representations = []
        for idx in range(0, len(tcrs), self._batch_size):
            batch = tcrs.iloc[idx : idx + self._batch_size]
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

    def calc_cdist_matrix(
        self, anchors: DataFrame, comparisons: DataFrame
    ) -> NDArray[np.float32]:
        """
        Generate a cdist matrix between two collections of TCRs.

        Parameters
        ----------
        anchors : DataFrame
            DataFrame specifying the first (anchor) collection of input TCRs.
            It must be in the :ref:`prescribed format <data_format>`.

        comparisons : DataFrame
            DataFrame specifying the second (comparison) collection of input TCRs.
            It must be in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        NDArray[numpy.float32]
            A 2D numpy ndarray representing a cdist matrix between TCRs from `anchors` and `comparisons`.
            The returned array will have shape :math:`(X, Y)` where :math:`X` is the number of TCRs in `anchors` and :math:`Y` is the number of TCRs in `comparisons`.
        """
        anchor_representations = self._calc_torch_representations(anchors)
        comparison_representations = self._calc_torch_representations(comparisons)
        cdist_matrix = torch.cdist(
            anchor_representations, comparison_representations, p=2
        )
        return cdist_matrix.cpu().numpy()

    def calc_pdist_vector(self, instances: DataFrame) -> NDArray[np.float32]:
        r"""
        Generate a pdist vector of distances between each pair of TCRs in the input data.

        Parameters
        ----------
        instances : DataFrame
            DataFrame specifying the input TCRs.
            It must be in the :ref:`prescribed format <data_format>`.

        Returns
        -------
        NDArray[numpy.float32]
            A 1D numpy ndarray representing a pdist vector of distances between each pair of TCRs in `instances`.
            The returned array will have shape :math:`(\frac{1}{2}N(N-1),)`, where :math:`N` is the number of TCRs in `instances`.
        """
        representations = self._calc_torch_representations(instances)
        pdist_vector = torch.pdist(representations, p=2)
        return pdist_vector.cpu().numpy()


def _get_hardware_accelerated_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Below is commented out for now until MPS support is more stable / complete
    # if torch.mps.is_available():
    #     return torch.device("mps")

    return torch.device("cpu")
