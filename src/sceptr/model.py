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
    compartment_mask: ndarray
    representation_array: ndarray


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

    def calc_residue_representations(self, instances: DataFrame) -> ResidueRepresentations:
        """
        Given multiple TCRs, map each TCR to a set of amino acid residue-level representations.
        The residue-level representations are taken from the output of the penultimate self-attention layer, and are the same ones used by the :py:func:`~sceptr.variant.average_pooling` variant when generating TCR receptor-level representations.

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
            compartment_masks = padded_batch[:, :, 3]

            residue_reps_collection.append(residue_reps)
            compartment_masks_collection.append(compartment_masks)

        residue_reps_combined = torch.concatenate(residue_reps_collection, dim=0)
        compartment_masks_combined = torch.concatenate(compartment_masks_collection, dim=0)

        return ResidueRepresentations()

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
