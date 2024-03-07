from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame

from sceptr._lib.tcr_metric import TcrMetric


class TcrRepresentationModel(TcrMetric):
    def calc_cdist_matrix(
        self, anchors: DataFrame, comparisons: DataFrame
    ) -> ndarray:
        super().calc_cdist_matrix(anchors, comparisons)
        anchor_tcr_representations = self.calc_vector_representations(anchors)
        comparison_tcr_representations = self.calc_vector_representations(
            comparisons
        )
        return self.calc_cdist_matrix_from_representations(
            anchor_tcr_representations, comparison_tcr_representations
        )

    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        super().calc_pdist_vector(instances)
        tcr_representations = self.calc_vector_representations(instances)
        return self.calc_pdist_vector_from_representations(tcr_representations)

    @abstractmethod
    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        pass

    @abstractmethod
    def calc_cdist_matrix_from_representations(
        self,
        anchor_tcr_representations: ndarray,
        comparison_tcr_representations: ndarray,
    ) -> ndarray:
        pass

    @abstractmethod
    def calc_pdist_vector_from_representations(
        self, tcr_representations: ndarray
    ) -> ndarray:
        pass
