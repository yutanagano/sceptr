from torch import FloatTensor, LongTensor
from torch.nn import Module
from typing import Tuple

from sceptr._lib.nn.bert import Bert


class TrainableModel(Module):
    def __init__(self, bert: Bert) -> None:
        super().__init__()
        self.bert = bert


class MlmTrainableModel(TrainableModel):
    def forward(self, tokenised_and_masked_tcrs: LongTensor) -> FloatTensor:
        return self.bert.get_mlm_token_predictions_for(tokenised_and_masked_tcrs)


class ClTrainableModel(TrainableModel):
    def forward(
        self,
        tokenised_tcrs: LongTensor,
        tokenised_and_masked_tcrs: LongTensor,
    ) -> Tuple[FloatTensor]:
        tcr_representations = self.bert.get_vector_representations_of(tokenised_tcrs)
        mlm_logits = self.bert.get_mlm_token_predictions_for(tokenised_and_masked_tcrs)
        return tcr_representations, mlm_logits
