from torch import BoolTensor, FloatTensor, LongTensor
from torch.nn import Module

from blastr._lib.nn.data.tokeniser.token_indices import DefaultTokenIndex
from blastr._lib.nn.token_embedder.token_embedder import TokenEmbedder
from blastr._lib.nn.mlm_token_prediction_projector import MlmTokenPredictionProjector
from blastr._lib.nn.self_attention_stack import SelfAttentionStack
from blastr._lib.nn.vector_representation_delegate import VectorRepresentationDelegate


class Bert(Module):
    def __init__(
        self,
        token_embedder: TokenEmbedder,
        self_attention_stack: SelfAttentionStack,
        mlm_token_prediction_projector: MlmTokenPredictionProjector,
        vector_representation_delegate: VectorRepresentationDelegate,
    ) -> None:
        super().__init__()

        self._token_embedder = token_embedder
        self._self_attention_stack = self_attention_stack
        self._mlm_token_prediction_projector = mlm_token_prediction_projector
        self._vector_representation_delegate = vector_representation_delegate

    @property
    def d_model(self) -> int:
        return self._self_attention_stack.d_model

    def get_vector_representations_of(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        raw_token_embeddings = self._embed(tokenised_tcrs)
        padding_mask = self._get_padding_mask(tokenised_tcrs)
        vector_representations = (
            self._vector_representation_delegate.get_vector_representations_of(
                raw_token_embeddings, padding_mask
            )
        )

        return vector_representations

    def get_mlm_token_predictions_for(
        self, tokenised_and_masked_tcrs: LongTensor
    ) -> FloatTensor:
        raw_token_embeddings = self._embed(tokenised_and_masked_tcrs)
        padding_mask = self._get_padding_mask(tokenised_and_masked_tcrs)
        contextualised_token_embeddings = self._self_attention_stack.forward(
            raw_token_embeddings, padding_mask
        )
        mlm_token_predictions = self._mlm_token_prediction_projector.forward(
            contextualised_token_embeddings
        )

        return mlm_token_predictions

    def _embed(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        return self._token_embedder.forward(tokenised_tcrs)

    def _get_padding_mask(self, tokenised_tcrs: LongTensor) -> BoolTensor:
        return tokenised_tcrs[:, :, 0] == DefaultTokenIndex.NULL
