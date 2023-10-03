from torch import Tensor
from torch.nn import functional as F

from blastr._lib.nn.self_attention_stack import SelfAttentionStack
from blastr._lib.nn.vector_representation_delegate import VectorRepresentationDelegate


LOCATION_OF_CLS_TOKEN = 0


class ClsVectorRepresentationDelegate(VectorRepresentationDelegate):
    def __init__(self, self_attention_stack: SelfAttentionStack) -> None:
        self._self_attention_stack = self_attention_stack

    def get_vector_representations_of(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        final_token_embeddings = self._self_attention_stack.forward(
            token_embeddings, padding_mask
        )
        final_cls_embeddings = final_token_embeddings[:, LOCATION_OF_CLS_TOKEN, :]
        l2_normed_cls_embeddings = F.normalize(final_cls_embeddings, p=2, dim=1)

        return l2_normed_cls_embeddings
