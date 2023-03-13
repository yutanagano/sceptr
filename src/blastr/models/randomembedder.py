from .embedder import _Embedder
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.nn.functional import normalize


class RandomEmbedder(_Embedder):
    '''
    Simplest baseline TCR representation method, where a fixed-size vector
    representation of a TCR is obtained by averaging the randomly initialised
    vector representations of the amino acids in the alpha CDR3 to produce one
    dim-dimensional vector, doing the same to the beta CDR3 to obtain another
    dim-dimensional vector, concatenating them, and l2-normalising the result.

    Compatible tokenisers: CDR3Tokeniser
    '''


    def __init__(self, dim: int = 5, name_idx: int = 0) -> None:
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=23, # <pad> + <mask> + <cls> + 20 amino acids
            embedding_dim=dim,
            padding_idx=0
        )
        self.embedding.weight.requires_grad = False
        self._name_idx = name_idx


    @property
    def name(self) -> str:
        return f'Random Embedder {self._name_idx}'


    def embed(self, x: Tensor) -> Tensor:
        '''
        :param x: Tensor representing a batch of tokenised TCRs. Expected
            shape is (B,L,2), where B is the batch size, L is the max length
            of the tokenised TCRs, and 2 corresponds to the fact that CDR3
            tokenisation produces two indices per CDR3 residue.
        :type x: Tensor
        :return: l2-normalised concatenation of the averaged randomly
            initialised vector representations of the amino acids in the alpha
            CDR3 (dims 0-4) and beta CDR3 (dims 5-9).
        '''

        padding_mask = (x[:,:,[0]] >= 2)
        alpha_mask = padding_mask * (x[:,:,[1]] == 1)
        beta_mask = padding_mask * (x[:,:,[1]] == 2)

        embedded = self.embedding(x[:,:,0])

        # Get averaged atchley vector for CDR3A
        cdr3as_as_atchley = embedded * alpha_mask
        cdr3as_summed = cdr3as_as_atchley.sum(1)
        alpha_mask_sum = alpha_mask.sum(1)
        alpha_mask_sum = alpha_mask_sum + (alpha_mask_sum == 0) # stop div by 0
        cdr3as_averaged = cdr3as_summed / alpha_mask_sum

        # Get averaged atchley vector for CDR3B
        cdr3bs_as_atchley = embedded * beta_mask
        cdr3bs_summed = cdr3bs_as_atchley.sum(1)
        beta_mask_sum = beta_mask.sum(1)
        beta_mask_sum = beta_mask_sum + (beta_mask_sum == 0) # stop div by 0
        cdr3bs_averaged = cdr3bs_summed / beta_mask_sum

        # Combine
        both_chains = torch.cat([cdr3as_averaged, cdr3bs_averaged], dim=1)

        # l2 normalise and return
        return normalize(both_chains, p=2, dim=1)