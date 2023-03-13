from .embedder import _Embedder
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.nn.functional import normalize


class AtchleyEmbedder(_Embedder):
    """
    Simplest baseline TCR representation method, where a fixed-size vector
    representation of a TCR is obtained by averaging the atchley factors for
    all amino acids in the alpha CDR3 to produce one 5-dimensional vector,
    doing the same to the beta CDR3 to obtain another 5-dimensional vector,
    concatenating them, and l2-normalising the result.

    Compatible tokenisers: CDR3Tokeniser
    """

    def __init__(self) -> None:
        super().__init__()

        atchley_factors_with_special_tokens = torch.tensor(
            [
                [0.000, 0.000, 0.000, 0.000, 0.000],  # <pad>
                [0.000, 0.000, 0.000, 0.000, 0.000],  # <mask>
                [0.000, 0.000, 0.000, 0.000, 0.000],  # <cls>
                [-0.591, -1.302, -0.733, 1.570, -0.146],  # A
                [-1.343, 0.465, -0.862, -1.020, -0.255],  # C
                [1.050, 0.302, -3.656, -0.259, -3.242],  # D
                [1.357, -1.453, 1.477, 0.113, -0.837],  # E
                [-1.006, -0.590, 1.891, -0.397, 0.412],  # F
                [-0.384, 1.652, 1.330, 1.045, 2.064],  # G
                [0.336, -0.417, -1.673, -1.474, -0.078],  # H
                [-1.239, -0.547, 2.131, 0.393, 0.816],  # I
                [1.831, -0.561, 0.533, -0.277, 1.648],  # K
                [-1.019, -0.987, -1.505, 1.266, -0.912],  # L
                [-0.663, -1.524, 2.219, -1.005, 1.212],  # M
                [0.945, 0.828, 1.299, -0.169, 0.933],  # N
                [0.189, 2.081, -1.628, 0.421, -1.392],  # P
                [0.931, -0.179, -3.005, -0.503, -1.853],  # Q
                [1.538, -0.055, 1.502, 0.440, 2.897],  # R
                [-0.228, 1.399, -4.760, 0.670, -2.647],  # S
                [-0.032, 0.326, 2.213, 0.908, 1.313],  # T
                [-1.337, -0.279, -0.544, 1.242, -1.262],  # V
                [-0.595, 0.009, 0.672, -2.128, -0.184],  # W
                [0.260, 0.830, 3.097, -0.838, 1.512],
            ],  # Y
            dtype=torch.float32,
        )

        self.embedding = Embedding.from_pretrained(
            embeddings=atchley_factors_with_special_tokens, freeze=True, padding_idx=0
        )

    @property
    def name(self) -> str:
        return "Atchley Embedder"

    def embed(self, x: Tensor) -> Tensor:
        """
        :param x: Tensor representing a batch of tokenised TCRs. Expected
            shape is (B,L,2), where B is the batch size, L is the max length
            of the tokenised TCRs, and 2 corresponds to the fact that CDR3
            tokenisation produces two indices per CDR3 residue.
        :type x: Tensor
        :return: l2-normalised concatenation of the average atchley factors in
            the alpha CDR3 (dims 0-4) and beta CDR3 (dims 5-9).
        """

        padding_mask = x[:, :, [0]] >= 2
        alpha_mask = padding_mask * (x[:, :, [3]] == 1)
        beta_mask = padding_mask * (x[:, :, [3]] == 2)

        as_atchley = self.embedding(x[:, :, 0])

        # Get averaged atchley vector for CDR3A
        cdr3as_as_atchley = as_atchley * alpha_mask
        cdr3as_summed = cdr3as_as_atchley.sum(1)
        alpha_mask_sum = alpha_mask.sum(1)
        alpha_mask_sum = alpha_mask_sum + (alpha_mask_sum == 0)  # stop div by 0
        cdr3as_averaged = cdr3as_summed / alpha_mask_sum

        # Get averaged atchley vector for CDR3B
        cdr3bs_as_atchley = as_atchley * beta_mask
        cdr3bs_summed = cdr3bs_as_atchley.sum(1)
        beta_mask_sum = beta_mask.sum(1)
        beta_mask_sum = beta_mask_sum + (beta_mask_sum == 0)  # stop div by 0
        cdr3bs_averaged = cdr3bs_summed / beta_mask_sum

        # Combine
        both_chains = torch.cat([cdr3as_averaged, cdr3bs_averaged], dim=1)

        # l2 normalise and return
        return normalize(both_chains, p=2, dim=1)
