'''
Custom dataloader classes.
'''


import random
from . import datasets
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Union


class TCRDataLoader(DataLoader):
    '''
    Base dataloader class.
    '''
    def __init__(
        self,
        dataset: datasets.TCRDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )


    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        '''
        Pad and batch tokenised TCRs.
        '''
        elem = batch[0]

        if isinstance(elem, list) or isinstance(elem, tuple):
            return tuple(
                map(
                    lambda x: pad_sequence(
                        sequences=x,
                        batch_first=True,
                        padding_value=0
                    ),
                    zip(*batch)
                )
            )

        return pad_sequence(
            sequences=batch,
            batch_first=True,
            padding_value=0
        )


class MLMDataLoader(TCRDataLoader):
    '''
    Masked-language modelling dataloader class.
    '''
    def __init__(
        self,
        dataset: datasets.TCRDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        p_mask: float = 0.15,
        p_mask_random: float = 0.1,
        p_mask_keep: float = 0.1
    ):
        if p_mask < 0 or p_mask >= 1:
            raise RuntimeError(f'p_mask must lie in [0,1): {p_mask}')
        
        if p_mask_random < 0 or p_mask_random > 1:
            raise RuntimeError(
                f'p_mask_random must lie in [0,1]: {p_mask_random}'
            )

        if p_mask_keep < 0 or p_mask_keep > 1:
            raise RuntimeError(
                f'p_mask_keep must lie in [0,1]: {p_mask_keep}'
            )

        if p_mask_random + p_mask_keep > 1:
            raise RuntimeError(
                'p_mask_random + p_mask_keep must be less than 1.'
            )

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            num_workers
        )

        self._vocabulary = set(range(3, dataset._tokeniser.vocab_size+3))
        self._p_mask = p_mask
        self._p_mask_random = p_mask_random
        self._p_mask_keep = p_mask_keep


    def _pick_masking_indices(self, seq_len: int) -> list:
        '''
        Decide on a set of token indices to mask. Never mask the first token,
        as it is reserved for the <cls> token, which will not be used during
        MLM.
        '''
        if self._p_mask == 0:
            return []
        
        num_to_be_masked = max(1, round(seq_len * self._p_mask))
        return random.sample(range(1, seq_len), num_to_be_masked)


    def _generate_masked(self, x: Tensor, indices: list) -> Tensor:
        x = x.detach().clone()

        for idx in indices:
            r = random.random()
            if r < self._p_mask_random:
                x[idx,0] = random.choice(tuple(self._vocabulary-{x[idx,0]}))
                continue
            
            if r < 1-self._p_mask_keep:
                x[idx,0] = 1
                continue

        return x


    def _generate_target(self, x: Tensor, indices: list) -> Tensor:
        target = torch.zeros_like(x[:,0])

        for idx in indices:
            target[idx] = x[idx,0]
        
        return target


    def _make_mlm_pair(self, x: Tensor) -> Tuple[Tensor]:
        seq_len = len(x)

        indices_to_mask = self._pick_masking_indices(seq_len)

        masked = self._generate_masked(x, indices_to_mask)
        target = self._generate_target(x, indices_to_mask)

        return (masked, target)


    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        batch = [self._make_mlm_pair(x) for x in batch]

        return super().collate_fn(batch)


class AutoContrastiveDataLoader(MLMDataLoader):
    '''
    Dataloader for unsupervised contrastive loss training.
    '''
    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        batch = [
            (x_lhs, x_rhs, *self._make_mlm_pair(x))
            for x, x_lhs, x_rhs in batch
        ]

        return super(MLMDataLoader, self).collate_fn(batch)


class EpitopeAutoContrastiveSuperDataLoader:
    def __init__(
        self,
        dataset_ac: datasets.AutoContrastiveDataset,
        dataset_ec: datasets.EpitopeContrastiveDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers_ac: int = 0,
        num_workers_ec: int = 0,
        p_mask_ac: float = 0.15,
        p_mask_random_ac: float = 0.1,
        p_mask_keep_ac: float = 0.1
    ) -> None:
        self._dataloader_ac = AutoContrastiveDataLoader(
            dataset=dataset_ac,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers_ac,
            p_mask=p_mask_ac,
            p_mask_random=p_mask_random_ac,
            p_mask_keep=p_mask_keep_ac
        )
        self._dataloader_ec = TCRDataLoader(
            dataset=dataset_ec,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers_ec
        )

        self._len = max(len(self._dataloader_ac), len(self._dataloader_ec))


    def __len__(self) -> int:
        return self._len


    def __iter__(self) -> 'EpitopeAutoContrastiveSuperDataLoader':
        # To begin iterating, instantiate an iterator for both constituent
        # dataloaders, and keep track of how many iterations have been made
        self._iter_ac = iter(self._dataloader_ac)
        self._iter_ec = iter(self._dataloader_ec)
        self._iterations = 0
        return self

    
    def __next__(self) -> Tensor:
        # Beginning a new iteration
        self._iterations += 1

        # If we're about to loop around on the longer of the two constituent
        # dataloaders, that means we are done- raise StopIteration
        if self._iterations > len(self):
            raise StopIteration

        # Get a batch from the autocontrastive dataloader, loop back if needed
        try:
            data_ac = next(self._iter_ac)
        except StopIteration:
            self._iter_ac = iter(self._dataloader_ac)
            data_ac = next(self._iter_ac)

        # Get a batch from the epitope-labelled dataloader, loop back if needed
        try:
            data_ec = next(self._iter_ec)
        except StopIteration:
            self._iter_ec = iter(self._dataloader_ec)
            data_ec = next(self._iter_ec)
        
        # Return combined batch
        return (*data_ac, *data_ec)