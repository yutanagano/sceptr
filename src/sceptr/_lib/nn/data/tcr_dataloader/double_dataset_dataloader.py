import torch
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from typing import Iterable, Tuple

from sceptr._lib.nn.data.tcr_dataset import TcrDataset
from sceptr._lib.nn.data.batch_collator import BatchCollator
from sceptr._lib.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class DoubleDatasetDataLoader:
    def __init__(
        self,
        dataset_1: TcrDataset,
        dataset_2: TcrDataset,
        batch_collator: BatchCollator,
        device: torch.device,
        batch_size_1: int,
        batch_size_2: int,
        num_workers_per_dataset: int,
        distributed: bool,
    ) -> None:
        self._batch_collator = batch_collator
        self._device = device
        self._dataloader_1 = self._make_internal_dataloader_for_dataset(
            dataset_1, batch_size_1, num_workers_per_dataset, distributed
        )
        self._dataloader_2 = self._make_internal_dataloader_for_dataset(
            dataset_2, batch_size_2, num_workers_per_dataset, distributed
        )

        self._len = max(len(self._dataloader_1), len(self._dataloader_2))

    def _make_internal_dataloader_for_dataset(
        self, dataset: TcrDataset, batch_size: int, num_workers: int, distributed: bool
    ) -> DataLoader:
        if distributed:
            sampler = DistributedSampler(dataset=dataset, shuffle=True)
        else:
            sampler = None

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._no_op_collation,
        )

    @staticmethod
    def _no_op_collation(
        tcr_pmhc_pairs: Iterable[TcrPmhcPair],
    ) -> Iterable[TcrPmhcPair]:
        return tcr_pmhc_pairs

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> "DoubleDatasetDataLoader":
        self._iterator_1 = iter(self._dataloader_1)
        self._iterator_2 = iter(self._dataloader_2)
        self._iterations = 0
        return self

    def __next__(self) -> Tuple[Tensor]:
        self._iterations += 1

        finished_epoch = self._iterations > len(self)
        if finished_epoch:
            raise StopIteration

        try:
            batch_1 = next(self._iterator_1)
        except StopIteration:
            self._iterator_1 = iter(self._dataloader_1)
            batch_1 = next(self._iterator_1)

        try:
            batch_2 = next(self._iterator_2)
        except StopIteration:
            self._iterator_2 = iter(self._dataloader_2)
            batch_2 = next(self._iterator_2)

        combined_batch = [*batch_1, *batch_2]

        return self._batch_collator.collate_fn(combined_batch)

    def set_epoch(self, epoch: int) -> None:
        self._dataloader_1.sampler.set_epoch(epoch)
        self._dataloader_2.sampler.set_epoch(epoch)
