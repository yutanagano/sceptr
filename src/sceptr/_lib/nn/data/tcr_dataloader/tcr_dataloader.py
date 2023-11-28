from abc import ABC, abstractmethod


class TcrDataLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def set_epoch(self, epoch: int) -> None:
        pass
