"""
Wrapper interface to pytorch models
"""


from .datahandling.dataloaders import TCRDataLoader
from .datahandling.datasets import TCRDataset
from .datahandling import tokenisers
from . import models
from numpy import ndarray
from pandas import DataFrame
import torch


class ModelWrapper:
    '''
    Pytorch embedder model wrapper to faciliate model sharing between different
    evaluation pipelines.
    '''
    def __init__(self, config: dict, state_dict: dict) -> None:
        # Instantiate model
        model = getattr(models, config['model']['class'])(
            **config['model']['config']
        )

        # Load weights
        model.load_state_dict(state_dict)

        # Transfer to GPU if possible
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        model.to(device)

        # Evaluation mode
        model.eval()

        # Instantiate tokeniser
        tokeniser = getattr(tokenisers, config['data']['tokeniser']['class'])(
            **config['data']['tokeniser']['config']
        )

        self.model = model
        self._tokeniser = tokeniser


    @property
    def name(self) -> str:
        return self.model.name
    

    def embed(self, data: DataFrame) -> ndarray:
        dl = self._generate_dataloader(data)
        return self._generate_embeddings(dl).detach().numpy()

    
    def pdist(self, data: DataFrame) -> ndarray:
        dl = self._generate_dataloader(data)
        embedded = self._generate_embeddings(dl)

        return torch.pdist(embedded, p=2).detach().numpy()


    def cdist(
        self,
        data_a: DataFrame,
        data_b: DataFrame
    ) -> ndarray:
        dl_1 = self._generate_dataloader(data_a)
        dl_2 = self._generate_dataloader(data_b)

        embedded_1 = self._generate_embeddings(dl_1)
        embedded_2 = self._generate_embeddings(dl_2)

        return torch.cdist(embedded_1, embedded_2, p=2).detach().numpy()


    def _generate_dataloader(self, data: DataFrame) -> TCRDataLoader:
        # Create missing columns if any and mark as empty
        for col in (
            'TRAV',
            'CDR3A',
            'TRAJ',
            'TRBV',
            'CDR3B',
            'TRBJ',
            'Epitope',
            'MHCA',
            'MHCB',
            'duplicate_count'
        ):
            if col not in data:
                data[col] = None
        
        # Generate dataset then wrap in dataloader
        ds = TCRDataset(data=data, tokeniser=self._tokeniser)
        return TCRDataLoader(dataset=ds, batch_size=512, shuffle=False)


    @torch.no_grad()
    def _generate_embeddings(self, dataloader: TCRDataLoader) -> torch.Tensor:
        embedded = [self.model.embed(batch) for batch in dataloader]
        return torch.concat(embedded)