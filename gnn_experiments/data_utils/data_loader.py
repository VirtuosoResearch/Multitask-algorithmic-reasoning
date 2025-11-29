import os 
import os.path as osp
import torch
import clrs
import numpy as np
import networkx as nx

import lightning.pytorch as pl
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from data_utils.utils import CLRSData, CLRSDataset, CLRSCollater

class CLRSDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch..
pip install importlib-resources
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset, # Union[Dataset, List[Data]]
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch = None, # Optional[List[str]]
        exclude_keys = None, # Optional[List[str]]
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=CLRSCollater(follow_batch,
                                             exclude_keys), **kwargs)
        

class CLRSDataModule(pl.LightningDataModule):
    """A Lightning DataModule for the CLRS dataset."""
    def __init__(self, train_dataset=None, val_datasets=None, test_datasets=None, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.reload_every_n_epochs = 0

        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        if not isinstance(test_datasets, list):
            self.test_datasets = [test_datasets]
        self.val_datasets = val_datasets

        self.test_batch_size = kwargs.pop("test_batch_size", None)
        self.kwargs = kwargs

        self._val_dataloaders = None
    
    def get_val_loader_nickname(self, idx):
        if isinstance(self.val_datasets, list):
            name = self.val_datasets[idx].nickname
        else:
            name = self.val_datasets.nickname
        return name if name else idx
    
    def get_test_loader_nickname(self, idx):
        name = self.test_datasets[idx].nickname
        return name if name else idx
    
    def dataloader(self, dataset: Dataset, **kwargs) -> CLRSDataLoader:
        return CLRSDataLoader(dataset, **kwargs)
    
    def train_dataloader(self) -> CLRSDataLoader:
        ds = self.train_dataset
        return self.dataloader(ds, shuffle=True, persistent_workers=True, **self.kwargs)
    
    def val_dataloader(self) -> CLRSDataLoader:
        if self._val_dataloaders is None:
            if isinstance(self.val_datasets, list):
                self._val_dataloaders = [self.dataloader(val_dataset, shuffle=False, persistent_workers=True, **self.kwargs) for val_dataset in self.val_datasets]
            else:
                self._val_dataloaders = self.dataloader(self.val_datasets, shuffle=False, persistent_workers=True, **self.kwargs)
        return self._val_dataloaders
    
    def test_dataloader(self) -> CLRSDataLoader:
        bs = self.test_batch_size
        kwargs = self.kwargs.copy()
        if bs is not None:
            kwargs["batch_size"] = bs
        
        kwargs["num_workers"] = 0 # we don't want to use multiprocessing for testing as there have been problems with shared memory
        return [self.dataloader(test_dataset, shuffle=False, **kwargs) for test_dataset in self.test_datasets]