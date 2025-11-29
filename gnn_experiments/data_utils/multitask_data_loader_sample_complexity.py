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

from torch.utils.data import DataLoader
from data_utils.utils import CLRSData, CLRSDataset, CLRSCollater
from data_utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator

from sample_complexity.baselines.salsaclrs.data import SALSACLRSDataModule, SALSACLRSDataset, SALSACLRSDataLoader

class MultiCLRSDataModuleSampleComplexity(pl.LightningDataModule):
    """A Lightning DataModule for the CLRS dataset."""
    def __init__(self, algorithms, data_dir, num_samples, node, graph_batch_dir, **kwargs):
        super().__init__()
        self.algorithms = algorithms
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.node = node
        self.graph_batch_dir = graph_batch_dir
        self.use_complete_graph = kwargs.pop("use_complete_graph", False)
        self.shuffle_train = kwargs.pop("shuffle_train", True)
        self.batch_size = kwargs.pop("batch_size", 8)
        self.kwargs = kwargs

    def setup(self, stage=None):
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_specs = {}
        for i, algorithm in enumerate(self.algorithms):
            # train_dataset = CLRSDataset(algorithm=algorithm, split="train", num_samples=1000, use_complete_graph=self.use_complete_graph)
            # val_dataset = CLRSDataset(algorithm=algorithm, split="val", num_samples=32, use_complete_graph=self.use_complete_graph)
            # test_dataset = CLRSDataset(algorithm=algorithm, split="test", num_samples=32, use_complete_graph=self.use_complete_graph) 
            train_dataset = SALSACLRSDataset(root=self.data_dir, split="train", algorithm=algorithm, num_samples=self.num_samples, graph_generator="er", graph_generator_kwargs={"n": [self.node, self.node], "p_range": (0.1, 0.3)}, hints=True, graph_batch_dir=self.graph_batch_dir)
            val_dataset = SALSACLRSDataset(root=self.data_dir, split="val", algorithm=algorithm, num_samples=512, graph_generator="er", graph_generator_kwargs={"n": [self.node, self.node], "p_range": (0.1, 0.3)}, hints=True)
            test_dataset = SALSACLRSDataset(root=self.data_dir, split="test", algorithm=algorithm, num_samples=512, graph_generator="er", graph_generator_kwargs={"n": [self.node, self.node], "p_range": (0.1, 0.3)}, hints=True)
            collator = CLRSCollater()
            specs = train_dataset.specs

            print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(algorithm, len(train_dataset), len(val_dataset), len(test_dataset)))
            self.task_to_train_datasets[algorithm] = train_dataset
            self.task_to_valid_datasets[algorithm] = val_dataset
            self.task_to_test_datasets[algorithm] = test_dataset
            self.task_to_collators[algorithm] = collator
            self.task_to_specs[algorithm] = specs
        self.multitask_train_dataset = MultitaskDataset(self.task_to_train_datasets)
        self.multitask_valid_dataset = MultitaskDataset(self.task_to_valid_datasets)
        self.multitask_test_dataset  = MultitaskDataset(self.task_to_test_datasets)
        self.multitask_collator = MultitaskCollator(self.task_to_collators)
        self.multitask_train_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_train_datasets.values()])), 
                                                                batch_size=self.batch_size, drop_last=False, task_to_datasets=self.task_to_train_datasets, shuffle=self.shuffle_train)
        self.multitask_valid_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_valid_datasets.values()])), 
                                                                batch_size=self.batch_size, drop_last=False, task_to_datasets=self.task_to_valid_datasets, shuffle=False)
        self.multitask_test_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_test_datasets.values()])), 
                                                                batch_size=self.batch_size, drop_last=False, task_to_datasets=self.task_to_test_datasets, shuffle=False)
    
    def train_dataloader(self):
        return DataLoader(
            self.multitask_train_dataset,
            batch_sampler=self.multitask_train_sampler,
            collate_fn=self.multitask_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.multitask_valid_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.multitask_test_dataset,
            batch_sampler=self.multitask_test_sampler,
            collate_fn=self.multitask_collator,
        )