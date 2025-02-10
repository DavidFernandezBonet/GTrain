# dataset/pyg_dataset.py
"""
PyG Dataset Module

Provides the MyPYGGraphDataset class to load graphs from .pt files.
"""

import os
import torch
from torch_geometric.data import InMemoryDataset, Data


class MyPYGGraphDataset(InMemoryDataset):
    """
    Loads a dataset of PyG objects from .pt files in a directory.

    Args:
        data_dir (str): Directory containing .pt files.
        root (str): Root directory (defaults to data_dir).
        transform: Optional transform.
        pre_transform: Optional pre-transform.
        normalize_per_graph (bool): Whether to normalize features per graph.
        force_reprocess (bool): Force reprocessing even if cached data exists.
    """
    def __init__(self, data_dir, root=None, transform=None, pre_transform=None,
                 normalize_per_graph=False, force_reprocess=True):
        self.data_dir = data_dir
        self.root = root if root else data_dir
        self.normalize_per_graph = normalize_per_graph
        self.force_reprocess = force_reprocess
        super().__init__(self.root, transform, pre_transform)
        if self.force_reprocess or not os.path.exists(self.processed_paths[0]):
            self.data, self.slices = self.process()
            torch.save((self.data, self.slices), self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        self.metadata = None
        for file_path in all_files:
            loaded_package = torch.load(file_path)
            if isinstance(loaded_package, dict):
                data = loaded_package.get("data", None)
                metadata = loaded_package.get("metadata", None)
            else:
                data = loaded_package
                metadata = None

            if data is None:
                continue

            self.metadata = metadata
            # If metadata contains graph attributes, attach them to the data object.
            if metadata and "graph_attr" in metadata:
                for key, value in metadata["graph_attr"].items():
                    setattr(data, key, torch.tensor(value) if isinstance(value, (int, float)) else value)

            if self.normalize_per_graph:
                data = self._normalize_per_graph(data)
            if self.pre_transform:
                data = self.pre_transform(data)
            data_list.append(data)
        data, slices = self.collate(data_list)
        return data, slices

    def _normalize_per_graph(self, data):
        """Normalizes node and edge attributes per graph."""
        if data.x is not None:
            data.x = self._normalize_tensor(data.x)
        if data.edge_attr is not None:
            data.edge_attr = self._normalize_tensor(data.edge_attr)
        return data

    def _normalize_tensor(self, tensor):
        """Normalizes a tensor using its mean and std."""
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True) + 1e-8
        return (tensor - mean) / std
