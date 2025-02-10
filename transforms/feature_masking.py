# transforms/feature_masking.py
"""
Feature Masking Transforms

Defines transforms for masking node and edge features as well as helper functions.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec


def preserve_data_attributes(original_data, modified_data):
    """
    Copies any missing attributes from the original Data object to the modified one.
    """
    for key, value in original_data.items():
        if key not in modified_data:
            setattr(modified_data, key, value)
    return modified_data


def add_node2vec_embeddings(
        data,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_epochs=2,
        lr=0.01
):
    """
    Adds Node2Vec embeddings to a PyG Data object.

    Args:
        data: PyG Data object.
        embedding_dim (int): Size of node embeddings.
        walk_length (int): Length of random walks.
        context_size (int): Context size.
        walks_per_node (int): Number of walks per node.
        num_epochs (int): Training epochs.
        lr (float): Learning rate.

    Returns:
        Data: Modified Data object with updated node features.
    """
    node2vec = Node2Vec(
        data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_nodes=data.num_nodes,
        sparse=True
    )
    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
    emb = node2vec.embedding.weight.detach()
    if data.x is not None:
        data.x = torch.cat([data.x, emb], dim=-1)
    else:
        data.x = emb
    return data


class NodeFeatureMasking(nn.Module):
    """
    Randomly masks node features with a given probability.
    """

    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x.clone()
        mask = torch.rand(x.size(1)) < self.p
        x[:, mask] = 0
        new_data = Data(x=x, edge_index=data.edge_index)
        return preserve_data_attributes(data, new_data)


class EdgeFeatureMasking(nn.Module):
    """
    Randomly masks edge features with a given probability.
    """

    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        edge_attr = data.edge_attr.clone() if data.edge_attr is not None else None
        if edge_attr is not None:
            mask = torch.rand(edge_attr.size(1)) < self.p
            edge_attr[:, mask] = 0
        new_data = Data(edge_index=data.edge_index, edge_attr=edge_attr)
        return preserve_data_attributes(data, new_data)
