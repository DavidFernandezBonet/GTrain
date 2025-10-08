# utils.py
"""
Utility Functions

Provides functions for splitting datasets, training, evaluation, and plotting.
"""

import math
import os
import time

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split


def split_pyg_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a PyG dataset into train, validation, and test sets based on unique graph IDs.

    Args:
        dataset: List of PyG Data objects.
        train_ratio (float): Proportion of graphs for training.
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.

    Returns:
        Tuple (train_dataset, val_dataset, test_dataset).
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1."
    graph_id_to_data = {}
    for data in dataset:
        graph_attr = getattr(data, "graph_attr", {})
        graph_id = graph_attr.get("graph_id", None)
        if graph_id:
            graph_id_to_data.setdefault(graph_id, []).append(data)
    unique_graph_ids = list(graph_id_to_data.keys())
    num_graphs = len(unique_graph_ids)
    if num_graphs < 3:
        raise ValueError("Not enough graphs for splitting!")
    num_train = int(num_graphs * train_ratio)
    num_temp = num_graphs - num_train
    num_val = max(1, int(num_graphs * val_ratio))
    num_test = num_temp - num_val
    train_ids, temp_ids = train_test_split(unique_graph_ids, train_size=num_train, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, train_size=num_val, test_size=num_test, random_state=42)
    train_dataset = [data for gid in train_ids for data in graph_id_to_data[gid]]
    val_dataset = [data for gid in val_ids for data in graph_id_to_data[gid]]
    test_dataset = [data for gid in test_ids for data in graph_id_to_data[gid]]
    return train_dataset, val_dataset, test_dataset


def train(model, loader, device, optimizer, pos_weight_value, augmentation=False):
    """
    Trains the model for one epoch.

    Args:
        model: The GPS model.
        loader: DataLoader for training data.
        device: Device to run on.
        optimizer: Optimizer.
        pos_weight_value: Weight for the positive class.
        augmentation (bool): Whether to apply feature masking.

    Returns:
        Average loss per edge.
    """
    model.train()
    total_loss = 0
    total_edges = 0
    for data in loader:
        data = data.to(device)
        if augmentation:
            # Import masking transforms locally to avoid circular imports.
            from transforms.feature_masking import NodeFeatureMasking, EdgeFeatureMasking
            data = NodeFeatureMasking(p=0.15)(data)
            data = EdgeFeatureMasking(p=0.15)(data)
        optimizer.zero_grad()
        out = model(data)
        if hasattr(data, 'edge_label'):
            loss = F.binary_cross_entropy_with_logits(
                out, data.edge_label, pos_weight=torch.tensor([pos_weight_value], device=device)
            )
        elif hasattr(data, 'y'):
            loss = F.binary_cross_entropy_with_logits(
                out, data.y.to(torch.float), pos_weight=torch.tensor([pos_weight_value], device=device)
            )
        else:
            raise ValueError("No edge_label or y in data")
        loss.backward()
        optimizer.step()
        num_edges = data.edge_index.size(1)
        total_loss += loss.item() * num_edges
        total_edges += num_edges
    return total_loss / total_edges


@torch.no_grad()
def test(model, loader, device):
    """
    Evaluates the model on a dataset.

    Args:
        model: The GPS model.
        loader: DataLoader for evaluation.
        device: Device to run on.

    Returns:
        Tuple of (average loss per edge, accuracy)
    """
    model.eval()
    total_loss = 0
    total_edges = 0
    total_correct = 0
    total_predictions = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        
        # Get the true labels
        if hasattr(data, 'edge_label'):
            labels = data.edge_label
        elif hasattr(data, 'y'):
            labels = data.y.to(torch.float)
        else:
            raise ValueError("No edge_label or y in data")
            
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(out, labels)
        
        # Compute accuracy
        predictions = (torch.sigmoid(out) > 0.5).float()  # Convert probabilities to binary predictions
        correct = (predictions == labels).sum().item()
        
        num_edges = data.edge_index.size(1)
        total_loss += loss.item() * num_edges
        total_edges += num_edges
        total_correct += correct
        total_predictions += labels.numel()
    
    avg_loss = total_loss / total_edges
    accuracy = total_correct / total_predictions
    
    return avg_loss, accuracy


def evaluate_graphwise(model, dataset, device):
    """
    Evaluates the model on each graph individually and computes performance metrics.

    Args:
        model: The GPS model.
        dataset: List of PyG Data objects.
        device: Device to run on.

    Returns:
        List of dictionaries with evaluation metrics per graph.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy()
            if hasattr(data, 'edge_label'):
                labels = data.edge_label.cpu().numpy()
            elif hasattr(data, 'y'):
                labels = data.y.cpu().numpy()
            else:
                raise ValueError("No edge_label or y in data")
            roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else None
            pr_auc = average_precision_score(labels, probs)
            precision, recall, thresholds = precision_recall_curve(labels, probs)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1 = f1_scores[best_idx]
            pred_labels = (probs >= best_thr).astype(int)
            cm = confusion_matrix(labels, pred_labels)
            results.append({
                'graph_index': i,
                'best_threshold': best_thr,
                'best_f1': best_f1,
                'confusion_matrix': cm,
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            })
    return results


def plot_losses(train_losses, val_losses, test_losses, save_path):
    """
    Plots and saves training, validation, and test losses over epochs.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        test_losses: List of test losses.
        save_path: File path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.savefig(save_path)


def count_labels(dataset):
    """
    Counts positive and negative labels in a dataset.

    Args:
        dataset: List of PyG Data objects.

    Returns:
        Tuple (num_positive, num_negative).
    """
    num_pos, num_neg = 0, 0
    for data in dataset:
        if hasattr(data, 'y') and data.y is not None:
            num_pos += (data.y == 1).sum().item()
            num_neg += (data.y == 0).sum().item()
    return num_pos, num_neg


def test_graph_model(graph_file_path, model_path, device=None, walk_length=20):
    """
    Tests a single graph using a saved model checkpoint.

    Args:
        graph_file_path (str): Path to the .pt file of the graph.
        model_path (str): Path to the saved model checkpoint.
        device: Device to run on.
        walk_length (int): Walk length for positional encoding.

    Returns:
        Dict of evaluation metrics.
    """
    import torch_geometric.transforms as T
    loaded_package = torch.load(graph_file_path, map_location=device)
    if not isinstance(loaded_package, dict) or "data" not in loaded_package:
        raise ValueError("Invalid file format. Expected a dict with 'data'.")
    graph_data = loaded_package["data"]
    pe_transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
    graph_data = pe_transform(graph_data)

    def _normalize_tensor(tensor):
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True) + 1e-8
        return (tensor - mean) / std

    if graph_data.x is not None:
        graph_data.x = _normalize_tensor(graph_data.x)
    if graph_data.edge_attr is not None:
        graph_data.edge_attr = _normalize_tensor(graph_data.edge_attr)
    graph_data = graph_data.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    from models.gps import GPS  # import model from our models module
    model = GPS(
        channels=model_config['channels'],
        pe_dim=model_config['pe_dim'],
        num_layers=model_config['num_layers'],
        edge_in_dim=model_config['edge_in_dim'],
        pe_input_dim=model_config['pe_input_dim'],
        graph_feat_dim=model_config['graph_feat_dim'],
        node_in_dim=model_config['node_in_dim'],
        attn_type=model_config['attn_type'],
        attn_kwargs=model_config['attn_kwargs']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        logits = model(graph_data)
        probs = torch.sigmoid(logits).cpu().numpy()

    if hasattr(graph_data, 'edge_label'):
        labels = graph_data.edge_label.cpu().numpy()
    elif hasattr(graph_data, 'y'):
        labels = graph_data.y.cpu().numpy()
    else:
        raise ValueError("Graph data has no 'edge_label' or 'y'.")
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else None
    pr_auc = average_precision_score(labels, probs)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    pred_labels = (probs >= best_thr).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    results = {
        'best_threshold': best_thr,
        'best_f1': best_f1,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    return results
