# train.py
"""
Main Training Script

Run this script to train or test the GPS model.
"""

import argparse
import os
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from dataset.pyg_dataset import MyPYGGraphDataset
from models.gps import GPS
from utils import (
    split_pyg_dataset,
    train,
    test,
    count_labels,
    evaluate_graphwise,
    plot_losses
)

import torch_geometric.transforms as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_type', default='multihead')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###     CHANGE THIS        ###   #TODO: Change these paths
    data_dir = 'C:/Users/marcd/OneDrive/Escritorio/KTH/SciLifeLab/Code/GTrain/Dataset_Lite/Hard Dataset/'  # directory with .pt files
    save_model_folder = 'C:/Users/marcd/OneDrive/Escritorio/KTH/SciLifeLab/Code/GTrain/Trained Models/'
    ### ---------------------- ###



    model_custom_name = 'GPS_model.pth'   # name for the model
    walk_length = 20
    use_node2vec = True
    node2vec_feats = 64
    data_augmentation = True
    n_epochs = 200
    channels = 64
    layers = 3
    n_edge_attributes = 3
    n_graph_attributes = 5
    n_node_attributes = 5
    attn_kwargs = {'dropout': 0.5}
    train_ratio = 0.6
    val_ratio = 0.20
    test_ratio = 0.20

    if use_node2vec:
        node_in_dim = n_node_attributes + node2vec_feats
    else:
        node_in_dim = n_node_attributes

    # Define a pre-transform (e.g. positional encoding)
    transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')

    if args.mode == "train":
        # Load dataset and split it
        dataset = MyPYGGraphDataset(data_dir, pre_transform=transform, normalize_per_graph=True)
        train_dataset, val_dataset, test_dataset = split_pyg_dataset(
            dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
        val_loader = DataLoader(val_dataset, batch_size=3)
        test_loader = DataLoader(test_dataset, batch_size=4)

        print(f"Train graphs: {len(train_dataset)}, Val graphs: {len(val_dataset)}, Test graphs: {len(test_dataset)}")

        model_config = {
            'channels': channels,
            'pe_dim': 32,
            'num_layers': layers,
            'edge_in_dim': n_edge_attributes,
            'pe_input_dim': walk_length,
            'graph_feat_dim': n_graph_attributes,
            'node_in_dim': node_in_dim,
            'attn_type': args.attn_type,
            'attn_kwargs': attn_kwargs,
        }

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

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5)
        num_pos, num_neg = count_labels(train_dataset)
        pos_weight_value = float(num_neg) / float(num_pos + 1e-8)
        if num_pos < 0.1:
            raise ValueError("Not enough positive examples.")

        train_losses = []
        val_losses = []
        test_losses = []
        start_time = time.time()
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            train_loss = train(model, train_loader, device, optimizer, pos_weight_value, augmentation=data_augmentation)
            val_loss, val_accuracy = test(model, val_loader, device)
            test_loss_val, test_acc = test(model, test_loader, device)
            scheduler.step(train_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict()
            elapsed_time = time.time() - epoch_start_time
            total_elapsed = time.time() - start_time
            remaining_epochs = n_epochs - epoch
            estimated_time_remaining = elapsed_time * remaining_epochs
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Best: {best_val_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss_val:.4f} | Val Accuracy: {val_accuracy:.4f} | Test Accuracy: {test_acc:.4f}| Epoch Time: {elapsed_time:.2f}s | ETA: {estimated_time_remaining:.2f}s | Total: {total_elapsed:.2f}s")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds.")


        os.makedirs(save_model_folder, exist_ok=True)
        model_save_path = os.path.join(save_model_folder, model_custom_name)
        torch.save(model.state_dict(), model_save_path)
        torch.save({
            'model_state_dict': best_model_state,
            'config': model_config,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
        }, os.path.join(save_model_folder, f"best_{model_custom_name}"))
        print(f"Best epoch: {best_epoch} with loss: {best_val_loss:.4f}, and accuracy: {val_accuracy:.4f}, {test_acc:.4f}")
        from utils import evaluate_graphwise
        graphwise_results = evaluate_graphwise(model, val_dataset, device)
        for r in graphwise_results:
            print(r)
        plot_losses(train_losses, val_losses, test_losses,
                    save_path=os.path.join(save_model_folder, f"{model_custom_name}_losses.png"))
    elif args.mode == "test":
        # Example for testing a single graph.
        from utils import test_graph_model
        graph_file_path = '/path/to/graph.pt'
        model_path = '/path/to/best_model.pth'
        results = test_graph_model(graph_file_path, model_path, device, walk_length)
        print(results)


if __name__ == '__main__':
    main()
