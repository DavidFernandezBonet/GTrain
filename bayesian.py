from models.gps import GPS
from skopt import BayesSearchCV

import numpy as np
import torch
import argparse
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from dataset.pyg_dataset import MyPYGGraphDataset
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

    data_dir = 'C:/Users/marcd/OneDrive/Escritorio/KTH/SciLifeLab/Code/GTrain/Dataset_Lite'  # directory with .pt files
    save_model_folder = 'C:/Users/marcd/OneDrive/Escritorio/KTH/SciLifeLab/Code/GTrain/Trained Models/'

    model_custom_name = 'GPS_model.pth'   # name for the model
    train_ratio = 0.9
    val_ratio = 0.05
    test_ratio = 0.05
    use_node2vec = True
    data_augmentation = True
    n_epochs = 200

    walk_length = 20
    node2vec_feats = 64
    channels = 64
    layers = 3
    n_edge_attributes = 3
    n_graph_attributes = 5
    n_node_attributes = 3
    attn_kwargs = {'dropout': 0.5}

    if use_node2vec:
        node_in_dim = n_node_attributes + node2vec_feats
    else:
        node_in_dim = n_node_attributes

    # Define a pre-transform (e.g. positional encoding)
    transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')

    # Load dataset and split it
    dataset = MyPYGGraphDataset(data_dir, pre_transform=transform, normalize_per_graph=True)
    train_dataset, val_dataset, test_dataset = split_pyg_dataset(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    print(train_dataset)
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    # val_loader = DataLoader(val_dataset, batch_size=3)
    # test_loader = DataLoader(test_dataset, batch_size=4)

    print(f"Train graphs: {len(train_dataset)}, Val graphs: {len(val_dataset)}, Test graphs: {len(test_dataset)}")

    model_config = {
        'pe_dim': 32,
        'attn_type': args.attn_type,
        'attn_kwargs': attn_kwargs,
    }

    param_space = {
        'channels': (0.01, 1.0, 'log-uniform'),
        'num_layers': layers,
        'edge_in_dim': n_edge_attributes,
        'pe_input_dim': walk_length,
        'graph_feat_dim': n_graph_attributes,
        'node_in_dim': node_in_dim,
    }

    optimizer = BayesSearchCV(
        estimator=GPS(pe_dim=model_config["pe_dim"], attn_type=model_config["attn_type"], attn_kwargs=model_config['attn_kwargs']),
        search_spaces=param_space,
        scoring='accuracy', # might need to change it to put a function that trains the model and outputs accuracy
        cv=3,
        n_iter=50,
    )

    #Fit the model
    # optimizer.fit(train_dataset, y_train)

    # After fitting, you can get the best parameters and score as follows:
    # best_params = opt.best_params_
    # best_score = opt.best_score_
    # print(f"Best parameters: {best_params}")
    # print(f"Best accuracy score: {best_score}")

if __name__ == '__main__':
    main()
