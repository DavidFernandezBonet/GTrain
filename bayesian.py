import ray
ray.init(ignore_reinit_error=True , local_mode=True)

from models.gps import GPS
from ray import tune
from ray.train import Checkpoint
from ray.tune import get_checkpoint
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from ray.tune.search.bayesopt import BayesOptSearch

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

def short_dirname(trial):
    return "trial_" + str(trial.trial_id)

def data_generation(walk_length = 20):
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_type', default='multihead')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    data_dir = '/mnt/nas/data_nexus/projects/marc_projects/Hard Dataset'

    # Define a pre-transform (e.g. positional encoding)
    transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')

    # Load dataset and split it
    dataset = MyPYGGraphDataset(data_dir, pre_transform=transform, normalize_per_graph=True)
    train_dataset, val_dataset, test_dataset = split_pyg_dataset(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=3)

    print(f"Train graphs: {len(train_dataset)}, Val graphs: {len(val_dataset)}, Test graphs: {len(test_dataset)}")

    num_pos, num_neg = count_labels(train_dataset)
    pos_weight_value = float(num_neg) / float(num_pos + 1e-8)

    return train_loader, val_loader, pos_weight_value



def train_bay(config, params):
    # config includes parameters we wish to tune.
    # params includes fixed parameters we do not wish to tune through Bayesian optimization.
    save_model_folder = '/home/big-computer/Documents/Marc_SP_Constraints/GTrain/Trained Models'
    model_custom_name = 'GPS_model.pth'   # name for the model
    use_node2vec = params["use_node2vec"]
    n_node_attributes = params["n_node_attributes"]
    if use_node2vec:
        node2vec_feats = params["node2vec_feats"]
        node_in_dim = n_node_attributes + node2vec_feats
    else:
        node_in_dim = n_node_attributes

    model = GPS(channels=config["channels"], pe_dim=params["pe_dim"], num_layers=config["num_layers"],
                edge_in_dim=params["edge_in_dim"], attn_type=params["attn_type"], attn_kwargs=config["attn_kwargs"],
                pe_input_dim=params["pe_input_dim"], graph_feat_dim=params["graph_feat_dim"],node_in_dim=node_in_dim,
                use_pe = params['use_pe'], use_pe_lin=params['use_pe_lin'], use_graph_feat_mlp= params['use_graph_feat_mlp'],
                use_edge_encoder=params['use_edge_encoder'], use_node_feat_lin=params['use_node_feat_lin'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_loader, val_loader, pos_weight_value = data_generation(walk_length=params["pe_input_dim"])

    train_losses = []
    val_losses = []
    start_time = time.time()
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    n_epochs = params["n_epochs"]
    data_augmentation = params["data_augmentation"]

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_start_time = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        train_loss = train(model, train_loader, device, optimizer, pos_weight_value, augmentation=data_augmentation)
        val_loss, val_accuracy = test(model, val_loader, device)
        
        scheduler.step(train_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            
        elapsed_time = time.time() - epoch_start_time
        total_elapsed = time.time() - start_time
        remaining_epochs = n_epochs - epoch
        estimated_time_remaining = elapsed_time * remaining_epochs
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Best: {best_val_loss:.4f} | Epoch Time: {elapsed_time:.2f}s | ETA: {estimated_time_remaining:.2f}s | Total: {total_elapsed:.2f}s")

        if epoch % 20 == 0:
            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory(dir='/mnt/nas/data_nexus/projects/marc_projects/tmp') as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                ray.train.report(
                    {"Accuracy": val_accuracy, "val_loss": val_loss, "train_loss": train_loss}, # Maybe include accuracy or ROC curve metrics ?
                    checkpoint=checkpoint,
                )
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_type', default='multihead')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    use_node2vec = True
    data_augmentation = True
    n_epochs = 200
    node2vec_feats = 64
    walk_length = 20

    n_node_attributes = 5
    n_graph_attributes = 5
    n_edge_attributes = 3

    params = {
            'pe_dim': 32,
            'pe_input_dim': walk_length,
            'attn_type': args.attn_type,
            'use_node2vec': use_node2vec,
            'data_augmentation':data_augmentation,
            'n_epochs':n_epochs,
            'node2vec_feats':node2vec_feats,
            'n_node_attributes': n_node_attributes,
            'graph_feat_dim': n_graph_attributes,
            'edge_in_dim': n_edge_attributes,
            'use_pe': True,
            'use_pe_lin': True,
            'use_graph_feat_mlp': True,
            'use_edge_encoder': True,
            'use_node_feat_lin': True
        }
    
    for num_config in range(0,6):
        versions = ['use_pe', 'use_pe_lin', 'use_graph_feat_mlp', 'use_edge_encoder', 'use_node_feat_lin', 'base']

        config = {
            'channels': tune.choice([2**i for i in range(5, 9)]),               # channels
            'num_layers': tune.choice([2,3,4]),                                 # layers
            'attn_kwargs': {'dropout': tune.uniform(0, 0.5)},                   # dropout
            'lr': tune.choice([10**(-x) for x in range(1, 8)]),                 # learning rate
            'weight_decay': tune.choice([10**(-x) for x in range(0, 8)])        # regularisation
            }

        max_num_epochs=200

        scheduler = ASHAScheduler(
            metric="Accuracy",
            mode="max",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2,
        )

        num_samples = 100
    
        result = tune.run(
            partial(train_bay, params=params),
            config=config,
            search_alg="bayesopt",
            num_samples=num_samples,
            scheduler=scheduler,
            trial_dirname_creator=short_dirname,
            storage_path='/mnt/nas/data_nexus/projects/marc_projects/tmp')
        
        best_trial = result.get_best_trial("Accuracy", "max", "last")
        result_string = f"""Best trial config: {best_trial.config}
                            Best trial validation accuracy: {best_trial.last_result['Accuracy']}
                            Best trial final validation loss: {best_trial.last_result['val_loss']}
                            Best trial final training loss: {best_trial.last_result['train_loss']}"""
        print(result_string)
        
        try:
            # Specify the directory where you want to save the files
            # save_dir = 'C:/Users/marcd/OneDrive/Escritorio/KTH/SciLifeLab/Code/GTrain/Results'
            save_dir = '/home/big-computer/Documents/Marc_SP_Constraints/GTrain/Results'
            
            # Create the directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Create full path for the file
            filename = os.path.join(save_dir, f'bayes-results-config{str(num_config)}.txt')
            
            with open(filename, 'w') as f:
                f.write(result_string + '\n')
                f.write(versions[num_config] + str(params['use_pe']) + str(params['use_pe_lin']) + 
                       str(params['use_graph_feat_mlp']) + str(params['use_edge_encoder']) + 
                       str(params['use_node_feat_lin']))
            print(f"Results written to {os.path.abspath(filename)}")
        except Exception as e:
            print(f"Error writing to file: {e}")

        params[versions[num_config]] = True
        if num_config < len(versions)-2:
            params[versions[num_config+1]] = False
        elif num_config == len(versions)-2:
            for i in range(0,len(versions)):
                params[versions[i]] = True

if __name__ == '__main__':
    main()
