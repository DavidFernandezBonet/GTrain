
# Graph Transformer for edge classification in proximity graphs 

<img width="700" alt="gps_architecture" src="https://github.com/user-attachments/assets/01de8e33-4cd2-4511-bfea-9c500303bbdc" />

This repository contains code for training a graph transformer using PyTorch Geometric (PyG) objects. The code is organized into modules for datasets, models, transforms, and utilities. Use `train.py` as the main entry point for training or testing the model.

## Repository Structure

- **dataset/**: Contains dataset loading classes (e.g. `MyPYGGraphDataset`).
- **models/**: Contains model definitions (e.g. the `GPS` model).
- **transforms/**: Contains data augmentation / feature masking transforms.
- **utils.py**: Utility functions for training, evaluation, plotting, etc.
- **train.py**: Main script for training/testing the model.

## Requirements

See [requirements.txt](requirements.txt) for the required packages.
