# Graph Transformer for Edge Classification in Proximity Graphs

This repository contains code for training a graph transformer using **PyTorch Geometric (PyG)**.
The model is designed for **edge classification** tasks in proximity graphs, with support for dataset handling, custom transforms, ablation studies, and hyperparameter optimization.

Use [`train.py`](train.py) as the main entry point for training or testing the model.

---

## Repository Structure

* **dataset/**
  Dataset loading and preprocessing classes (e.g. `MyPYGGraphDataset`). *(unchanged)*

* **models/**
  Graph model definitions (e.g. the `GPS` model). *(modified, see below)*

* **transforms/**
  Data augmentation and feature masking transforms. *(unchanged)*

* **utils.py**
  General utility functions for training, evaluation, plotting, and logging. *(modified, see below)*

* **train.py**
  Main script for training/testing the model. *(unchanged)*

* **bayesian.py**
  Script for Bayesian optimization of model hyperparameters. *(new)*

---

## Files Created and Modified

### 🟢 Files Created

* **`bayesian.py`**
  Implements a **Bayesian optimization scheme** for hyperparameter tuning using `ray[tune]`.

  * Defines `train_bay()` which takes two parameter sets:

    * `config`: hyperparameters to tune (defined as `tune.choice()` or `tune.uniform()` search spaces).
    * `params`: fixed parameters that remain constant during optimization.
    * The function trains a model given a candidate hyperparameter configuration.
  * The main function runs Bayesian optimization across different **ablated model configurations**, using `functools.partial` with `tune.run()`.

### ✏️ Files Modified

* **`models/gps.py`**

  * Extended the **GPS model architecture** to support **ablation studies** on input features.
  * The model now accepts multiple boolean parameters during instantiation, each controlling whether a specific feature is included.
  * Input dimensions dynamically adjust depending on which features are enabled or disabled.

* **`utils.py`**

  * Updated the `test()` function to return **both accuracy and test loss**, instead of only the loss.

## Requirements
All required packages are listed in [requirements.txt](requirements.txt).

