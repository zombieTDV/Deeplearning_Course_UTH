# DATASET.md

## Overview

The `dataset.py` module provides functions for downloading, loading, and verifying the CIFAR-10 dataset. It serves as the foundation for the data pipeline, handling the initial data acquisition and basic dataset information retrieval.

## Objectives

- Download CIFAR-10 dataset to the project's data directory
- Load CIFAR-10 training and test sets
- Provide dataset information and metadata
- Ensure data integrity through basic verification

## Workflow

1. **Download**: Use `download_cifar10()` to fetch the dataset
2. **Load**: Use `load_cifar10_dataset()` to access the data
3. **Inspect**: Use `get_dataset_info()` to retrieve metadata

## Responsibilities

- Dataset download and caching
- Dataset loading with optional transforms
- Dataset metadata extraction
- Path management for data storage

## Public API

### `download_cifar10(root: str = DEFAULT_DATA_ROOT) -> None`

Downloads CIFAR-10 dataset to the specified directory.

**Parameters:**
- `root` (str): Directory where the dataset will be stored. Defaults to `data/external/CIFAR-10`.

**Returns:**
- None

**Raises:**
- `RuntimeError`: If download fails.

**Example:**
```python
from data.dataset import download_cifar10

download_cifar10(root="data/external/CIFAR-10")
```

### `load_cifar10_dataset(root: str = DEFAULT_DATA_ROOT, train: bool = True, transform: object | None = None) -> Dataset`

Loads CIFAR-10 dataset.

**Parameters:**
- `root` (str): Directory where the dataset is stored.
- `train` (bool): If True, loads training set. If False, loads test set.
- `transform` (object | None): Optional transform to be applied on a sample.

**Returns:**
- `Dataset`: CIFAR10 dataset instance.

**Raises:**
- `RuntimeError`: If dataset is not found and download is not requested.

**Example:**
```python
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
test_set = load_cifar10_dataset(train=False)
```

### `get_dataset_info(dataset: Dataset) -> dict`

Gets information about the dataset.

**Parameters:**
- `dataset` (Dataset): PyTorch Dataset instance.

**Returns:**
- `dict`: Dictionary with dataset information including:
  - `name`: Dataset name
  - `num_samples`: Number of samples
  - `num_classes`: Number of classes
  - `classes`: List of class names

**Example:**
```python
from data.dataset import load_cifar10_dataset, get_dataset_info

train_set = load_cifar10_dataset(train=True)
info = get_dataset_info(train_set)
print(f"Classes: {info['classes']}")
print(f"Samples: {info['num_samples']}")
```

## Expected Outputs

- Downloaded CIFAR-10 dataset in `data/external/CIFAR-10/`
- Loaded dataset objects ready for transformation and batching
- Dataset metadata including class names and sample counts

## Notes

- The dataset is downloaded only once and cached locally
- CIFAR-10 contains 50,000 training images and 10,000 test images
- Each image is 32x32 RGB with 10 classes
- The default data root is `data/external/CIFAR-10`

## Integration

This module is used by:
- `data/inspection.py` for data validation
- `data/statistics.py` for computing normalization statistics
- `data/dataloader.py` for creating data loaders
