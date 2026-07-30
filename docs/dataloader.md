# dataloader.md

## Overview

The `dataloader.py` module provides DataLoader creation for CIFAR-10 with persistent train/val/test split. It ensures consistent data splitting across all phases of the project by persisting the split indices to disk.

## Objectives

- Create DataLoaders for train, validation, and test sets
- Maintain consistent train/val/test split across all runs
- Support configurable batch processing options
- Enable efficient data loading with multiprocessing
- Provide flexible transform configuration

## Workflow

1. **Split**: Generate or load persistent train/val/test split
2. **Transform**: Apply appropriate transforms to each split
3. **Load**: Create DataLoaders with configured parameters
4. **Batch**: Iterate through batches during training/evaluation

## Responsibilities

- Split persistence and management
- DataLoader creation and configuration
- Transform application
- Batch processing optimization

## Public API

### `get_cifar10_loaders(batch_size: int = 64, num_workers: int = 2, train_transform: object | None = None, eval_transform: object | None = None, pin_memory: bool = True, persistent_workers: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]`

Returns (train_loader, val_loader, test_loader).

The split is loaded from data/processed/cifar10_split_seed42.json. If the file does not exist yet it is generated once and persisted. Every subsequent call (in any phase) reuses the persisted file.

**Parameters:**
- `batch_size` (int): Batch size for DataLoaders (default: 64).
- `num_workers` (int): Number of worker processes for data loading (default: 2).
- `train_transform` (object | None): Transform to apply to training data.
- `eval_transform` (object | None): Transform to apply to validation/test data.
- `pin_memory` (bool): If True, uses pinned memory for faster GPU transfer (default: True).
- `persistent_workers` (bool): If True, keeps workers alive between epochs (default: False).

**Returns:**
- `tuple[DataLoader, DataLoader, DataLoader]`: Tuple of (train_loader, val_loader, test_loader).

**Example:**
```python
from data.dataloader import get_cifar10_loaders
from data.transforms import get_cifar10_transforms

train_transform, eval_transform = get_cifar10_transforms()
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=64,
    train_transform=train_transform,
    eval_transform=eval_transform
)
```

### `get_single_loader(dataset: torch.utils.data.Dataset, batch_size: int = 64, shuffle: bool = False, num_workers: int = 2, pin_memory: bool = True) -> DataLoader`

Creates a single DataLoader for a given dataset.

**Parameters:**
- `dataset` (torch.utils.data.Dataset): PyTorch Dataset instance.
- `batch_size` (int): Batch size for the DataLoader (default: 64).
- `shuffle` (bool): If True, shuffles the data (default: False).
- `num_workers` (int): Number of worker processes for data loading (default: 2).
- `pin_memory` (bool): If True, uses pinned memory for faster GPU transfer (default: True).

**Returns:**
- `DataLoader`: DataLoader instance.

**Example:**
```python
from data.dataloader import get_single_loader
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
train_loader = get_single_loader(train_set, batch_size=64, shuffle=True)
```

## Split Persistence

The train/val/test split is generated **once** and persisted to disk:

- **Split file**: `data/processed/cifar10_split_seed42.json`
- **Seed**: 42 (fixed for reproducibility)
- **Train ratio**: 0.9 (45k train / 5k val out of 50k)
- **Test set**: Full official 10k test samples

The split file is treated as **read-only** once created. No script may regenerate or mutate it. This ensures:
- Consistent evaluation across all experiments
- Reproducible results
- Fair comparison between different models

## DataLoader Configuration

### Batch Size
- Default: 64
- Affects memory usage and training speed
- Larger batches may require more GPU memory

### Num Workers
- Default: 2
- Number of subprocesses for data loading
- Higher values can speed up data loading but use more CPU

### Pin Memory
- Default: True
- Uses pinned (page-locked) memory
- Faster transfer from CPU to GPU
- Recommended for GPU training

### Persistent Workers
- Default: False
- Keeps worker processes alive between epochs
- Reduces startup overhead for each epoch
- Useful for long training runs

## Expected Outputs

- Three DataLoaders: train, validation, and test
- Persistent split file for reproducibility
- Efficient batch processing
- Configurable data loading options

## Notes

- The split is generated on the first call and persisted for all subsequent calls
- Training data is shuffled, validation and test data are not
- Default transforms are used if custom transforms are not provided
- The split file should be manually deleted if a new split is needed
- For CIFAR-10: 45k train, 5k validation, 10k test samples

## Integration

This module is used by:
- Training scripts for batch iteration
- Evaluation scripts for test set processing
- Model training pipelines
