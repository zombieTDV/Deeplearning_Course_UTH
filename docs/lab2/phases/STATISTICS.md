# STATISTICS.md

## Overview

The `statistics.py` module computes and manages dataset statistics for normalization. It provides functions to calculate mean and standard deviation from datasets, save/load statistics to/from JSON files, and retrieve normalization parameters for transform pipelines.

## Objectives

- Compute dataset mean and standard deviation for normalization
- Persist statistics to disk for reuse
- Load saved statistics for transform configuration
- Support GPU-accelerated computation for large datasets
- Provide flexible normalization parameter retrieval

## Workflow

1. **Compute**: Use `compute_dataset_statistics()` to calculate mean/std
2. **Save**: Use `save_statistics()` to persist to JSON
3. **Load**: Use `load_statistics()` to retrieve saved statistics
4. **Apply**: Use `get_normalization_transform()` to get parameters for transforms

## Responsibilities

- Dataset statistics computation
- Statistics persistence and retrieval
- Normalization parameter management
- GPU support for efficient computation

## Public API

### `compute_dataset_statistics(dataset: Dataset, batch_size: int = 1000, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]`

Computes mean and standard deviation of dataset for normalization.

**Parameters:**
- `dataset` (Dataset): PyTorch Dataset instance with image tensors.
- `batch_size` (int): Batch size for processing (default: 1000).
- `device` (torch.device | None): Device to use for computation. If None, uses CPU.

**Returns:**
- `tuple[torch.Tensor, torch.Tensor]`: Tuple of (mean, std) as tensors with shape (3,).

**Example:**
```python
from data.statistics import compute_dataset_statistics
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
mean, std = compute_dataset_statistics(train_set, batch_size=1000)
print(f"Mean: {mean}")
print(f"Std: {std}")
```

### `save_statistics(mean: torch.Tensor | list[float], std: torch.Tensor | list[float], filepath: str = DEFAULT_STATS_FILE) -> None`

Saves dataset statistics to JSON file.

**Parameters:**
- `mean` (torch.Tensor | list[float]): Mean values as tensor or list.
- `std` (torch.Tensor | list[float]): Standard deviation values as tensor or list.
- `filepath` (str): Path to save the statistics file.

**Example:**
```python
from data.statistics import save_statistics

save_statistics(mean, std, "data/processed/dataset_statistics.json")
```

### `load_statistics(filepath: str = DEFAULT_STATS_FILE) -> dict[str, Any]`

Loads dataset statistics from JSON file.

**Parameters:**
- `filepath` (str): Path to the statistics file.

**Returns:**
- `dict[str, Any]`: Dictionary with 'mean' and 'std' keys.

**Raises:**
- `FileNotFoundError`: If the statistics file does not exist.

**Example:**
```python
from data.statistics import load_statistics

stats = load_statistics("data/processed/dataset_statistics.json")
mean = stats['mean']
std = stats['std']
```

### `get_normalization_transform(mean: torch.Tensor | list[float] | None = None, std: torch.Tensor | list[float] | None = None, stats_file: str | None = None) -> tuple[list[float], list[float]]`

Gets normalization parameters for transforms.

**Parameters:**
- `mean` (torch.Tensor | list[float] | None): Custom mean values. If None, loads from stats_file.
- `std` (torch.Tensor | list[float] | None): Custom std values. If None, loads from stats_file.
- `stats_file` (str | None): Path to statistics file if mean/std not provided.

**Returns:**
- `tuple[list[float], list[float]]`: Tuple of (mean, std) as lists.

**Raises:**
- `ValueError`: If neither custom values nor stats_file are provided.

**Example:**
```python
from data.statistics import get_normalization_transform

# Load from file
mean, std = get_normalization_transform(stats_file="data/processed/dataset_statistics.json")

# Use custom values
mean, std = get_normalization_transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

## Expected Outputs

- Computed mean and standard deviation tensors
- JSON file with saved statistics
- Loaded statistics dictionary
- Normalization parameters for transforms

## Notes

- Statistics computation is performed in a single pass for efficiency
- GPU acceleration can significantly speed up computation for large datasets
- Saved statistics can be reused across different runs to ensure consistency
- The default statistics file is `data/processed/dataset_statistics.json`
- Computed statistics are channel-wise (3 values for RGB)

## Integration

This module is used by:
- `data/transforms.py` for normalization parameters
- Data preprocessing pipelines
- Model training configuration
