# inspection.md

## Overview

The `inspection.py` module provides data validation and quality checks for the CIFAR-10 dataset. It includes functions to verify label integrity, check class distribution, detect corrupted images, and validate image shapes.

## Objectives

- Validate dataset integrity and quality
- Check class distribution balance
- Detect corrupted or invalid images
- Verify label consistency
- Ensure image shape uniformity

## Workflow

1. **Count**: Use `count_images()` to verify sample count
2. **Distribution**: Use `check_distribution()` to analyze class balance
3. **Labels**: Use `verify_labels()` to validate label integrity
4. **Corruption**: Use `detect_corrupted_images()` to find invalid samples
5. **Shapes**: Use `check_image_shapes()` to verify consistency

## Responsibilities

- Data quality validation
- Label integrity verification
- Corruption detection
- Distribution analysis
- Shape consistency checks

## Public API

### `count_images(dataset: Dataset) -> int`

Counts the total number of images in the dataset.

**Parameters:**
- `dataset` (Dataset): PyTorch Dataset instance.

**Returns:**
- `int`: Total number of samples in the dataset.

**Example:**
```python
from data.inspection import count_images
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
count = count_images(train_set)
print(f"Total images: {count}")
```

### `check_distribution(dataset: Dataset) -> dict[str, int]`

Checks the distribution of classes in the dataset.

**Parameters:**
- `dataset` (Dataset): PyTorch Dataset instance with labels.

**Returns:**
- `dict[str, int]`: Dictionary mapping class names to their counts.

**Example:**
```python
from data.inspection import check_distribution
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
dist = check_distribution(train_set)
for class_name, count in dist.items():
    print(f"{class_name}: {count}")
```

### `verify_labels(dataset: Dataset) -> dict[str, Any]`

Verifies label integrity in the dataset.

**Parameters:**
- `dataset` (Dataset): PyTorch Dataset instance.

**Returns:**
- `dict[str, Any]`: Dictionary with verification results including:
  - `num_classes`: Number of unique labels
  - `label_range`: Min and max label values
  - `valid_labels`: Boolean indicating if all labels are valid

**Example:**
```python
from data.inspection import verify_labels
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
label_info = verify_labels(train_set)
print(f"Valid labels: {label_info['valid_labels']}")
print(f"Label range: {label_info['label_range']}")
```

### `detect_corrupted_images(dataset: Dataset, num_samples: int = 100) -> dict[str, Any]`

Detects potentially corrupted images by checking for loading errors.

**Parameters:**
- `dataset` (Dataset): PyTorch Dataset instance.
- `num_samples` (int): Number of samples to check (default: 100).

**Returns:**
- `dict[str, Any]`: Dictionary with inspection results including:
  - `checked`: Number of samples checked
  - `corrupted`: Number of corrupted samples found
  - `corrupted_indices`: List of corrupted sample indices

**Example:**
```python
from data.inspection import detect_corrupted_images
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
result = detect_corrupted_images(train_set, num_samples=1000)
if result['corrupted'] > 0:
    print(f"Found {result['corrupted']} corrupted images")
```

### `check_image_shapes(dataset: Dataset, num_samples: int = 100) -> dict[str, Any]`

Checks the shape consistency of images in the dataset.

**Parameters:**
- `dataset` (Dataset): PyTorch Dataset instance.
- `num_samples` (int): Number of samples to check (default: 100).

**Returns:**
- `dict[str, Any]`: Dictionary with shape information including:
  - `checked`: Number of samples checked
  - `unique_shapes`: Set of unique shapes found
  - `consistent`: Boolean indicating if all shapes are the same

**Example:**
```python
from data.inspection import check_image_shapes
from data.dataset import load_cifar10_dataset

train_set = load_cifar10_dataset(train=True)
shape_info = check_image_shapes(train_set)
print(f"Shapes consistent: {shape_info['consistent']}")
print(f"Unique shapes: {shape_info['unique_shapes']}")
```

## Expected Outputs

- Sample count verification
- Class distribution breakdown
- Label integrity confirmation
- Corruption detection results
- Shape consistency report

## Notes

- Inspection functions are designed to work with any PyTorch Dataset
- For large datasets, use the `num_samples` parameter to limit inspection scope
- Corrupted images are detected by checking for loading errors and invalid tensors
- Shape consistency is important for batch processing

## Integration

This module is used by:
- Data quality validation pipelines
- Pre-flight checks before training
- Dataset debugging and troubleshooting
