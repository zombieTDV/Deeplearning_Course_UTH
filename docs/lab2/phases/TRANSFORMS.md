# TRANSFORMS.md

## Overview

The `transforms.py` module provides transform pipelines for CIFAR-10 preprocessing and data augmentation. It includes predefined configurations for both ImageNet and CIFAR-10 statistics, with support for training augmentation and evaluation preprocessing.

## Objectives

- Provide configurable transform pipelines for training and evaluation
- Support data augmentation for improved generalization
- Include predefined normalization statistics (ImageNet and CIFAR-10)
- Enable flexible transform configuration
- Support both pretrained model and custom training scenarios

## Workflow

1. **Select**: Choose a transform configuration or create custom parameters
2. **Configure**: Set mean, std, resize size, and augmentation options
3. **Apply**: Use transforms with dataset loading

## Responsibilities

- Transform pipeline creation
- Data augmentation configuration
- Normalization parameter management
- Predefined transform configurations

## Public API

### `get_train_transform(mean: list[float] | None = None, std: list[float] | None = None, resize_size: int = 224, crop_size: int = 224, augmentation: bool = True) -> transforms.Compose`

Gets transform pipeline for training data.

**Parameters:**
- `mean` (list[float] | None): Normalization mean values. Defaults to ImageNet statistics.
- `std` (list[float] | None): Normalization std values. Defaults to ImageNet statistics.
- `resize_size` (int): Size to resize images to (default: 224).
- `crop_size` (int): Size for random crop (default: 224).
- `augmentation` (bool): If True, applies data augmentation (default: True).

**Returns:**
- `transforms.Compose`: Composed transform pipeline.

**Example:**
```python
from data.transforms import get_train_transform

train_transform = get_train_transform(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    augmentation=True
)
```

### `get_eval_transform(mean: list[float] | None = None, std: list[float] | None = None, resize_size: int = 224) -> transforms.Compose`

Gets transform pipeline for validation/test data.

**Parameters:**
- `mean` (list[float] | None): Normalization mean values. Defaults to ImageNet statistics.
- `std` (list[float] | None): Normalization std values. Defaults to ImageNet statistics.
- `resize_size` (int): Size to resize images to (default: 224).

**Returns:**
- `transforms.Compose`: Composed transform pipeline.

**Example:**
```python
from data.transforms import get_eval_transform

eval_transform = get_eval_transform(resize_size=224)
```

### `get_cifar10_transforms(use_cifar10_stats: bool = False, augmentation: bool = True) -> tuple[transforms.Compose, transforms.Compose]`

Gets standard CIFAR-10 transforms.

**Parameters:**
- `use_cifar10_stats` (bool): If True, uses CIFAR-10 statistics. If False, uses ImageNet statistics.
- `augmentation` (bool): If True, applies augmentation to training transform.

**Returns:**
- `tuple[transforms.Compose, transforms.Compose]`: Tuple of (train_transform, eval_transform).

**Example:**
```python
from data.transforms import get_cifar10_transforms

# Use ImageNet stats with augmentation
train_transform, eval_transform = get_cifar10_transforms(
    use_cifar10_stats=False,
    augmentation=True
)
```

### `get_transform_config(config_name: str = "imagenet_standard") -> dict[str, Any]`

Gets predefined transform configurations.

**Parameters:**
- `config_name` (str): Name of the configuration.
  Options: "imagenet_standard", "cifar10_standard", "cifar10_augmented", "imagenet_augmented".

**Returns:**
- `dict[str, Any]`: Dictionary with transform configuration parameters.

**Raises:**
- `ValueError`: If config_name is not recognized.

**Example:**
```python
from data.transforms import get_transform_config

config = get_transform_config("imagenet_augmented")
print(config)
```

## Predefined Configurations

### ImageNet Statistics
- **Mean**: [0.485, 0.456, 0.406]
- **Std**: [0.229, 0.224, 0.225]
- **Use case**: Pretrained models (ResNet, DenseNet, etc.)

### CIFAR-10 Statistics
- **Mean**: [0.4914, 0.4822, 0.4465]
- **Std**: [0.2470, 0.2435, 0.2616]
- **Use case**: Training from scratch on CIFAR-10

### Available Configurations
1. **imagenet_standard**: ImageNet stats, no augmentation, 224px resize
2. **imagenet_augmented**: ImageNet stats, with augmentation, 224px resize
3. **cifar10_standard**: CIFAR-10 stats, no augmentation, 32px resize
4. **cifar10_augmented**: CIFAR-10 stats, with augmentation, 32px resize

## Expected Outputs

- Composed transform pipelines ready for dataset application
- Configurable augmentation options
- Properly normalized image tensors

## Notes

- Training transforms include data augmentation (random flip, random crop)
- Evaluation transforms are deterministic (no augmentation)
- Normalization is applied after conversion to tensor
- Resize size should match the expected input size of the model
- For pretrained models, use ImageNet statistics
- For training from scratch, compute and use dataset-specific statistics

## Integration

This module is used by:
- `data/dataloader.py` for transform configuration
- Dataset loading pipelines
- Model training and evaluation scripts
