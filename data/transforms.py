"""
transforms.py — Transform pipelines for CIFAR-10 preprocessing and augmentation.

Usage:
    from data.transforms import get_train_transform, get_eval_transform
    from data.statistics import load_statistics
    
    train_transform = get_train_transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    eval_transform = get_eval_transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
"""

from __future__ import annotations

from typing import Any

import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ImageNet statistics for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR-10 statistics (computed from dataset)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_train_transform(
    mean: list[float] | None = None,
    std: list[float] | None = None,
    resize_size: int = 224,
    crop_size: int = 224,
    augmentation: bool = True,
) -> transforms.Compose:
    """Get transform pipeline for training data.

    Args:
        mean: Normalization mean values. Defaults to ImageNet statistics.
        std: Normalization std values. Defaults to ImageNet statistics.
        resize_size: Size to resize images to.
        crop_size: Size for random crop.
        augmentation: If True, applies data augmentation.

    Returns:
        Composed transform pipeline.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize(resize_size))
    
    if augmentation:
        # Data augmentation
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomCrop(crop_size, padding=16))
    
    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_eval_transform(
    mean: list[float] | None = None,
    std: list[float] | None = None,
    resize_size: int = 224,
) -> transforms.Compose:
    """Get transform pipeline for validation/test data.

    Args:
        mean: Normalization mean values. Defaults to ImageNet statistics.
        std: Normalization std values. Defaults to ImageNet statistics.
        resize_size: Size to resize images to.

    Returns:
        Composed transform pipeline.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    transform_list = [
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    return transforms.Compose(transform_list)


def get_cifar10_transforms(
    use_cifar10_stats: bool = False,
    augmentation: bool = True,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Get standard CIFAR-10 transforms.

    Args:
        use_cifar10_stats: If True, uses CIFAR-10 statistics.
                           If False, uses ImageNet statistics.
        augmentation: If True, applies augmentation to training transform.

    Returns:
        Tuple of (train_transform, eval_transform).
    """
    mean = CIFAR10_MEAN if use_cifar10_stats else IMAGENET_MEAN
    std = CIFAR10_STD if use_cifar10_stats else IMAGENET_STD
    
    train_transform = get_train_transform(
        mean=mean, std=std, augmentation=augmentation
    )
    eval_transform = get_eval_transform(mean=mean, std=std)
    
    return train_transform, eval_transform


class CustomTransform:
    """Base class for custom transforms."""
    
    def __call__(self, img: Any) -> Any:
        """Apply transform to image."""
        raise NotImplementedError("Subclasses must implement __call__")


def get_transform_config(
    config_name: str = "imagenet_standard",
) -> dict[str, Any]:
    """Get predefined transform configurations.

    Args:
        config_name: Name of the configuration.
                    Options: "imagenet_standard", "cifar10_standard",
                             "cifar10_augmented", "imagenet_augmented".

    Returns:
        Dictionary with transform configuration parameters.

    Raises:
        ValueError: If config_name is not recognized.
    """
    configs = {
        "imagenet_standard": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "resize_size": 224,
            "augmentation": False,
        },
        "imagenet_augmented": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "resize_size": 224,
            "augmentation": True,
        },
        "cifar10_standard": {
            "mean": CIFAR10_MEAN,
            "std": CIFAR10_STD,
            "resize_size": 32,
            "augmentation": False,
        },
        "cifar10_augmented": {
            "mean": CIFAR10_MEAN,
            "std": CIFAR10_STD,
            "resize_size": 32,
            "augmentation": True,
        },
    }
    
    if config_name not in configs:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[config_name]
