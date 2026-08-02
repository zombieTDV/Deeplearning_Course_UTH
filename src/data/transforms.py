"""
transforms.py — Transform pipelines for CIFAR-10 preprocessing and augmentation.

Usage:
    from data.transforms import get_train_transform, get_eval_transform
    from data.statistics import load_statistics
    
    train_transform = get_train_transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    eval_transform = get_eval_transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
"""

from __future__ import annotations

import logging
from typing import Any

import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


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
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
    random_horizontal_flip: bool = True,
    random_crop: bool = True,
    random_crop_padding: int = 16,
    random_rotation: int = 0,
    color_jitter: dict[str, float] | None = None,
) -> transforms.Compose:
    """Get transform pipeline for training data.

    Args:
        mean: Normalization mean values. Defaults to ImageNet statistics.
        std: Normalization std values. Defaults to ImageNet statistics.
        resize_size: Size to resize images to.
        crop_size: Size for random crop.
        augmentation: If True, applies data augmentation.
        interpolation: Interpolation mode for resizing. Defaults to BICUBIC for high quality.
        random_horizontal_flip: If True, applies random horizontal flip.
        random_crop: If True, applies random crop with padding.
        random_crop_padding: Padding size for random crop.
        random_rotation: Maximum rotation angle in degrees. 0 disables rotation.
        color_jitter: Dictionary with color jitter parameters (brightness, contrast, saturation, hue).
                     If None or empty, color jitter is disabled.

    Returns:
        Composed transform pipeline.
    """
    logger.debug(f"Creating train transform with augmentation={augmentation}, interpolation={interpolation}")
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    transform_list = []
    
    # Resize with high-quality interpolation and anti-aliasing
    transform_list.append(transforms.Resize(resize_size, interpolation=interpolation, antialias=True))
    
    if augmentation:
        # Data augmentation
        if random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if random_crop:
            transform_list.append(transforms.RandomCrop(crop_size, padding=random_crop_padding))
        
        if random_rotation > 0:
            transform_list.append(transforms.RandomRotation(random_rotation))
        
        if color_jitter and any(v > 0 for v in color_jitter.values()):
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter.get('brightness', 0),
                contrast=color_jitter.get('contrast', 0),
                saturation=color_jitter.get('saturation', 0),
                hue=color_jitter.get('hue', 0),
            ))
    
    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_eval_transform(
    mean: list[float] | None = None,
    std: list[float] | None = None,
    resize_size: int = 224,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
) -> transforms.Compose:
    """Get transform pipeline for validation/test data.

    Args:
        mean: Normalization mean values. Defaults to ImageNet statistics.
        std: Normalization std values. Defaults to ImageNet statistics.
        resize_size: Size to resize images to.
        interpolation: Interpolation mode for resizing. Defaults to BICUBIC for high quality.

    Returns:
        Composed transform pipeline.
    """
    logger.debug(f"Creating eval transform with interpolation={interpolation}")
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    transform_list = [
        transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    return transforms.Compose(transform_list)


def get_cifar10_transforms(
    use_cifar10_stats: bool = False,
    augmentation: bool = True,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
    random_horizontal_flip: bool = True,
    random_crop: bool = True,
    random_crop_padding: int = 16,
    random_rotation: int = 0,
    color_jitter: dict[str, float] | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Get standard CIFAR-10 transforms.

    Args:
        use_cifar10_stats: If True, uses CIFAR-10 statistics.
                           If False, uses ImageNet statistics.
        augmentation: If True, applies augmentation to training transform.
        interpolation: Interpolation mode for resizing.
        random_horizontal_flip: If True, applies random horizontal flip.
        random_crop: If True, applies random crop with padding.
        random_crop_padding: Padding size for random crop.
        random_rotation: Maximum rotation angle in degrees. 0 disables rotation.
        color_jitter: Dictionary with color jitter parameters.

    Returns:
        Tuple of (train_transform, eval_transform).
    """
    logger.debug(f"Getting CIFAR-10 transforms (use_cifar10_stats={use_cifar10_stats})")
    mean = CIFAR10_MEAN if use_cifar10_stats else IMAGENET_MEAN
    std = CIFAR10_STD if use_cifar10_stats else IMAGENET_STD
    
    train_transform = get_train_transform(
        mean=mean, std=std, augmentation=augmentation,
        interpolation=interpolation,
        random_horizontal_flip=random_horizontal_flip,
        random_crop=random_crop,
        random_crop_padding=random_crop_padding,
        random_rotation=random_rotation,
        color_jitter=color_jitter,
    )
    eval_transform = get_eval_transform(mean=mean, std=std, interpolation=interpolation)
    
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
    logger.debug(f"Loading transform config: {config_name}")
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
        logger.error(f"Unknown config '{config_name}'")
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[config_name]


def get_advanced_train_transform(
    mean: list[float] | None = None,
    std: list[float] | None = None,
    resize_size: int = 224,
    use_randaugment: bool = True,
    use_random_erasing: bool = True,
) -> transforms.Compose:
    """Get advanced augmentation pipeline (RandAugment + RandomErasing)."""
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    transform_list = []
    transform_list.append(transforms.Resize(resize_size))
    
    if use_randaugment:
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    else:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomCrop(resize_size, padding=16 if resize_size==224 else 4))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    if use_random_erasing:
        transform_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))

    return transforms.Compose(transform_list)

