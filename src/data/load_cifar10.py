"""
load_cifar10.py — CIFAR-10 loading, split persistence, DataLoader creation.

This module orchestrates the complete data preparation pipeline:
1. Download CIFAR-10 dataset (if missing)
2. Create or load persistent train/val/test split
3. Apply ImageNet-compatible transforms with high-quality resizing
4. Create PyTorch DataLoaders

Usage:
    from src.data.load_cifar10 import load_cifar10
    train_loader, val_loader, test_loader = load_cifar10()

The train/val/test split is generated ONCE and persisted to
the configured split file. Every subsequent call (across
all phases — model, training, eval) reuses that same file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

from src.data.config import (
    load_config,
    get_dataset_config,
    get_normalization_config,
    get_dataloader_config,
    get_transform_config,
)
from src.data.transforms import get_train_transform, get_eval_transform

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Absolute path so CIFAR-10 always downloads to the project's data/
# regardless of the current working directory at runtime.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default paths (can be overridden by config)
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data" / "external" / "CIFAR-10")
DEFAULT_SPLIT_FILE = str(_PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.9  # 45k train / 5k val out of 50k

# Backward compatibility: expose constants for notebooks
SPLIT_FILE = DEFAULT_SPLIT_FILE
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Split persistence (project-wide — generated once, reused forever)
# ---------------------------------------------------------------------------
def _ensure_split(
    data_root: str,
    split_file: str,
    split_seed: int,
    train_ratio: float,
) -> dict[str, Any]:
    """Return the split dict, loaded from disk or generated on first call.

    Args:
        data_root: Directory where CIFAR-10 is stored.
        split_file: Path to the split JSON file.
        split_seed: Random seed for reproducible splitting.
        train_ratio: Ratio of training data (e.g., 0.9 for 90% train, 10% val).

    Returns:
        Dictionary with split indices.
    """
    if os.path.exists(split_file):
        logger.debug(f"Loading split from {split_file}")
        with open(split_file, "r") as f:
            return json.load(f)

    # --- First call ever: generate and persist ---
    logger.info(f"Generating new split with seed={split_seed}, train_ratio={train_ratio}")
    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True
    )
    _ = torchvision.datasets.CIFAR10(  # ensure test set is downloaded too
        root=data_root, train=False, download=True
    )
    n_full = len(full_train)  # 50k

    train_len = int(n_full * train_ratio)  # 45k
    val_len = n_full - train_len           # 5k

    generator = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = random_split(
        full_train, [train_len, val_len], generator=generator
    )

    split = {
        "seed": split_seed,
        "train_ratio": train_ratio,
        "train_indices": train_subset.indices,
        "val_indices": val_subset.indices,
        # Test set uses the full official 10k — every index 0..9999
        "test_indices": list(range(10_000)),
    }

    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split, f, indent=2)
    
    logger.info(f"Split persisted to {split_file}")
    return split


# ---------------------------------------------------------------------------
# Transform creation
# ---------------------------------------------------------------------------
def _create_train_transform(config: dict[str, Any]) -> transforms.Compose:
    """Create training transform from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        Composed transform pipeline for training.
    """
    norm_config = get_normalization_config(config, use_imagenet=True)
    transform_config = get_transform_config(config, split="train")
    split_config = config.get('split', {})
    
    # Get color jitter config if enabled
    color_jitter_config = transform_config.get('color_jitter', {})
    color_jitter = None
    if color_jitter_config.get('enabled', False):
        color_jitter = {
            'brightness': color_jitter_config.get('brightness', 0),
            'contrast': color_jitter_config.get('contrast', 0),
            'saturation': color_jitter_config.get('saturation', 0),
            'hue': color_jitter_config.get('hue', 0),
        }
    
    # Use BICUBIC interpolation for high-quality resizing
    interpolation = transforms.InterpolationMode.BICUBIC
    
    return get_train_transform(
        mean=norm_config['mean'],
        std=norm_config['std'],
        resize_size=transform_config.get('resize', 224),
        crop_size=transform_config.get('random_crop', 224),
        augmentation=True,
        interpolation=interpolation,
        random_horizontal_flip=transform_config.get('random_horizontal_flip', True),
        random_crop=transform_config.get('random_crop', 224) > 0,
        random_crop_padding=transform_config.get('random_crop_padding', 16),
        random_rotation=transform_config.get('random_rotation', 0),
        color_jitter=color_jitter,
    )


def _create_eval_transform(config: dict[str, Any]) -> transforms.Compose:
    """Create evaluation transform from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        Composed transform pipeline for evaluation.
    """
    norm_config = get_normalization_config(config, use_imagenet=True)
    transform_config = get_transform_config(config, split="eval")
    
    # Use BICUBIC interpolation for high-quality resizing
    interpolation = transforms.InterpolationMode.BICUBIC
    
    return get_eval_transform(
        mean=norm_config['mean'],
        std=norm_config['std'],
        resize_size=transform_config.get('resize', 224),
        interpolation=interpolation,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_cifar10(
    config_path: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    drop_last: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 dataset and create DataLoaders.

    This function orchestrates the complete data preparation pipeline:
    1. Download CIFAR-10 dataset (if missing)
    2. Create or load persistent train/val/test split
    3. Apply ImageNet-compatible transforms with high-quality resizing
    4. Create PyTorch DataLoaders

    Args:
        config_path: Path to configuration file. If None, uses default path.
        batch_size: Override batch size from config.
        num_workers: Override num_workers from config.
        pin_memory: Override pin_memory from config.
        persistent_workers: Override persistent_workers from config.
        drop_last: Override drop_last from config.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Example:
        >>> from src.data.load_cifar10 import load_cifar10
        >>> train_loader, val_loader, test_loader = load_cifar10()
    """
    # Load configuration (use default path if None)
    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "data.yaml"
    config = load_config(config_path)
    
    # Get dataset configuration
    dataset_config = get_dataset_config(config)
    data_root = dataset_config.get('root', DEFAULT_DATA_ROOT)
    auto_download = dataset_config.get('download', True)
    
    # Get split configuration
    split_config = config.get('split', {})
    split_file = split_config.get('split_file', DEFAULT_SPLIT_FILE)
    split_seed = split_config.get('seed', DEFAULT_SPLIT_SEED)
    train_ratio = split_config.get('train_ratio', DEFAULT_TRAIN_RATIO)
    
    # Get DataLoader configuration
    dataloader_config = get_dataloader_config(config)
    batch_size = batch_size if batch_size is not None else dataloader_config.get('batch_size', 64)
    num_workers = num_workers if num_workers is not None else dataloader_config.get('num_workers', 2)
    pin_memory = pin_memory if pin_memory is not None else dataloader_config.get('pin_memory', True)
    persistent_workers = persistent_workers if persistent_workers is not None else dataloader_config.get('persistent_workers', False)
    drop_last = drop_last if drop_last is not None else dataloader_config.get('drop_last', False)
    shuffle_train = dataloader_config.get('shuffle_train', True)
    shuffle_eval = dataloader_config.get('shuffle_eval', False)
    
    # Disable pin_memory for MPS (Apple Silicon) as it's not supported
    # Also disable for CPU since pin_memory is only useful for CUDA
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device_type = torch.device('mps')
    if device_type.type != 'cuda':
        pin_memory = False
    
    logger.info(f"Loading CIFAR-10 from {data_root}")
    logger.info(f"Split file: {split_file}")
    
    # Ensure split exists (generates on first call)
    split = _ensure_split(data_root, split_file, split_seed, train_ratio)
    
    # Create transforms from configuration
    train_transform = _create_train_transform(config)
    eval_transform = _create_eval_transform(config)
    
    # Load full datasets with the appropriate transforms.
    # We create separate CIFAR10 instances per transform variant.
    # Memory overhead is acceptable for CIFAR-10 (~150 MB raw).
    train_full = torchvision.datasets.CIFAR10(
        root=data_root, train=True,
        transform=train_transform, download=auto_download,
    )
    val_full = torchvision.datasets.CIFAR10(
        root=data_root, train=True,
        transform=eval_transform, download=False,
    )
    test_full = torchvision.datasets.CIFAR10(
        root=data_root, train=False,
        transform=eval_transform, download=False,
    )

    train_set = Subset(train_full, split["train_indices"])
    val_set = Subset(val_full, split["val_indices"])
    test_set = Subset(test_full, split["test_indices"])

    logger.info(f"Train: {len(train_set):>6}  |  Val: {len(val_set):>6}  |  Test: {len(test_set):>6}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=shuffle_train, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=shuffle_eval, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=shuffle_eval, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader


# Backward compatibility alias
def get_cifar10_loaders(*args, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Alias for load_cifar10 for backward compatibility.
    
    Deprecated: Use load_cifar10 instead.
    """
    logger.warning("get_cifar10_loaders is deprecated. Use load_cifar10 instead.")
    return load_cifar10(*args, **kwargs)
