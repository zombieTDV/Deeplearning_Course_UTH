"""
config.py — Configuration loader for data pipeline.

Usage:
    from data.config import load_config
    config = load_config("configs/data.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory paths for training outputs
TB_LOG_DIR = str(_PROJECT_ROOT / "experiments" / "tensorboard_logs")
CKPT_DIR = str(_PROJECT_ROOT / "experiments" / "checkpoints")


def load_config(config_path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary with configuration parameters.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_dataset_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get dataset configuration.

    Args:
        config: Full configuration dictionary. If None, loads from default path.

    Returns:
        Dataset configuration dictionary.
    """
    if config is None:
        config = load_config()
    
    return config.get('dataset', {})


def get_normalization_config(
    config: dict[str, Any] | None = None,
    use_imagenet: bool = True,
) -> dict[str, list[float]]:
    """Get normalization configuration.

    Args:
        config: Full configuration dictionary. If None, loads from default path.
        use_imagenet: If True, returns ImageNet stats. Otherwise, CIFAR-10 stats.

    Returns:
        Dictionary with 'mean' and 'std' lists.
    """
    if config is None:
        config = load_config()
    
    norm_config = config.get('normalization', {})
    
    if use_imagenet:
        return norm_config.get('imagenet', {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]})
    else:
        return norm_config.get('cifar10', {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]})


def get_dataloader_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get DataLoader configuration.

    Args:
        config: Full configuration dictionary. If None, loads from default path.

    Returns:
        DataLoader configuration dictionary.
    """
    if config is None:
        config = load_config()
    
    return config.get('dataloader', {})


def get_transform_config(
    config: dict[str, Any] | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Get transform configuration.

    Args:
        config: Full configuration dictionary. If None, loads from default path.
        split: Either "train" or "eval".

    Returns:
        Transform configuration dictionary.

    Raises:
        ValueError: If split is not "train" or "eval".
    """
    if config is None:
        config = load_config()
    
    transforms_config = config.get('transforms', {})
    
    if split == "train":
        return transforms_config.get('train', {})
    elif split == "eval":
        return transforms_config.get('eval', {})
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'eval'.")
