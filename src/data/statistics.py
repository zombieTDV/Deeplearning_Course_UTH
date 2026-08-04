"""
statistics.py — Compute and manage dataset statistics for normalization.

Usage:
    from data.statistics import compute_dataset_statistics, save_statistics, load_statistics
    from data.dataset import load_cifar10_dataset
    
    train_set = load_cifar10_dataset(train=True)
    mean, std = compute_dataset_statistics(train_set)
    save_statistics(mean, std, "data/processed/dataset_statistics.json")
    stats = load_statistics("data/processed/dataset_statistics.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_FILE = str(_PROJECT_ROOT / "data" / "processed" / "dataset_statistics.json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_dataset_statistics(
    dataset: Dataset,
    batch_size: int = 1000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and standard deviation of dataset for normalization.

    Args:
        dataset: PyTorch Dataset instance with image tensors.
        batch_size: Batch size for processing.
        device: Device to use for computation. If None, uses CPU.

    Returns:
        Tuple of (mean, std) as tensors with shape (3,).
    """
    logger.info(f"Computing dataset statistics with batch_size={batch_size}")
    if device is None:
        device = torch.device('cpu')
    
    # Load dataset into memory for efficient computation
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    sum_ = torch.zeros(3).to(device)
    sum_sq = torch.zeros(3).to(device)
    count = 0
    
    for images, _ in loader:
        images = images.to(device)
        # images shape: (batch, channels, height, width)
        batch_sum = images.sum(dim=[0, 2, 3])  # sum over batch, height, width
        batch_sum_sq = (images ** 2).sum(dim=[0, 2, 3])
        
        sum_ += batch_sum
        sum_sq += batch_sum_sq
        count += images.size(0) * images.size(2) * images.size(3)
    
    mean = sum_ / count
    std = torch.sqrt((sum_sq / count) - (mean ** 2))
    
    logger.debug(f"Computed mean: {mean.tolist()}, std: {std.tolist()}")
    return mean.cpu(), std.cpu()


def save_statistics(
    mean: torch.Tensor | list[float],
    std: torch.Tensor | list[float],
    filepath: str = DEFAULT_STATS_FILE,
) -> None:
    """Save dataset statistics to JSON file.

    Args:
        mean: Mean values as tensor or list.
        std: Standard deviation values as tensor or list.
        filepath: Path to save the statistics file.
    """
    logger.info(f"Saving statistics to {filepath}")
    if isinstance(mean, torch.Tensor):
        mean = mean.tolist()
    if isinstance(std, torch.Tensor):
        std = std.tolist()
    
    stats = {
        'mean': mean,
        'std': std,
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.debug(f"Statistics saved: {stats}")


def load_statistics(filepath: str = DEFAULT_STATS_FILE) -> dict[str, Any]:
    """Load dataset statistics from JSON file.

    Args:
        filepath: Path to the statistics file.

    Returns:
        Dictionary with 'mean' and 'std' keys.

    Raises:
        FileNotFoundError: If the statistics file does not exist.
    """
    logger.debug(f"Loading statistics from {filepath}")
    with open(filepath, 'r') as f:
        stats = json.load(f)
    
    logger.debug(f"Loaded statistics: {stats}")
    return stats


def get_normalization_transform(
    mean: torch.Tensor | list[float] | None = None,
    std: torch.Tensor | list[float] | None = None,
    stats_file: str | None = None,
) -> tuple[list[float], list[float]]:
    """Get normalization parameters for transforms.

    Args:
        mean: Custom mean values. If None, loads from stats_file.
        std: Custom std values. If None, loads from stats_file.
        stats_file: Path to statistics file if mean/std not provided.

    Returns:
        Tuple of (mean, std) as lists.

    Raises:
        ValueError: If neither custom values nor stats_file are provided.
    """
    if mean is None or std is None:
        if stats_file is None:
            raise ValueError("Must provide either mean/std or stats_file")
        stats = load_statistics(stats_file)
        mean = stats['mean'] if mean is None else mean
        std = stats['std'] if std is None else std
    
    if isinstance(mean, torch.Tensor):
        mean = mean.tolist()
    if isinstance(std, torch.Tensor):
        std = std.tolist()
    
    logger.debug(f"Normalization parameters: mean={mean}, std={std}")
    return mean, std
