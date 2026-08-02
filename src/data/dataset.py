"""
dataset.py — CIFAR-10 dataset download, loading, and verification.

Usage:
    from data.dataset import download_cifar10, load_cifar10_dataset
    download_cifar10(root="data/external/CIFAR-10")
    train_set, test_set = load_cifar10_dataset(root="data/external/CIFAR-10")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data" / "external" / "CIFAR-10")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def download_cifar10(root: str = DEFAULT_DATA_ROOT) -> None:
    """Download CIFAR-10 dataset to the specified directory.

    Args:
        root: Directory where the dataset will be stored.
              Defaults to data/external/CIFAR-10.

    Raises:
        RuntimeError: If download fails.
    """
    logger.info(f"Downloading CIFAR-10 dataset to {root}")
    Path(root).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download both train and test sets
        torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        logger.info("CIFAR-10 dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download CIFAR-10 dataset: {e}")
        raise


def load_cifar10_dataset(
    root: str = DEFAULT_DATA_ROOT,
    train: bool = True,
    transform: object | None = None,
    auto_download: bool = True,
) -> Dataset:
    """Load CIFAR-10 dataset.

    Automatically downloads the dataset if it is not found locally.

    Args:
        root: Directory where the dataset is stored.
        train: If True, loads training set. If False, loads test set.
        transform: Optional transform to be applied on a sample.
        auto_download: If True, automatically downloads dataset if missing.

    Returns:
        CIFAR10 dataset instance.

    Raises:
        RuntimeError: If dataset is not found and auto_download is False.
    """
    logger.debug(f"Loading CIFAR-10 dataset from {root} (train={train})")
    
    # Check if dataset exists
    data_path = Path(root)
    if not data_path.exists():
        if auto_download:
            logger.info(f"Dataset not found at {root}, auto-downloading...")
            download_cifar10(root=root)
        else:
            raise RuntimeError(f"Dataset not found at {root} and auto_download is False")
    
    return torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=auto_download,  # Let torchvision handle download if needed
        transform=transform,
    )


def get_dataset_info(dataset: Dataset) -> dict[str, Any]:
    """Get information about the dataset.

    Args:
        dataset: PyTorch Dataset instance.

    Returns:
        Dictionary with dataset information including:
            - name: Dataset name
            - num_samples: Number of samples
            - num_classes: Number of classes
            - classes: List of class names
    """
    if hasattr(dataset, 'classes'):
        classes = dataset.classes
    else:
        classes = getattr(dataset, 'classes', ['unknown'])

    info = {
        'name': dataset.__class__.__name__,
        'num_samples': len(dataset),
        'num_classes': len(classes),
        'classes': classes,
    }
    logger.debug(f"Dataset info: {info}")
    return info
