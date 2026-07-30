"""
dataloader.py — DataLoader creation for CIFAR-10 with split persistence.

Usage:
    from data.dataloader import get_cifar10_loaders
    from data.transforms import get_cifar10_transforms
    
    train_transform, eval_transform = get_cifar10_transforms()
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=64
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data" / "external" / "CIFAR-10")
SPLIT_FILE = str(_PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")
SPLIT_SEED = 42
TRAIN_RATIO = 0.9  # 45k train / 5k val out of 50k


# ---------------------------------------------------------------------------
# Split persistence (project-wide — generated once, reused forever)
# ---------------------------------------------------------------------------
def _ensure_split() -> dict[str, Any]:
    """Return the split dict, loaded from disk or generated on first call."""
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            return json.load(f)

    # --- First call ever: generate and persist ---
    full_train = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_ROOT, train=True, download=True
    )
    _ = torchvision.datasets.CIFAR10(  # ensure test set is downloaded too
        root=DEFAULT_DATA_ROOT, train=False, download=True
    )
    n_full = len(full_train)  # 50k

    train_len = int(n_full * TRAIN_RATIO)  # 45k
    val_len = n_full - train_len           # 5k

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_subset, val_subset = random_split(
        full_train, [train_len, val_len], generator=generator
    )

    split = {
        "seed": SPLIT_SEED,
        "train_ratio": TRAIN_RATIO,
        "train_indices": train_subset.indices,
        "val_indices": val_subset.indices,
        # Test set uses the full official 10k — every index 0..9999
        "test_indices": list(range(10_000)),
    }

    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)

    return split


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_cifar10_loaders(
    batch_size: int = 64,
    num_workers: int = 2,
    train_transform: object | None = None,
    eval_transform: object | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    The split is loaded from data/processed/cifar10_split_seed42.json.
    If the file does not exist yet it is generated once and persisted.
    Every subsequent call (in any phase) reuses the persisted file.

    Args:
        batch_size: Batch size for DataLoaders.
        num_workers: Number of worker processes for data loading.
        train_transform: Transform to apply to training data.
        eval_transform: Transform to apply to validation/test data.
        pin_memory: If True, uses pinned memory for faster GPU transfer.
        persistent_workers: If True, keeps workers alive between epochs.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from data.transforms import get_train_transform, get_eval_transform
    
    split = _ensure_split()

    # Use default transforms if not provided
    if train_transform is None:
        train_transform = get_train_transform()
    if eval_transform is None:
        eval_transform = get_eval_transform()

    # Load full datasets with the appropriate transforms.
    # We create separate CIFAR10 instances per transform variant.
    # Memory overhead is acceptable for CIFAR-10 (~150 MB raw).
    train_full = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_ROOT, train=True,
        transform=train_transform, download=False,
    )
    val_full = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_ROOT, train=True,
        transform=eval_transform, download=False,
    )
    test_full = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_ROOT, train=False,
        transform=eval_transform, download=False,
    )

    train_set = Subset(train_full, split["train_indices"])
    val_set = Subset(val_full, split["val_indices"])
    test_set = Subset(test_full, split["test_indices"])

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader


def get_single_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a single DataLoader for a given dataset.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size for the DataLoader.
        shuffle: If True, shuffles the data.
        num_workers: Number of worker processes for data loading.
        pin_memory: If True, uses pinned memory for faster GPU transfer.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
