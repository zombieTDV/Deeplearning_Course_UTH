"""
load_cifar10.py — CIFAR-10 loading, split persistence, DataLoader creation.

Usage:
    from src.data.load_cifar10 import get_cifar10_loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)

The train/val/test split is generated ONCE and persisted to
data/processed/cifar10_split_seed42.json.  Every subsequent call (across
all phases — model, training, eval) reuses that same file.  No script
may regenerate or mutate the split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Absolute path so CIFAR-10 always downloads to the project's data/raw/
# regardless of the current working directory at runtime.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = str(_PROJECT_ROOT / "data" / "raw")
SPLIT_FILE = str(_PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")
SPLIT_SEED = 42
TRAIN_RATIO = 0.9  # 45k train / 5k val out of 50k

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Split persistence (project-wide — generated once, reused forever)
# ---------------------------------------------------------------------------
def _ensure_split() -> dict:
    """Return the split dict, loaded from disk or generated on first call."""
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            return json.load(f)

    # --- First call ever: generate and persist ---
    full_train = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True
    )
    _ = torchvision.datasets.CIFAR10(  # ensure test set is downloaded too
        root=DATA_ROOT, train=False, download=True
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
# Transforms
# ---------------------------------------------------------------------------
def _train_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_cifar10_loaders(
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    The split is loaded from ``data/processed/cifar10_split_seed42.json``.
    If the file does not exist yet it is generated once and persisted.
    Every subsequent call (in any phase) reuses the persisted file.
    """
    split = _ensure_split()

    # Load full datasets with the appropriate transforms.
    # We create separate CIFAR10 instances per transform variant.
    # Memory overhead is acceptable for CIFAR-10 (~150 MB raw).
    train_full = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True,
        transform=_train_transform(), download=True,
    )
    val_full = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True,
        transform=_eval_transform(), download=True,
    )
    test_full = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False,
        transform=_eval_transform(), download=True,
    )

    train_set = Subset(train_full, split["train_indices"])
    val_set = Subset(val_full, split["val_indices"])
    test_set = Subset(test_full, split["test_indices"])

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
