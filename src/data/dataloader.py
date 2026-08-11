"""
dataloader.py — DataLoader creation for CIFAR-10 with split persistence.

Usage:
    from src.data.dataloader import get_cifar10_loaders

    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)

Defaults (batch size, worker count, data root) are taken from the centralized
``configs/data.yaml`` when present (see ``src.data.config``), falling back to
the constants below — editing the YAML actually takes effect.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (used when configs/data.yaml is absent or incomplete)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Single canonical data root: every loader (canonical and legacy) points at
# data/raw — the legacy external-data directory was retired to avoid
# duplicate dataset copies.
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data" / "raw")
SPLIT_FILE = str(_PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")
SPLIT_SEED = 42
TRAIN_RATIO = 0.9  # 45k train / 5k val out of 50k
DEFAULT_BATCH_SIZE = 64
# num_workers=0 is the safe default: multiprocess loading was unstable on
# Python 3.14 (see agents/bugs/BUG_01_DATALOADER_BROKEN_PIPE_PYTHON314.md).
DEFAULT_NUM_WORKERS = 0


# ---------------------------------------------------------------------------
# Config (configs/data.yaml) — single source of truth for loader defaults
# ---------------------------------------------------------------------------
def _config_defaults() -> dict[str, Any]:
    """Return loader defaults from configs/data.yaml, or {} on any failure."""
    try:
        from src.data.config import load_config
        cfg = load_config("configs/data.yaml")
        dl = cfg.get("dataloader", {})
        ds = cfg.get("dataset", {})
        root = ds.get("root", "data/raw")
        return {
            "batch_size": int(dl.get("batch_size", DEFAULT_BATCH_SIZE)),
            "num_workers": int(dl.get("num_workers", DEFAULT_NUM_WORKERS)),
            "data_root": str(_PROJECT_ROOT / root) if not os.path.isabs(root) else root,
            "pin_memory": bool(dl.get("pin_memory", True)),
            "persistent_workers": bool(dl.get("persistent_workers", False)),
        }
    except Exception:
        return {}


def _cifar10_present(data_root: str | None = None) -> bool:
    """True if the CIFAR-10 batch files already exist at *data_root*.

    Used to gate ``download=True`` so we never re-fetch the dataset when it is
    already on disk.
    """
    root = Path(data_root) if data_root else Path(DEFAULT_DATA_ROOT)
    return (root / "cifar-10-batches-py").is_dir()


class _ApplyTransform:
    """Apply a transform on access to a shared raw dataset.

    Lets one raw ``CIFAR10`` instance serve train AND validation subsets with
    different transforms, avoiding a second full in-memory copy (PERF-3).
    """

    def __init__(self, dataset: torch.utils.data.Dataset, transform: object | None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Split persistence (project-wide — generated once, reused forever)
# ---------------------------------------------------------------------------
def _ensure_split() -> dict[str, Any]:
    """Return the split dict, loaded from disk or generated on first call."""
    if os.path.exists(SPLIT_FILE):
        logger.debug(f"Loading split from {SPLIT_FILE}")
        with open(SPLIT_FILE) as f:
            return json.load(f)

    # --- First call ever: generate and persist ---
    logger.info(f"Generating new split with seed={SPLIT_SEED}")
    full_train = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_ROOT, train=True, download=not _cifar10_present()
    )
    _ = torchvision.datasets.CIFAR10(  # ensure test set is downloaded too
        root=DEFAULT_DATA_ROOT, train=False, download=not _cifar10_present()
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

    logger.info(f"Split persisted to {SPLIT_FILE}")
    return split


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_cifar10_loaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
    train_transform: object | None = None,
    eval_transform: object | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    data_root: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    The split is loaded from data/processed/cifar10_split_seed42.json.
    If the file does not exist yet it is generated once and persisted.
    Every subsequent call (in any phase) reuses the persisted file.

    Defaults resolve from ``configs/data.yaml`` (``src.data.config``) when
    present; explicit arguments override them.

    Args:
        batch_size: Batch size for DataLoaders (default: config, else 64).
        num_workers: Worker processes for data loading. Kept 0 by default on
            Python 3.14 — see BUG-01 (BrokenPipeError); set explicitly to
            re-enable workers on a stable runtime.
        train_transform: Transform applied to training samples.
        eval_transform: Transform applied to validation/test samples.
        pin_memory: If True, uses pinned memory for faster GPU transfer.
        persistent_workers: If True, keeps workers alive between epochs.
        data_root: CIFAR-10 root (default: config/data/raw).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    cfg = _config_defaults()
    data_root = data_root or cfg.get("data_root", DEFAULT_DATA_ROOT)
    batch_size = batch_size if batch_size is not None else cfg.get("batch_size", DEFAULT_BATCH_SIZE)
    num_workers = num_workers if num_workers is not None else cfg.get("num_workers", DEFAULT_NUM_WORKERS)
    pin_memory = pin_memory if pin_memory is not None else cfg.get("pin_memory", True)
    persistent_workers = persistent_workers if persistent_workers is not None else cfg.get("persistent_workers", False)

    logger.info(f"Creating CIFAR-10 DataLoaders with batch_size={batch_size} "
                f"num_workers={num_workers} root={data_root}")
    print(f"[load] CIFAR-10 dataset root -> {Path(data_root)}")
    from src.data.transforms import get_eval_transform, get_train_transform

    split = _ensure_split()

    if num_workers == 0:
        persistent_workers = False

    # Use default transforms if not provided
    if train_transform is None:
        train_transform = get_train_transform()
    if eval_transform is None:
        eval_transform = get_eval_transform()

    # Build ONE raw train instance + ONE raw test instance, then apply
    # per-subset transforms via _ApplyTransform (avoids a third in-memory copy).
    train_raw = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=None,
        download=not _cifar10_present(data_root),
    )
    test_raw = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=None,
        download=not _cifar10_present(data_root),
    )

    train_set = Subset(_ApplyTransform(train_raw, train_transform), split["train_indices"])
    val_set = Subset(_ApplyTransform(train_raw, eval_transform), split["val_indices"])
    test_set = Subset(_ApplyTransform(test_raw, eval_transform), split["test_indices"])

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

    logger.debug(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}, Test samples: {len(test_set)}")
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
    logger.debug(f"Creating single DataLoader with batch_size={batch_size}, shuffle={shuffle}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
