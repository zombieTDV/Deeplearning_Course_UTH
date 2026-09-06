"""Shared pytest fixtures/guards for the LAB2 test suite."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cifar-10-batches-py"

# True when the real dataset is present locally. Data-dependent tests are
# skipped in CI/other machines so the suite stays runnable everywhere
# (audit TST-4: unit tests use synthetic fixtures; integration tests skip).
DATA_PRESENT = DATA_ROOT.is_dir()

requires_data = pytest.mark.skipif(
    not DATA_PRESENT,
    reason="CIFAR-10 dataset not present at data/raw (run train_lab2_models or place data)",
)


def pretrained_available() -> bool:
    """Best-effort check that torchvision pretrained weights are cached locally.

    Building pretrained models triggers a download on first use; skip when the
    network is unavailable (CI) instead of failing.
    """
    import torch
    import torchvision  # noqa: F401

    cache = Path(torch.hub.get_dir()) / "checkpoints"
    return cache.exists() and any(cache.iterdir())


requires_pretrained = pytest.mark.skipif(
    not pretrained_available(),
    reason="torchvision pretrained weights not cached locally",
)
