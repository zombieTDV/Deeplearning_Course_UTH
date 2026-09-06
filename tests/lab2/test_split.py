"""test_split.py — deterministic, persisted 45k/5k/10k split (audit TST-2)."""

import json

from src.lab2.data.dataloader import SPLIT_FILE, SPLIT_SEED, TRAIN_RATIO, _ensure_split


def test_split_counts_and_seed():
    split = _ensure_split()
    assert split["seed"] == SPLIT_SEED
    assert split["train_ratio"] == TRAIN_RATIO
    assert len(split["train_indices"]) == 45_000
    assert len(split["val_indices"]) == 5_000
    assert len(split["test_indices"]) == 10_000
    # train + val partition the 50k train samples without overlap
    assert sorted(split["train_indices"] + split["val_indices"]) == list(range(50_000))


def test_split_is_persisted_and_deterministic():
    import os

    assert os.path.exists(SPLIT_FILE)
    with open(SPLIT_FILE) as f:
        on_disk = json.load(f)
    assert on_disk["train_indices"] == _ensure_split()["train_indices"]
    assert on_disk["val_indices"] == _ensure_split()["val_indices"]

