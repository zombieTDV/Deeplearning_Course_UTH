"""test_checkpoint_utils.py — artifact discovery + safe loading (TST-2, SEC-1)."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.checkpoint_utils import (
    find_best_checkpoint,
    find_latest_run_dir,
    load_model_weights,
)


def _tiny_state_dict():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 2)).state_dict()


def _write_run(tmp: Path, run_name: str, ts: str) -> Path:
    run_dir = tmp / f"{ts}_{run_name}" / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_tiny_state_dict(), run_dir / f"{run_name}_best.pt")
    torch.save(_tiny_state_dict(), run_dir / f"{run_name}_last.pt")
    return run_dir


def test_find_best_checkpoint_legacy_fallback(tmp_path):
    legacy = tmp_path / "checkpoints"
    legacy.mkdir()
    torch.save(_tiny_state_dict(), legacy / "ResNet18-sota_best.pt")
    found = find_best_checkpoint("ResNet18-sota", runs_root=tmp_path / "runs",
                                 fallback_dir=legacy)
    assert found is not None and found.name == "ResNet18-sota_best.pt"


def test_find_best_checkpoint_newest_run_wins(tmp_path):
    _write_run(tmp_path, "ResNet18-sota", "20260810_100000")
    time.sleep(0.01)
    _write_run(tmp_path, "ResNet18-sota", "20260810_110000")
    found = find_best_checkpoint("ResNet18-sota", runs_root=tmp_path,
                                 fallback_dir=tmp_path / "none")
    assert found is not None and "110000" in str(found)


def test_find_latest_run_dir_skips_smoke_and_truncated(tmp_path):
    _write_run(tmp_path, "ResNet18-sota", "20260810_100000")
    real = tmp_path / "20260810_100000_ResNet18-sota"
    # smoke root (underscore-prefixed dir) must be ignored
    (tmp_path / "_smoke" / "20260810_120000_ResNet18-sota" / "checkpoints").mkdir(parents=True)
    # truncated _last.pt (< 1024 B) must be ignored
    truncated = tmp_path / "20260810_130000_ResNet18-sota" / "checkpoints"
    truncated.mkdir(parents=True)
    (truncated / "ResNet18-sota_last.pt").write_bytes(b"x" * 500)
    latest = find_latest_run_dir("ResNet18-sota", runs_root=tmp_path)
    assert latest == real


def test_load_model_weights_full_state_and_bare(tmp_path):
    run_dir = _write_run(tmp_path, "M", "20260810_100000")
    bare = run_dir / "M_best.pt"
    # bare state dict
    model = nn.Sequential(nn.Linear(4, 2))
    model2 = load_model_weights(model, bare, torch.device("cpu"))
    assert model2 is model
    # full-state checkpoint
    full_path = tmp_path / "full.pt"
    torch.save({"model_state_dict": _tiny_state_dict()}, full_path)
    model3 = nn.Sequential(nn.Linear(4, 2))
    load_model_weights(model3, full_path, torch.device("cpu"))
    # weights changed from random init -> loaded correctly
    torch.manual_seed(123)
    fresh = nn.Sequential(nn.Linear(4, 2))
    assert not all(torch.equal(a, b) for a, b in zip(model3.parameters(),
                                                     fresh.parameters(), strict=False))


def test_registry_lookup(tmp_path):
    run_dir = _write_run(tmp_path, "ResNet18-sota", "20260810_100000")
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"ResNet18-sota": str(run_dir)}), encoding="utf-8")
    found = find_best_checkpoint("ResNet18-sota", runs_root=tmp_path,
                                 fallback_dir=tmp_path / "none")
    assert found is not None and "100000" in str(found)
