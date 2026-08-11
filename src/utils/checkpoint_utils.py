"""
checkpoint_utils.py — Locate and load trained checkpoints across runs.

Checkpoints produced by ``src/training/train_lab2_models.py`` live in
``experiments/runs/<ts>_<run_name>/checkpoints/<run_name>_{best,last}.pt`` and
are tracked in ``experiments/runs/registry.json`` (run_name -> latest run dir).

Legacy checkpoints (state-dict-only, from the old notebook-based training) live
flat in ``experiments/checkpoints/<run_name>_best.pt`` and remain loadable.

Usage:
    from src.utils.checkpoint_utils import find_best_checkpoint, load_model_weights

    path = find_best_checkpoint("ResNet18-sota")     # newest artifact first
    model = load_model_weights(model, path, device)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _announce(resource: str, path) -> None:
    """Print the full path of a loaded/required resource (consistent logging)."""
    print(f"[load] {resource} -> {path if path is not None else 'NOT FOUND'}")


def find_best_checkpoint(
    run_name: str,
    runs_root: str | Path = "experiments/runs",
    fallback_dir: str | Path = "experiments/checkpoints",
) -> Path | None:
    """Return the most recent ``<run_name>_best.pt`` path, or None.

    Relative paths are resolved against the repository root, so notebooks can
    call this from any working directory.

    Search order:
        1. registry.json -> latest run dir for *run_name*
        2. newest ``*_<run_name>*`` run dir found under *runs_root*
        3. legacy ``<fallback_dir>/<run_name>_best.pt``
    """
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def _abs(path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    runs_root = _abs(runs_root)
    fallback_dir = _abs(fallback_dir)

    # 1. registry
    registry = runs_root / "registry.json"
    if registry.exists():
        try:
            reg = json.loads(registry.read_text(encoding="utf-8"))
            run_dir = reg.get(run_name)
            if run_dir:
                cand = Path(run_dir) / "checkpoints" / f"{run_name}_best.pt"
                if cand.exists():
                    return cand
        except (json.JSONDecodeError, OSError):
            pass

    # 2. newest matching run dir (name appears anywhere in the dir name);
    #    directories starting with "_" are smoke/test roots — always skipped
    best: Path | None = None
    best_mtime = 0.0
    if runs_root.is_dir():
        for run_dir in runs_root.iterdir():
            if (run_dir.is_dir() and not run_dir.name.startswith("_")
                    and run_name in run_dir.name):
                cand = run_dir / "checkpoints" / f"{run_name}_best.pt"
                if cand.exists() and cand.stat().st_mtime > best_mtime:
                    best, best_mtime = cand, cand.stat().st_mtime

    # 3. legacy flat checkpoints
    if best is None:
        legacy = Path(fallback_dir) / f"{run_name}_best.pt"
        if legacy.exists():
            best = legacy

    _announce(f"best checkpoint [{run_name}]", best)
    return best


def find_latest_run_dir(
    run_name: str,
    runs_root: str | Path = "experiments/runs",
) -> Path | None:
    """Return the most recent run directory containing *run_name*, or None.

    Used to resolve ``--resume`` targets: the resume source is the newest
    ``runs/<ts>_<run_name>/`` dir (whose ``checkpoints/<run_name>_last.pt`` is
    then loaded). Smoke roots (``_*`` dirs) are skipped.
    """
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    p = Path(runs_root)
    runs_root = p if p.is_absolute() else _PROJECT_ROOT / p
    best: Path | None = None
    best_mtime = 0.0
    if runs_root.is_dir():
        for run_dir in runs_root.iterdir():
            ck = run_dir / "checkpoints" / f"{run_name}_last.pt"
            if (run_dir.is_dir() and not run_dir.name.startswith("_")
                    and run_name in run_dir.name
                    and ck.exists() and ck.stat().st_size > 1024):  # skip truncated/corrupt
                mtime = ck.stat().st_mtime
                if mtime > best_mtime:
                    best, best_mtime = run_dir, mtime
    _announce(f"latest run dir [{run_name}]",
              best / "checkpoints" / f"{run_name}_last.pt" if best else None)
    return best


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load weights from a full-state or state-dict-only checkpoint (safe load).

    New full-state checkpoints (see ``train_model.save_checkpoint``) contain a
    ``model_state_dict`` key; legacy checkpoints are bare state dicts.
    """
    _announce("model weights", Path(path))
    state: Any = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
