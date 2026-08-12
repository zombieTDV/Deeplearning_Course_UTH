"""checkpoint_utils — safe checkpoint I/O and run-directory helpers.

Owned by `src/utils/` (audit finding AQ-1). Enforces `weights_only=True` on all
`torch.load` calls per LOGGING_CHECKPOINT_RULES.md §3 and PYTORCH_FRAMEWORK_RULES.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def safe_load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a full-state checkpoint dict with `weights_only=True` (security rule)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint {path} is not a dict")
    return ckpt


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """Persist a full-state checkpoint dict (model+optimizer+scheduler+rng+history+config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**state, "timestamp": datetime.now().isoformat()}
    torch.save(state, path)


def latest_run_dir(run_root: str | Path, run_name: str) -> Path | None:
    """Find the newest `experiments/runs/<ts>_<run_name>/` for a run name."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        return None
    matches = [p for p in run_root.glob(f"*_{run_name}") if p.is_dir()]
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.name, reverse=True)[0]


def next_run_dir(run_root: str | Path, run_name: str) -> Path:
    """Return a fresh timestamped run dir; never overwrites an existing run."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def checkpoint_paths(run_dir: str | Path, run_name: str) -> tuple[Path, Path]:
    """Return (best, last) checkpoint paths for a run dir."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    return ckpt_dir / f"{run_name}_best.pt", ckpt_dir / f"{run_name}_last.pt"


def update_registry(run_root: str | Path, run_name: str, run_dir: str | Path) -> None:
    """Point `experiments/runs/registry.json` at the latest run dir."""
    registry_path = Path(run_root) / "registry.json"
    registry: dict[str, Any] = {}
    if registry_path.exists():
        try:
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            registry = {}
    registry[run_name] = str(Path(run_dir))
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
