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


def latest_run_dir(run_root: str | Path, run_name: str | None = None) -> Path | None:
    """Find the registered or newest `experiments/lab3/runs/<ts>_<run_name>/` for a run name, or newest overall."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        return None
    if run_name:
        registry_path = run_root / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, encoding="utf-8") as f:
                    registry = json.load(f)
                if run_name in registry:
                    reg_path = Path(registry[run_name])
                    if reg_path.is_dir():
                        return reg_path
            except (json.JSONDecodeError, OSError):
                pass
        matches = [p for p in run_root.glob(f"*_{run_name}") if p.is_dir()]
    else:
        matches = [p for p in run_root.glob("*_*") if p.is_dir() and (p / "checkpoints").exists()]

    if not matches:
        return None
    return sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)[0]


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


def resolve_run_files(run_dir: str | Path) -> dict[str, Path | None]:
    """Find (best.pt, last.pt, swa.pt, history.jsonl) in a run directory regardless of timestamp formatting."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"

    best_pt = None
    last_pt = None
    swa_pt = None
    history_jsonl = None

    if ckpt_dir.exists():
        best_matches = sorted(ckpt_dir.glob("*_best.pt"))
        last_matches = sorted(ckpt_dir.glob("*_last.pt"))
        swa_matches = sorted(ckpt_dir.glob("*_swa.pt"))

        if best_matches:
            best_pt = best_matches[0]
        if last_matches:
            last_pt = last_matches[0]
        if swa_matches:
            swa_pt = swa_matches[0]

    if metrics_dir.exists():
        hist_matches = sorted(metrics_dir.glob("*_history.jsonl"))
        if hist_matches:
            history_jsonl = hist_matches[0]

    return {
        "best": best_pt,
        "last": last_pt,
        "swa": swa_pt,
        "history": history_jsonl,
    }


def update_registry(run_root: str | Path, run_name: str, run_dir: str | Path) -> None:
    """Point `<run_root>/registry.json` at the latest run dir."""
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


def average_checkpoints(checkpoint_paths: list[str | Path], output_path: str | Path) -> Path:
    """Perform Stochastic Weight Averaging (SWA) over multiple checkpoint files.

    Averages parameter-wise `model_state_dict`s across the specified checkpoints
    and persists the averaged state dict into `output_path`.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths list cannot be empty")

    output_path = Path(output_path)
    loaded_ckpts = [safe_load_checkpoint(p, device="cpu") for p in checkpoint_paths]

    base_state = loaded_ckpts[0]
    avg_model_state = {}
    ref_weights = base_state["model_state_dict"]

    for key, val in ref_weights.items():
        if torch.is_floating_point(val):
            stacked = torch.stack([ckpt["model_state_dict"][key].float() for ckpt in loaded_ckpts], dim=0)
            avg_model_state[key] = torch.mean(stacked, dim=0).to(val.dtype)
        else:
            avg_model_state[key] = val.clone()

    new_state = {
        **base_state,
        "model_state_dict": avg_model_state,
        "swa_num_checkpoints": len(checkpoint_paths),
        "swa_source_paths": [str(p) for p in checkpoint_paths],
    }
    save_checkpoint(output_path, new_state)
    return output_path

