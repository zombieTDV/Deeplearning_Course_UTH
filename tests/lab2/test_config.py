"""test_config.py — configs/data.yaml is loaded and consistent (TST-2, CQ-2, ARC-1)."""

import os
from pathlib import Path

from src.lab2.data.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_config_reads_yaml():
    cfg = load_config(str(PROJECT_ROOT / "configs" / "data.yaml"))
    assert cfg["dataset"]["name"] == "CIFAR-10"
    assert cfg["dataset"]["root"] == "data/raw"          # ARC-1: single canonical root
    assert cfg["dataset"]["num_classes"] == 10
    assert cfg["dataloader"]["batch_size"] == 64
    assert cfg["dataloader"]["num_workers"] == 0         # safe default (BUG-01)
    assert cfg["split"]["seed"] == 42
    assert cfg["normalization"]["imagenet"]["mean"] == [0.485, 0.456, 0.406]


def test_data_yaml_root_exists_on_disk():
    root = PROJECT_ROOT / cfg_data_root()
    assert root.exists(), f"configured data root missing: {root}"


def cfg_data_root() -> str:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "data.yaml"))
    return cfg["dataset"]["root"]


def test_no_external_root_references_in_src():
    """ARC-1: no module should point at the retired data/external root."""
    import subprocess

    out = subprocess.run(
        ["git", "grep", "-l", "data/external", "--", "src/", "configs/"],
        capture_output=True, text=True, cwd=PROJECT_ROOT)
    # allow the audit report (historical finding) but not code/config
    offenders = [line for line in out.stdout.splitlines() if "codebase-audit" not in line
                 and os.path.basename(line) != "ERROR_ANALYSIS.md"]
    assert not offenders, f"stale data/external references: {offenders}"

