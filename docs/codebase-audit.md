# Codebase Audit Report

**Project:** Deep Learning Course — CIFAR-10 / FashionMNIST Transfer-Learning Labs
**Date:** 2026-08-08
**Branch:** `LAB2`
**Method:** Static + targeted runtime inspection of `src/`, `tests/`, `notebooks/`, `configs/`, `experiments/`, and `agents/`; dependency matrix checked against the active `.venv`; git history reviewed.

---

## Executive Summary

The codebase is **well-structured and documented**: a clean `src/` package layout (`data`, `models`, `training`, `eval`, `experiments`, `utils`), disciplined experiment tracking under `agents/`, consistent naming conventions, safe checkpoint loading, and a clean, meaningful git history on `LAB2` (16 focused commits, working tree clean).

However, three issues block maturity:

1. **Dependencies are unpinned and the test suite cannot run** (`pytest` and `opencv-python` are declared but not installed).
2. **There is no single source of truth for the CIFAR-10 data root** (`data/raw` vs `data/external/CIFAR-10` across three modules + config) — the direct cause of the re-download bug fixed this week in `dataloader.py`.
3. **Test coverage is minimal** (4 files, none for models/training/eval), and there is **no CI** gate.

Overall health: **Good structure and process, low reproducibility and weak automated verification.** The prioritized action plan (Section 6) closes these gaps.

---

## 1. Findings Summary

| ID | Area | Severity | Title |
|---|---|---|---|
| DEP-1 | Dependencies | **High** | Unpinned `requirements.txt` (non-reproducible) |
| DEP-2 | Dependencies | **High** | `pytest` + `opencv-python` declared but not installed |
| TST-1 | Testing | **High** | Test suite cannot run; no CI gate |
| ARC-1 | Architecture | **Medium** | Data-root inconsistency (`data/raw` vs `data/external/CIFAR-10`) |
| CQ-1 | Code quality | **Medium** | Duplicate/overlapping data-loading modules |
| PERF-1 | Performance | **Medium** | `num_workers=0` everywhere (single-threaded loading) |
| PERF-2 | Performance | **Medium** | Experiments recompute features each run (no caching) |
| TST-2 | Testing | **Medium** | Minimal coverage; hot paths untested |
| TST-3 | Testing | **Medium** | No CI / smoke-test enforcement |
| CQ-2 | Code quality | **Low** | `configs/data.yaml` not consumed by loaders (code/config drift) |
| CQ-4 | Code quality | **Low** | Duplicate SOTA model-building logic in notebooks/scripts |
| DEP-3 | Dependencies | **Low** | `tabulate`, `tqdm` declared but unused |
| DEP-4 | Dependencies | **Low** | Bleeding-edge runtime (Py 3.14, torch 2.13, pandas 3.0) |
| CQ-3 | Code quality | **Low** | `scratch/build_notebook.py` drifts from actual notebooks |
| ARC-3 | Architecture | **Low** | Agents/docs references stale data location |
| SEC-1 | Security | **Low** | Checkpoint loading — mitigated (`weights_only=True`), audit notebooks |
| TST-4 | Testing | **Low** | Tests depend on local data/network |
| PERF-3 | Performance | **Low** | Triple in-memory CIFAR-10 dataset instances |
| PERF-4 | Performance | **Low** | Notebooks not fully partial-run safe (`FORCE_RETRAIN`) |
| CQ-5 | Code quality | **Low** | No linting/type-check configuration |

---

## 2. Code Quality

### CQ-1 — Duplicate / overlapping data-loading modules — **Medium**
- **Description:** CIFAR-10 loading is implemented three times with different roots and APIs: `src/data/dataloader.py` (uses `data/raw`, provides `get_cifar10_loaders`), `src/data/load_cifar10.py` (uses `data/external/CIFAR-10`, also `get_cifar10_loaders`), and `src/data/dataset.py` (uses `data/external/CIFAR-10`, `download_cifar10`/`load_cifar10_dataset`). Notebooks import from different modules (one used the stale `from data.dataloader import …`).
- **Affected:** `src/data/dataloader.py`, `src/data/load_cifar10.py`, `src/data/dataset.py`
- **Remediation:** Pick one loader as canonical (recommend `dataloader.py`), route the others through it, delete the dead duplicate, and add a test that all entry points resolve the same root.

### CQ-2 — Config file not consumed by code — **Low**
- **Description:** `configs/data.yaml` is detailed (dataset root, `num_workers: 2`, batch size 64, split seed) but loaders hardcode their defaults (`num_workers=0` in code vs `2` in config). The config and code have drifted, so editing YAML silently does nothing.
- **Affected:** `configs/data.yaml`, `src/data/config.py`, `src/data/dataloader.py`
- **Remediation:** Make `get_cifar10_loaders` accept an optional config object/path and load defaults from `data.yaml`, or delete the config until it is wired up.

### CQ-3 — Notebook generator drift — **Low**
- **Description:** `src/scratch/build_notebook.py` (1131 lines) regenerates notebooks but hardcodes the stale `data/external/CIFAR-10` root (line ~269) and duplicated notebook code that no longer matches the checked-in notebooks (e.g., after the TTA revert and profiling cell were added directly to `practice_2.ipynb`).
- **Affected:** `src/scratch/build_notebook.py`, `notebooks/practice_2.ipynb`
- **Remediation:** Either retire the generator and treat notebooks as the source of truth, or sync the generator with the current notebooks and regenerate to verify.

### CQ-4 — Duplicate SOTA model-building logic — **Low**
- **Description:** The EXP-07 "full SOTA" unfreezing (`layer3+layer4+fc` / `denseblock3+4+norm5+classifier`) is re-implemented inside `notebooks/practice_2.ipynb` (cell 7) and in `src/experiments/exp_07_*.py`, rather than living once in `src/models/build_model.py`.
- **Affected:** `notebooks/practice_2.ipynb`, `src/experiments/exp_07_resnet_densenet_sota.py`, `src/models/build_model.py`
- **Remediation:** Add `build_resnet18_full_sota` / `build_densenet121_full_sota` to `src/models/build_model.py` and import them everywhere.

### CQ-5 — No linting / type-check configuration — **Low**
- **Description:** No `pyproject.toml`, `ruff`/`mypy`/`flake8` config, or formatting standard enforced. Code is manually consistent (snake_case, verb-first functions, UPPER_SNAKE constants) but nothing enforces it.
- **Affected:** repository root
- **Remediation:** Add `pyproject.toml` with `ruff` (lint + format) and run it in CI/pre-commit.

---

## 3. Security Vulnerabilities

### SEC-1 — Checkpoint loading — **Low (mitigated)**
- **Description:** Only one `torch.load` call exists in `src/` (`src/eval/evaluate_model.py:105`), and it correctly passes **`weights_only=True`**, which blocks pickle-based RCE from tampered checkpoints. `src/training/train_model.py:223` is a `torch.save` (outbound). Notebooks load checkpoints through `load_checkpoint`, so they inherit the safe path.
- **Affected:** `src/eval/evaluate_model.py`; audit notebooks' ad-hoc `torch.load` if any are added
- **Remediation:** Keep `weights_only=True` everywhere; if any notebook calls `torch.load` directly, route it through `load_checkpoint`. No known vulnerable `eval(`/`exec(`, `subprocess`, `os.system`, or `shell=True` usage (verified by grep).

### SEC-2 — Secrets / environment handling — **Info (good)**
- **Description:** No hardcoded credentials, no secrets committed. `.env` is gitignored. No network-exposed service.
- **Remediation:** Continue the policy; never commit `.env` or database files (already ignored).

### SEC-3 — Third-party / untrusted checkpoints — **Low**
- **Description:** The project loads pretrained checkpoints from `experiments/checkpoints/`. If any come from an untrusted source, `weights_only=True` protects state-dict loads, but full-model `torch.save`/`load` paths (none currently) would not.
- **Remediation:** Standardize on state-dict checkpoints + `weights_only=True`; document that checkpoints are trusted files.

---

## 4. Dependency Health

### DEP-1 — Unpinned dependencies — **High**
- **Description:** `requirements.txt` lists bare package names (`torch`, `torchvision`, `pandas`, …) with no version constraints. The active venv resolves to bleeding-edge versions (torch 2.13.0+cu130, pandas 3.0.5, Python 3.14). Any reinstall is non-reproducible and a supply-chain risk.
- **Affected:** `requirements.txt`
- **Remediation:** Pin exact versions (`pip freeze > requirements.lock`) or use constrained ranges; consider `requirements-dev.txt` for tooling; document the supported Python (3.12/3.13 vs 3.14).

### DEP-2 — Declared but not installed — **High**
- **Description:** `pytest` and `opencv-python` are in `requirements.txt` but **missing from the active `.venv`**. The entire test suite is unrunnable, and `cv2` is referenced in 5 notebooks (`practice_2*.ipynb`).
- **Affected:** `requirements.txt`, `.venv`
- **Remediation:** Install `pytest` (and `opencv-python` if any notebook cell actually imports it at runtime); otherwise prune it from requirements.

### DEP-3 — Unused declared dependencies — **Low**
- **Description:** `tabulate` and `tqdm` are declared but not referenced in `src/`, `tests/`, or `notebooks/`.
- **Affected:** `requirements.txt`
- **Remediation:** Remove or actually use them.

### DEP-4 — Bleeding-edge runtime — **Low**
- **Description:** Python 3.14 + torch 2.13 + pandas 3.0 are very new major versions. The Python-3.14 DataLoader `BrokenPipeError` (documented as BUG-01) was a real instability; pandas 3.0 changes default behaviors.
- **Affected:** `.venv`, `requirements.txt`
- **Remediation:** Pin to a known-good tested matrix (e.g., Python 3.12/3.13, torch 2.5–2.6) or document the 3.14 constraints explicitly.

---

## 5. Architecture Consistency

### ARC-1 — Data-root inconsistency — **Medium**
- **Description:** The CIFAR-10 dataset physically lives at `data/raw`, but three modules and the config still reference `data/external/CIFAR-10`: `src/data/dataset.py`, `src/data/load_cifar10.py`, and `configs/data.yaml`. This is the root cause of the re-download bug fixed in `dataloader.py` this week and will resurface wherever the old path is used.
- **Affected:** `src/data/dataset.py`, `src/data/load_cifar10.py`, `configs/data.yaml`, `src/scratch/build_notebook.py`
- **Remediation:** Define one `DATA_ROOT` (recommend `data/raw`) in a single shared module (e.g., `src/data/config.py`), make every loader and the config consume it, and add a regression test that asserts the root resolves and is present.

### ARC-2 — Duplicate loader APIs with divergent imports — **Medium**
- **Description:** `dataloader.py` and `load_cifar10.py` both expose `get_cifar10_loaders`; notebooks and scripts import from different places (`src.data.dataloader`, `src.data.load_cifar10`, one even `data.dataloader`). This causes silent behavioral differences (data root, transforms).
- **Affected:** `src/data/*`, `notebooks/*`, `src/experiments/*`
- **Remediation:** Keep one canonical loader; deprecate the other; sweep imports to the canonical one; add an import test.

### ARC-3 — Agent docs reference stale layout — **Low**
- **Description:** Several `agents/` and `docs/` files reference `data/external/CIFAR-10` and pre-fix notebook structure (e.g., `agents/phases/*`, `docs/overview.md`), diverging from the current code.
- **Affected:** `agents/`, `docs/`
- **Remediation:** Sweep docs for the stale data path and notebook section references.

### ARC-4 — Experiment scripts duplicate evaluation logic — **Low**
- **Description:** `src/experiments/benchmark_sota.py`, `catdog_confusion_reduction.py`, and `feature_level_tta.py` re-implement metrics/confusion-matrix helpers instead of importing from `src/eval/evaluate_model.py`.
- **Affected:** `src/experiments/*`
- **Remediation:** Reuse `src/eval` helpers; keep experiments thin orchestrators.

---

## 6. Test Coverage

### TST-1 — Test suite cannot run — **High**
- **Description:** `pytest` is not installed in the venv, so `tests/` is dead code. No `pyproject.toml`/`pytest.ini`.
- **Affected:** `tests/`, `.venv`, `requirements.txt`
- **Remediation:** Install `pytest`, add `pytest.ini`, and run `pytest` as a documented step.

### TST-2 — Minimal coverage of hot paths — **Medium**
- **Description:** Only 4 test files exist (`dataloader`, `dataset`, `statistics`, `transforms`). `test_dataloader.py` tests only `get_single_loader` — **not** `get_cifar10_loaders`, the persisted split, or the download gating (the areas that actually broke). There are no tests for `build_model.py` modes, `train_model.py`, `evaluate_model.py`, or `src/data/config.py`.
- **Affected:** `tests/`
- **Remediation:** Add tests for split persistence (`_ensure_split`), `get_cifar10_loaders` (root + no-redownload), `build_resnet18/build_densenet121` freeze modes, `evaluate`/`per_class_accuracy`, and `train_model` (1-epoch smoke).

### TST-3 — No CI / smoke-test enforcement — **Medium**
- **Description:** A smoke-test checklist template exists (`agents/templates/SMOKE_TEST_CHECKLIST.md`) and the rules mandate smoke tests, but there is no CI workflow (`.github/` absent) and no automated runner enforcing it.
- **Affected:** repository root
- **Remediation:** Add a GitHub Actions workflow (or pre-commit) that runs `pytest` and a notebook-smoke script on every PR.

### TST-4 — Tests depend on local data/network — **Low**
- **Description:** Data tests use the real `data/raw` CIFAR-10 and may trigger downloads on first run, making them slow and environment-dependent.
- **Affected:** `tests/test_dataloader.py`, `tests/test_dataset.py`
- **Remediation:** Use small synthetic datasets / fixtures for unit tests; keep a separate integration test for real data.

---

## 7. Performance Bottlenecks

### PERF-1 — `num_workers=0` everywhere — **Medium**
- **Description:** All loaders default to `num_workers=0` (the Python-3.14 BrokenPipe workaround, BUG-01), so data loading is single-threaded. On GPU training with 224×224 CIFAR-10 this can starve the GPU.
- **Affected:** `src/data/dataloader.py`, `notebooks/*`
- **Remediation:** Re-enable workers with a spawn-safe pattern, or document the intentional single-threaded trade-off and profile whether loading is the bottleneck.

### PERF-2 — Experiments recompute features each run — **Medium**
- **Description:** `catdog_confusion_reduction.py` and `feature_level_tta.py` re-extract frozen backbone features / run full test forward passes on every execution (~5–13 min each), with no on-disk cache.
- **Affected:** `src/experiments/catdog_confusion_reduction.py`, `src/experiments/feature_level_tta.py`
- **Remediation:** Cache extracted features/probabilities to `experiments/results/*.npy` (keyed by checkpoint hash) and reuse across runs.

### PERF-3 — Triple in-memory CIFAR-10 instances — **Low**
- **Description:** `dataloader.py` builds three full `CIFAR10` objects (train/val/test) with separate transforms (~150 MB raw each), and `load_cifar10.py` does similar.
- **Affected:** `src/data/dataloader.py`, `src/data/load_cifar10.py`
- **Remediation:** Build the raw dataset once and apply transforms via `Subset`/mapping; acceptable if memory is not a constraint.

### PERF-4 — Notebooks not fully partial-run safe — **Low**
- **Description:** `practice_2.ipynb` has `FORCE_RETRAIN = True`, so re-running retrains for hours; notebook rules (output persistence + cell independence) were added but not yet fully applied.
- **Affected:** `notebooks/practice_2.ipynb`
- **Remediation:** Gate training on checkpoint existence (load-if-present, train-if-missing) and persist metrics to `experiments/results/` per the new rules.

### PERF-5 — Model sizes / checkpoints — **Info**
- **Description:** ResNet18 checkpoint ~43 MB, DenseNet121 ~27 MB; measured latency (batch 64, CUDA): ResNet18 ≈ 39 ms, DenseNet121 ≈ 135 ms, ensemble ≈ 175 ms (profiling added to `practice_2.ipynb`). DenseNet121 is ~3.4× slower than ResNet18 despite fewer params.
- **Remediation:** No action needed; documented for deployment decisions.

---

## 8. Overall Project Health

| Dimension | Rating | Notes |
|---|---|---|
| Structure & conventions | **Strong** | Clean `src/` package layout, naming conventions, `agents/` documentation discipline |
| Code quality | **Good** | Well-documented modules; some duplication and drift |
| Security | **Good** | `weights_only=True`, no shell/secrets issues |
| Dependency health | **Weak** | Unpinned, missing declared deps, bleeding-edge stack |
| Architecture consistency | **Fair** | Data-root inconsistency is the main defect |
| Test coverage | **Weak** | Minimal coverage; suite currently unrunnable |
| Performance | **Fair** | Correct but single-threaded loading; recompute-heavy experiments |

**One-line health:** A well-organized, well-documented research codebase whose reproducibility and automated verification are the weak points.

---

## 9. Prioritized Action Plan

### P0 — Fix now (blocks trust/reproducibility)
1. **Pin dependencies** — `pip freeze > requirements.lock` (or constrained ranges in `requirements.txt`); document the tested Python version. *(DEP-1)*
2. **Restore test executability** — install `pytest` (+ `opencv-python` if needed or prune), add `pytest.ini`. *(DEP-2, TST-1)*
3. **Unify the data root** — single `DATA_ROOT` (recommend `data/raw`) consumed by `dataset.py`, `load_cifar10.py`, `configs/data.yaml`; add a regression test that the root resolves. *(ARC-1, CQ-1, ARC-2)*

### P1 — Next iteration (raises confidence)
4. **Expand tests** — split persistence/download gating, model builder modes, train/eval smokes. *(TST-2)*
5. **Add CI** — GitHub Actions running `pytest` + a notebook-smoke script on every PR. *(TST-3)*
6. **Cache experiment features** to disk (`.npy`) so re-runs are minutes, not tens of minutes. *(PERF-2)*
7. **Move SOTA full-unfreeze builders into `src/models/build_model.py`** and deduplicate across notebooks/scripts. *(CQ-4)*

### P2 — Polish (when time permits)
8. Re-enable DataLoader workers with a spawn-safe pattern (or document the `num_workers=0` trade-off). *(PERF-1)*
9. Add `pyproject.toml` with `ruff`/`mypy`; enforce in CI. *(CQ-5)*
10. Reconcile or retire `scratch/build_notebook.py`; sweep docs for stale paths. *(CQ-3, ARC-3)*
11. Make `practice_2.ipynb` fully partial-run safe (checkpoint-gated training, persisted metrics). *(PERF-4)*
12. Remove unused deps (`tabulate`, `tqdm`) and prune dead loader paths. *(DEP-3, CQ-1)*
