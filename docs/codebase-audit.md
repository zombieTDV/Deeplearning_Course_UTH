# Codebase Audit Report — LAB2 (CIFAR-10 Transfer Learning)

- **Motivation/Background**: LAB2 is the main CIFAR-10 transfer-learning deliverable (pretrained ResNet18/DenseNet121, SOTA recipe, ensemble, TTA); a baseline audit is needed to track engineering maturity across branches. 1–3 sentences.
- **Purpose**: Establish a baseline of code quality, security, dependency health, architecture consistency, test coverage, and performance for the `LAB2` branch.
- **Overview Pipeline**: Inspected the git tree and `src/`, `tests/`, `notebooks/`, `configs/`, `agents/`; ran static greps for security markers; compared `requirements.txt` against the active `.venv`; assessed compliance against `agents/rules/*`.
- **Detailed Plan**: Executive Summary; Findings Summary; per-area findings (code quality, security, dependencies, architecture, tests, performance); Compliance; Risk Analysis; Overall Health; Prioritized Action Plan.
- **References**: `git`, `grep`, `importlib.metadata`/`pip`, `agents/rules/*`, `configs/data.yaml`.

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Findings Summary](#2-findings-summary)
- [3. Code Quality](#3-code-quality)
- [4. Security Vulnerabilities](#4-security-vulnerabilities)
- [5. Dependency Health](#5-dependency-health)
- [6. Architecture Consistency](#6-architecture-consistency)
- [7. Test Coverage](#7-test-coverage)
- [8. Performance Bottlenecks](#8-performance-bottlenecks)
- [9. Compliance with Policies and Procedures](#9-compliance-with-policies-and-procedures)
- [10. Detailed Risk Analysis](#10-detailed-risk-analysis)
- [11. Overall Project Health](#11-overall-project-health)
- [12. Prioritized Action Plan](#12-prioritized-action-plan)

---

## 1. Executive Summary

> **Scope:** branch `LAB2`, HEAD `96820be`; date 2026-08-08; method — static + targeted runtime inspection (dependency matrix, security grep, tree audit).

The codebase is **well-structured and documented**: a clean `src/` package layout, disciplined experiment tracking under `agents/`, consistent naming, safe checkpoint loading, and a clean 16-commit history.

What blocks maturity:
1. **Dependencies are unpinned and the test suite cannot run** (`pytest` and `opencv-python` declared but not installed).
2. **No single source of truth for the CIFAR-10 data root** (`data/raw` vs `data/external/CIFAR-10` across modules + config).
3. **Test coverage is minimal and there is no CI gate.**

Overall health: **Good structure and process; low reproducibility and weak automated verification** (see [§11](#11-overall-project-health), top actions in [§12](#12-prioritized-action-plan)).

---

## 2. Findings Summary

| ID | Area | Severity | Title | Section |
|---|---|---|---|---|
| DEP-1 | Dependencies | **High** | Unpinned `requirements.txt` | [5. Dependency Health](#5-dependency-health) |
| DEP-2 | Dependencies | **High** | `pytest` + `opencv-python` declared but not installed | [5. Dependency Health](#5-dependency-health) |
| TST-1 | Testing | **High** | Test suite cannot run; no CI gate | [7. Test Coverage](#7-test-coverage) |
| ARC-1 | Architecture | **Medium** | Data-root inconsistency | [6. Architecture Consistency](#6-architecture-consistency) |
| CQ-1 | Code quality | **Medium** | Duplicate overlapping data-loading modules | [3. Code Quality](#3-code-quality) |
| PERF-1 | Performance | **Medium** | `num_workers=0` everywhere | [8. Performance Bottlenecks](#8-performance-bottlenecks) |
| PERF-2 | Performance | **Medium** | Experiments recompute features each run | [8. Performance Bottlenecks](#8-performance-bottlenecks) |
| TST-2 | Testing | **Medium** | Minimal coverage; hot paths untested | [7. Test Coverage](#7-test-coverage) |
| TST-3 | Testing | **Medium** | No CI and smoke-test enforcement | [7. Test Coverage](#7-test-coverage) |
| CQ-2 | Code quality | **Low** | `configs/data.yaml` not consumed by loaders | [3. Code Quality](#3-code-quality) |
| CQ-4 | Code quality | **Low** | Duplicate SOTA model-building logic | [3. Code Quality](#3-code-quality) |
| DEP-3 | Dependencies | **Low** | `tabulate`, `tqdm` declared but unused | [5. Dependency Health](#5-dependency-health) |
| DEP-4 | Dependencies | **Low** | Bleeding-edge runtime | [5. Dependency Health](#5-dependency-health) |
| CQ-3 | Code quality | **Low** | `scratch/build_notebook.py` drifts | [3. Code Quality](#3-code-quality) |
| ARC-3 | Architecture | **Low** | Agents/docs reference stale data location | [6. Architecture Consistency](#6-architecture-consistency) |
| SEC-1 | Security | **Low** | Checkpoint loading — mitigated | [4. Security Vulnerabilities](#4-security-vulnerabilities) |
| TST-4 | Testing | **Low** | Tests depend on local data/network | [7. Test Coverage](#7-test-coverage) |
| PERF-3 | Performance | **Low** | Triple in-memory CIFAR-10 instances | [8. Performance Bottlenecks](#8-performance-bottlenecks) |
| PERF-4 | Performance | **Low** | Notebooks not fully partial-run safe | [8. Performance Bottlenecks](#8-performance-bottlenecks) |
| CQ-5 | Code quality | **Low** | No linting / type-check config | [3. Code Quality](#3-code-quality) |

Severity totals feed the [risk analysis §10](#10-detailed-risk-analysis); policy mapping is in [§9](#9-compliance-with-policies-and-procedures).

---

## 3. Code Quality

### CQ-1: Duplicate overlapping data-loading modules
- **Severity:** Medium
- **Description:** CIFAR-10 loading is implemented three times with different roots/APIs: `src/data/dataloader.py` (`data/raw`), `src/data/load_cifar10.py` (`data/external/CIFAR-10`), `src/data/dataset.py` (`data/external/CIFAR-10`). Notebooks import from different modules; one used the stale `from data.dataloader import ...`.
- **Affected:** `src/data/dataloader.py`, `src/data/load_cifar10.py`, `src/data/dataset.py`
- **Remediation:** Pick one canonical loader, route the others through it, delete dead code, add a single-root test — tracked in [Action P1.3](#12-prioritized-action-plan).

### CQ-2: Config file not consumed by code
- **Severity:** Low
- **Description:** `configs/data.yaml` (dataset root, `num_workers: 2`, batch 64) is detailed but loaders hardcode defaults (`num_workers=0`), so editing YAML silently does nothing.
- **Affected:** `configs/data.yaml`, `src/data/config.py`, `src/data/dataloader.py`
- **Remediation:** Load defaults from `data.yaml` or delete it until wired — [Action P2.3](#12-prioritized-action-plan).

### CQ-3: Notebook generator drift
- **Severity:** Low
- **Description:** `src/scratch/build_notebook.py` (1131 lines) regenerates notebooks but hardcodes the stale `data/external/CIFAR-10` root and no longer matches the checked-in notebooks.
- **Affected:** `src/scratch/build_notebook.py`, `notebooks/practice_2.ipynb`
- **Remediation:** Retire the generator or sync it to the current notebooks — [Action P2.2](#12-prioritized-action-plan).

### CQ-4: Duplicate SOTA model-building logic
- **Severity:** Low
- **Description:** EXP-07 "full SOTA" unfreezing is re-implemented in `notebooks/practice_2.ipynb` and `src/experiments/exp_07_*.py` instead of living once in `src/models/build_model.py`.
- **Affected:** `notebooks/practice_2.ipynb`, `src/experiments/exp_07_resnet_densenet_sota.py`
- **Remediation:** Add `build_resnet18_full_sota`/`build_densenet121_full_sota` to `src/models` and import them — [Action P1.4](#12-prioritized-action-plan).

### CQ-5: No linting or type-check configuration
- **Severity:** Low
- **Description:** No `pyproject.toml`, `ruff`/`mypy` config; conventions are manual.
- **Affected:** repository root
- **Remediation:** Add `pyproject.toml` + `ruff`, enforce in CI — [Action P2.1](#12-prioritized-action-plan).

---

## 4. Security Vulnerabilities

### SEC-1: Checkpoint loading (mitigated)
- **Severity:** Low
- **Description:** The only `torch.load` in `src/` (`src/eval/evaluate_model.py:105`) correctly passes **`weights_only=True`** (blocks pickle RCE). `src/training/train_model.py:223` is `torch.save` (outbound). No `eval(`/`exec(`, `subprocess`, `os.system`, or `shell=True` usage (grep-verified). No hardcoded secrets; `.env` is gitignored.
- **Affected:** `src/eval/evaluate_model.py`; notebooks via `load_checkpoint`
- **Remediation:** Keep `weights_only=True` everywhere; route any notebook `torch.load` through `load_checkpoint` — [Action P0.1](#12-prioritized-action-plan).

---

## 5. Dependency Health

### DEP-1: Unpinned dependencies
- **Severity:** High
- **Description:** `requirements.txt` lists bare names; the venv resolves to bleeding-edge versions (torch 2.13.0+cu130, pandas 3.0.5, Python 3.14). Reinstalls are non-reproducible.
- **Affected:** `requirements.txt`
- **Remediation:** Pin exact versions (`pip freeze > requirements.lock`) — [Action P0.2](#12-prioritized-action-plan).

### DEP-2: Declared but not installed
- **Severity:** High
- **Description:** `pytest` and `opencv-python` are declared but missing from the venv — tests are unrunnable and `cv2` is referenced in 5 notebooks.
- **Affected:** `requirements.txt`, `.venv`
- **Remediation:** Install `pytest` (+`opencv-python` if used at runtime) or prune — [Action P0.2](#12-prioritized-action-plan).

### DEP-3: Unused declared dependencies
- **Severity:** Low
- **Description:** `tabulate` and `tqdm` are declared but unreferenced in `src/`, `tests/`, `notebooks/`.
- **Affected:** `requirements.txt`
- **Remediation:** Remove or use them — [Action P2.3](#12-prioritized-action-plan).

### DEP-4: Bleeding-edge runtime
- **Severity:** Low
- **Description:** Python 3.14 + torch 2.13 + pandas 3.0 are very new majors; BUG-01 (DataLoader `BrokenPipeError`) was a real 3.14 instability.
- **Affected:** `.venv`, `requirements.txt`
- **Remediation:** Pin a known-good matrix or document 3.14 constraints — [Action P0.2](#12-prioritized-action-plan).

---

## 6. Architecture Consistency

### ARC-1: Data-root inconsistency
- **Severity:** Medium
- **Description:** Data lives at `data/raw`, but `src/data/dataset.py`, `src/data/load_cifar10.py`, and `configs/data.yaml` still reference `data/external/CIFAR-10` — the root cause of the re-download bug fixed in `dataloader.py`.
- **Affected:** `src/data/dataset.py`, `src/data/load_cifar10.py`, `configs/data.yaml`, `src/scratch/build_notebook.py`
- **Remediation:** Define one `DATA_ROOT` in `src/data/config.py` and have every loader consume it; add a regression test — [Action P0.3](#12-prioritized-action-plan).

### ARC-2: Duplicate loader APIs with divergent imports
- **Severity:** Medium
- **Description:** `dataloader.py` and `load_cifar10.py` both expose `get_cifar10_loaders`; notebooks import from different modules with behavioral differences.
- **Affected:** `src/data/*`, `notebooks/*`, `src/experiments/*`
- **Remediation:** Keep one canonical loader; sweep imports — [Action P1.3](#12-prioritized-action-plan).

### ARC-3: Agent docs reference stale layout
- **Severity:** Low
- **Description:** Several `agents/` and `docs/` files reference `data/external/CIFAR-10` and pre-fix notebook structure.
- **Affected:** `agents/`, `docs/`
- **Remediation:** Sweep docs for stale paths — [Action P2.2](#12-prioritized-action-plan).

---

## 7. Test Coverage

### TST-1: Test suite cannot run
- **Severity:** High
- **Description:** `pytest` is not installed; `tests/` is dead code; no `pytest.ini`.
- **Affected:** `tests/`, `.venv`, `requirements.txt`
- **Remediation:** Install `pytest`, add `pytest.ini` — [Action P0.2](#12-prioritized-action-plan).

### TST-2: Minimal coverage of hot paths
- **Severity:** Medium
- **Description:** Only 4 test files; `test_dataloader.py` tests only `get_single_loader`, not `get_cifar10_loaders`/split/download gating. No tests for `build_model`, `train_model`, `evaluate_model`, `config`.
- **Affected:** `tests/`
- **Remediation:** Add tests for split persistence, loaders, model modes, train/eval smokes — [Action P1.1](#12-prioritized-action-plan).

### TST-3: No CI and smoke-test enforcement
- **Severity:** Medium
- **Description:** Smoke-test checklist template exists but there is no CI workflow or automated runner.
- **Affected:** repository root
- **Remediation:** Add a CI workflow running `pytest` + notebook smoke — [Action P1.2](#12-prioritized-action-plan).

### TST-4: Tests depend on local data/network
- **Severity:** Low
- **Description:** Data tests use the real `data/raw` and may download.
- **Affected:** `tests/test_dataloader.py`, `tests/test_dataset.py`
- **Remediation:** Use synthetic fixtures for unit tests — [Action P1.1](#12-prioritized-action-plan).

---

## 8. Performance Bottlenecks

### PERF-1: `num_workers=0` everywhere
- **Severity:** Medium
- **Description:** All loaders default to `num_workers=0` (Python-3.14 workaround), single-threaded loading; can starve GPU at 224×224.
- **Affected:** `src/data/dataloader.py`, `notebooks/*`
- **Remediation:** Re-enable workers safely or document the trade-off — [Action P2.4](#12-prioritized-action-plan).

### PERF-2: Experiments recompute features each run
- **Severity:** Medium
- **Description:** `catdog_confusion_reduction.py` and `feature_level_tta.py` re-extract features every run (~5–13 min), with no cache.
- **Affected:** `src/experiments/catdog_confusion_reduction.py`, `src/experiments/feature_level_tta.py`
- **Remediation:** Cache features to `.npy` keyed by checkpoint hash — [Action P1.5](#12-prioritized-action-plan).

### PERF-3: Triple in-memory CIFAR-10 instances
- **Severity:** Low
- **Description:** Loaders build three full `CIFAR10` objects with separate transforms.
- **Affected:** `src/data/dataloader.py`, `src/data/load_cifar10.py`
- **Remediation:** Build raw dataset once, apply transforms via `Subset` — [Action P2.4](#12-prioritized-action-plan).

### PERF-4: Notebooks not fully partial-run safe
- **Severity:** Low
- **Description:** `practice_2.ipynb` has `FORCE_RETRAIN=True`; re-running retrains for hours.
- **Affected:** `notebooks/practice_2.ipynb`
- **Remediation:** Gate training on checkpoint existence; persist metrics — [Action P1.6](#12-prioritized-action-plan).

---

## 9. Compliance with Policies and Procedures

Assessed against `agents/rules/*` (naming, folder structure, MD convention, notebook header, agent rules).

| Policy / procedure | Compliance | Evidence / gap | Related finding |
|---|---|---|---|
| NAMING_CONVENTION | **Partial** | snake_case/verb-first compliant; some notebooks miss `NN_` prefix | [CQ-4](#cq-4-duplicate-sota-model-building-logic) |
| FOLDER_STRUCTURE | **Partial** | Top-level layout matches; several undocumented files (`docs/md_convention.md`, root `purpose.md`) | [ARC-3](#arc-3-agent-docs-reference-stale-layout) |
| MD_CONVENTION | **Partial** | New docs have header + TOC; older docs partial | [ARC-3](#arc-3-agent-docs-reference-stale-layout) |
| NOTEBOOK_HEADER_CONVENTION | **Partial** | Headers present; output-persistence & cell-independence rules added but not fully applied | [PERF-4](#perf-4-notebooks-not-fully-partial-run-safe) |
| AGENT_AI (docs/logging) | **Compliant** | `agents/` phase/status/experiment docs maintained | — |

Cross-reference: non-compliance maps to [risk §10](#10-detailed-risk-analysis) and [action §12](#12-prioritized-action-plan).

---

## 10. Detailed Risk Analysis

| Risk | Likelihood | Impact | Overall | Description & mitigation | Related finding |
|---|---|---|---|---|---|
| Reproducibility failure | High | High | **High** | Unpinned deps; pin `requirements.lock` | [DEP-1](#dep-1-unpinned-dependencies) |
| Tests unrunnable / regressions | High | High | **High** | `pytest` missing; install + CI | [DEP-2](#dep-2-declared-but-not-installed), [TST-1](#tst-1-test-suite-cannot-run) |
| Re-download / wrong data root | Medium | Medium | **Medium** | Stale `data/external` paths; unify root | [ARC-1](#arc-1-data-root-inconsistency) |
| GPU starvation (loading) | Medium | Low | **Low** | `num_workers=0`; re-enable safely | [PERF-1](#perf-1-num_workers0-everywhere) |
| Slow experiment iterations | Medium | Low | **Low** | Feature recompute; cache `.npy` | [PERF-2](#perf-2-experiments-recompute-features-each-run) |
| Pickle RCE from checkpoints | Low | High | **Low** | `weights_only=True` already used | [SEC-1](#sec-1-checkpoint-loading-mitigated) |
| Policy exposure | Medium | Low | **Low** | Partial rule compliance; remediate per plan | [CQ-5](#cq-5-no-linting-or-type-check-configuration) |

---

## 11. Overall Project Health

| Dimension | Rating | Notes |
|---|---|---|
| Structure & conventions | **Strong** | Clean `src/` packages, naming, `agents/` docs |
| Code quality | **Good** | Documented modules; some duplication |
| Security | **Good** | `weights_only=True`; no shell/secrets issues |
| Dependency health | **Weak** | Unpinned, missing deps, bleeding-edge stack |
| Architecture consistency | **Fair** | Data-root inconsistency |
| Test coverage | **Weak** | Minimal; suite unrunnable |
| Performance | **Fair** | Single-threaded loading; recompute-heavy |

Evidence for each dimension is in [§3](#3-code-quality)–[§8](#8-performance-bottlenecks).

---

## 12. Prioritized Action Plan

### P0 — Fix now (blocks trust / reproducibility / security)
- **P0.1** Keep `weights_only=True` on all `torch.load`; route notebook loads through `load_checkpoint` — addresses [SEC-1](#sec-1-checkpoint-loading-mitigated).
- **P0.2** Pin `requirements.lock`; install `pytest` (+`opencv-python` or prune); add `pytest.ini` — addresses [DEP-1](#dep-1-unpinned-dependencies), [DEP-2](#dep-2-declared-but-not-installed), [TST-1](#tst-1-test-suite-cannot-run), [DEP-4](#dep-4-bleeding-edge-runtime).
- **P0.3** Unify the data root under one `DATA_ROOT` (`data/raw`) across `dataset.py`, `load_cifar10.py`, `configs/data.yaml`; add a regression test — addresses [ARC-1](#arc-1-data-root-inconsistency).

### P1 — Next iteration (raises confidence)
- **P1.1** Add tests: split persistence, download gating, model modes, train/eval smokes — [TST-2](#tst-2-minimal-coverage-of-hot-paths), [TST-4](#tst-4-tests-depend-on-local-datanetwork).
- **P1.2** Add CI (GitHub Actions: `pytest` + notebook smoke) — [TST-3](#tst-3-no-ci-and-smoke-test-enforcement).
- **P1.3** Canonicalize the loader; sweep imports — [CQ-1](#cq-1-duplicate-overlapping-data-loading-modules), [ARC-2](#arc-2-duplicate-loader-apis-with-divergent-imports).
- **P1.4** Move SOTA builders into `src/models/build_model.py` — [CQ-4](#cq-4-duplicate-sota-model-building-logic).
- **P1.5** Cache experiment features to `.npy` — [PERF-2](#perf-2-experiments-recompute-features-each-run).
- **P1.6** Make `practice_2.ipynb` partial-run safe (checkpoint-gated training, persisted metrics) — [PERF-4](#perf-4-notebooks-not-fully-partial-run-safe).

### P2 — Polish (when time permits)
- **P2.1** Add `pyproject.toml` + `ruff`/`mypy` — [CQ-5](#cq-5-no-linting-or-type-check-configuration).
- **P2.2** Sync/retire `scratch/build_notebook.py`; sweep docs for stale paths — [CQ-3](#cq-3-notebook-generator-drift), [ARC-3](#arc-3-agent-docs-reference-stale-layout).
- **P2.3** Wire or remove `configs/data.yaml`; prune unused deps — [CQ-2](#cq-2-config-file-not-consumed-by-code), [DEP-3](#dep-3-unused-declared-dependencies).
- **P2.4** Re-enable DataLoader workers; reduce dataset duplication — [PERF-1](#perf-1-num_workers0-everywhere), [PERF-3](#perf-3-triple-in-memory-cifar-10-instances).

Each item links back to its finding; the summary table is in [§2](#2-findings-summary).
