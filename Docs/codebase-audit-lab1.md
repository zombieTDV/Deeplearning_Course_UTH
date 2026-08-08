# Codebase Audit Report — LAB1 (FashionMNIST Classification)

- **Motivation/Background**: LAB1 is the FashionMNIST practice deliverable (MLP/CNN, phase-based error analysis); a baseline audit is needed to compare engineering maturity against LAB2. 1–3 sentences.
- **Purpose**: Establish a baseline of code quality, security, dependency health, architecture consistency, test coverage, and performance for the `LAB1` branch, plus compliance with its own rulebase.
- **Overview Pipeline**: Audited the branch via git (working tree untouched): file tree, `src/`, notebooks, `requirements.txt`, `.gitignore`, tracked artifacts, and the project rulebase (`Docs/Rulebase.md`, `agents/rules.md`).
- **Detailed Plan**: Executive Summary; Findings Summary; per-area findings (code quality, security, dependencies, architecture, tests, performance); Compliance; Risk Analysis; Overall Health; Prioritized Action Plan.
- **References**: `git`, `grep`, `Docs/Rulebase.md`, `agents/rules.md`.

> **Note on the reference document:** the Google Docs link in the request was a placeholder (no URL), and Google Docs/Drive is not accessible from this environment. This report therefore follows the structure and terminology of the local reference audit (`docs/codebase-audit.md`) and is saved locally as `docs/codebase-audit-lab1.md` for upload to the Drive folder.

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

> **Scope:** branch `LAB1-FashionMNIST-Classification/main`, HEAD `9d99b07`; date 2026-08-08; method — static audit via git (no working-tree changes).

LAB1 is a **well-documented experimental workflow** for FashionMNIST: phase-based notebooks (14 CNN + 4 MLP), comprehensive changelogs, a team-contribution doc, and an explicit AI-assisted development rulebase. Its strengths are **documentation and experiment traceability**.

What blocks maturity:
1. **No tests exist** — `pytest` is absent and there is no `tests/` directory (Rulebase §9 non-compliant).
2. **Unsafe checkpoint loading** — `src/model_utils.py` calls `torch.load` without `weights_only=True`, and **14 `.pth` weights + 2 Optuna `.db` are tracked in git** (Rulebase §12).
3. **Unpinned dependencies** — `requirements.txt` lists bare names.

Overall health: **Good documentation and process; weak automated verification, dependency hygiene, and security hardening** (see [§11](#11-overall-project-health), top actions in [§12](#12-prioritized-action-plan)).

---

## 2. Findings Summary

| ID | Area | Severity | Title | Section |
|---|---|---|---|---|
| SEC-1 | Security | **High** | `torch.load` without `weights_only=True` | [4. Security Vulnerabilities](#4-security-vulnerabilities) |
| DEP-1 | Dependencies | **High** | Unpinned `requirements.txt` | [5. Dependency Health](#5-dependency-health) |
| TST-1 | Testing | **High** | No tests; `pytest` not declared | [7. Test Coverage](#7-test-coverage) |
| ARC-2 | Architecture | **Medium** | `.pth` / `.db` / PNG tracked in git | [6. Architecture Consistency](#6-architecture-consistency) |
| CQ-1 | Code quality | **Medium** | Hardcoded relative paths | [3. Code Quality](#3-code-quality) |
| CQ-2 | Code quality | **Medium** | No validation / early stopping in train loop | [3. Code Quality](#3-code-quality) |
| TST-2 | Testing | **Medium** | No CI and smoke-test enforcement | [7. Test Coverage](#7-test-coverage) |
| ARC-1 | Architecture | **Medium** | Flat `src/` + duplicated notebook logic | [6. Architecture Consistency](#6-architecture-consistency) |
| DEP-2 | Dependencies | **Low** | Bleeding-edge runtime; no constraints | [5. Dependency Health](#5-dependency-health) |
| CQ-3 | Code quality | **Low** | No linting / type-check config | [3. Code Quality](#3-code-quality) |
| ARC-3 | Architecture | **Low** | Empty `config/` directory | [6. Architecture Consistency](#6-architecture-consistency) |
| PERF-1 | Performance | **Low** | `num_workers=0` (single-threaded) | [8. Performance Bottlenecks](#8-performance-bottlenecks) |
| PERF-2 | Performance | **Low** | Recompute-heavy notebooks | [8. Performance Bottlenecks](#8-performance-bottlenecks) |
| SEC-2 | Security | **Info** | No hardcoded credentials (good) | [4. Security Vulnerabilities](#4-security-vulnerabilities) |
| CQ-4 | Code quality | **Info** | Documentation strength | [3. Code Quality](#3-code-quality) |

Severity totals feed the [risk analysis §10](#10-detailed-risk-analysis); policy mapping is in [§9](#9-compliance-with-policies-and-procedures).

---

## 3. Code Quality

### CQ-1: Hardcoded relative paths
- **Severity:** Medium
- **Description:** All `src/` utilities write to hardcoded paths such as `'../outputs/practice_1/metrics'` (`src/eval_utils.py` `METRICS_DIR`, `src/train_utils.py`, `src/model_utils.py`), breaking when run from any directory other than `notebooks/`.
- **Affected:** `src/eval_utils.py`, `src/train_utils.py`, `src/model_utils.py`, `src/data_utils.py`
- **Remediation:** Resolve paths from a `PROJECT_ROOT` helper (as in LAB2) — tracked in [Action P1.1](#12-prioritized-action-plan).

### CQ-2: No validation tracking in the training loop
- **Severity:** Medium
- **Description:** `train_utils.train_model` records only training loss; no validation loop, early stopping, or scheduler support, so overfitting is not observable during training.
- **Affected:** `src/train_utils.py`
- **Remediation:** Add `validate()` + best-model tracking — [Action P1.2](#12-prioritized-action-plan).

### CQ-3: No linting or type-check configuration
- **Severity:** Low
- **Description:** No `pyproject.toml`, `ruff`, `mypy`, or `flake8` config.
- **Affected:** repository root
- **Remediation:** Add `pyproject.toml` + `ruff` — [Action P2.1](#12-prioritized-action-plan).

### CQ-4: Documentation strength (positive)
- **Severity:** Info
- **Description:** `CNN_Experiment_Changelog.md`, `MLP_Experiment_Changelog.md`, `Docs/PROJECT_STRUCTURE_*.md`, `Docs/LAB1_TEAM_CONTRIBUTION.md`, and per-phase headers are thorough and satisfy Rulebase §7.
- **Affected:** `Docs/`, changelogs
- **Remediation:** None — maintain.

---

## 4. Security Vulnerabilities

### SEC-1: Unsafe checkpoint loading
- **Severity:** High
- **Description:** `src/model_utils.py:12` loads checkpoints with `torch.load(path, map_location=device)` **without `weights_only=True`** — arbitrary pickle code execution risk if a checkpoint is tampered. Exposure is amplified because **14 `.pth` files are committed to the repo** (see [ARC-2](#arc-2-tracked-binary-artifacts)).
- **Affected:** `src/model_utils.py`; notebooks calling `load_model` (`phase10_ensemble.ipynb`, `phase4_mlp_optuna_merged.ipynb`, `_optuna_mlp_hp_search.py`)
- **Remediation:** Add `weights_only=True`; stop tracking `.pth` — [Action P0.1](#12-prioritized-action-plan).

### SEC-2: Secrets / credentials
- **Severity:** Info
- **Description:** No hardcoded credentials, API keys, or secrets found in `src/` (grep-verified); no network-exposed service.
- **Affected:** repository
- **Remediation:** Continue current policy; never commit `.env` or database files.

---

## 5. Dependency Health

### DEP-1: Unpinned dependencies
- **Severity:** High
- **Description:** `requirements.txt` lists bare names; the README install targets a specific CUDA 130 build, so results are not portable.
- **Affected:** `requirements.txt`, `README.MD`
- **Remediation:** Pin exact versions (`requirements.lock`); add `requirements-dev.txt` with `pytest` — [Action P0.3](#12-prioritized-action-plan).

### DEP-2: Runtime risk
- **Severity:** Low
- **Description:** No version constraints; documented install pulls a very recent torch/CUDA build.
- **Affected:** `requirements.txt`
- **Remediation:** Same as DEP-1; record actual versions — [Action P0.3](#12-prioritized-action-plan).

---

## 6. Architecture Consistency

### ARC-1: Flat src with duplicated notebook logic
- **Severity:** Medium
- **Description:** Flat `src/{data,eval,model,train,vis}_utils.py` layout (no packages); core logic is duplicated inside 18 phase notebooks rather than imported.
- **Affected:** `src/*.py`, `notebooks/error_analysis/**/*.ipynb`
- **Remediation:** Adopt the LAB2 package layout; make notebooks thin orchestrators — [Action P2.2](#12-prioritized-action-plan).

### ARC-2: Tracked binary artifacts
- **Severity:** Medium
- **Description:** The branch tracks **14 `.pth` checkpoints, 2 Optuna `.db`, 5 `.html`, 28 `.png`** (272 files total). `.gitignore` ignores `*.pt` but **not** `*.pth`/`*.db`. Bloat, merge conflicts, accidental weight distribution, corrupt DBs.
- **Affected:** `.gitignore`, `outputs/**`
- **Remediation:** Add `*.pth`/`*.db` to `.gitignore`; `git rm --cached` the binaries — [Action P0.2](#12-prioritized-action-plan).

### ARC-3: Empty config directory
- **Severity:** Low
- **Description:** `config/` contains only `.gitkeep`.
- **Affected:** `config/`
- **Remediation:** Populate or remove — [Action P2.3](#12-prioritized-action-plan).

---

## 7. Test Coverage

### TST-1: No tests
- **Severity:** High
- **Description:** No `tests/` directory; `pytest` not declared; the overfitting-fix (commit `1e51fe4`) shipped untested.
- **Affected:** repository root, `requirements.txt`
- **Remediation:** Create `tests/` (data_utils, train_utils smoke, eval_utils, model_utils round-trip) — [Action P0.4](#12-prioritized-action-plan).

### TST-2: No CI and smoke-test enforcement
- **Severity:** Medium
- **Description:** No `.github/workflows` or pre-commit; Rulebase testing not automated.
- **Affected:** repository root
- **Remediation:** Add CI running `pytest` + notebook smoke — [Action P1.3](#12-prioritized-action-plan).

---

## 8. Performance Bottlenecks

### PERF-1: Single-threaded data loading
- **Severity:** Low
- **Description:** `get_dataloaders` and notebooks do not set `num_workers` (default 0). Low impact for FashionMNIST.
- **Affected:** `src/data_utils.py`, notebooks
- **Remediation:** Set `num_workers` explicitly — [Action P2.4](#12-prioritized-action-plan).

### PERF-2: Recompute-heavy notebooks
- **Severity:** Low
- **Description:** Each phase re-trains/re-evaluates with no result cache; low impact (tiny models).
- **Affected:** `notebooks/error_analysis/**`
- **Remediation:** Cache metrics/features to `outputs/`; load-if-present — [Action P2.4](#12-prioritized-action-plan).

---

## 9. Compliance with Policies and Procedures

Assessed against `Docs/Rulebase.md` (15 sections) and `agents/rules.md`.

| Policy / procedure | Compliance | Evidence / gap | Related finding |
|---|---|---|---|
| §1 Role Definition | **Compliant** | AI/developer responsibilities documented | — |
| §2 Transparency & AI Logging | **Compliant** | `Docs/agents_log.md` present and populated | — |
| §3 Development Workflow | **Partial** | Iterative commits; binaries committed alongside code | [ARC-2](#arc-2-tracked-binary-artifacts) |
| §4 Software Architecture Rules | **Partial** | Flat `src/`; hardcoded paths | [ARC-1](#arc-1-flat-src-with-duplicated-notebook-logic), [CQ-1](#cq-1-hardcoded-relative-paths) |
| §5 Coding Rules | **Partial** | Consistent naming; no linting | [CQ-3](#cq-3-no-linting-or-type-check-configuration) |
| §7 Documentation Standards | **Compliant** | Changelogs, headers, `Docs/` | [CQ-4](#cq-4-documentation-strength-positive) |
| §8 Report Writing Rules | **Compliant** | `LAB1_TEAM_CONTRIBUTION.md` | — |
| §9 Testing Rules | **Non-compliant** | No `tests/`, no `pytest` | [TST-1](#tst-1-no-tests) |
| §10 Version Control Rules | **Partial** | Commit volume good; tracked binaries | [ARC-2](#arc-2-tracked-binary-artifacts) |
| §11 Technical Decision Records | **Partial** | Changelogs as de-facto TDR | — |
| §12 Security Rules | **Non-compliant** | `torch.load` without `weights_only`; weights committed | [SEC-1](#sec-1-unsafe-checkpoint-loading) |
| §13 Debugging Rules | **Compliant** | Systematic `error_analysis/` phases | — |
| §15 Project Completion Checklist | **Partial** | No checklist artifact present | — |

Cross-reference: non-compliance maps to [risk §10](#10-detailed-risk-analysis) and [action §12](#12-prioritized-action-plan).

---

## 10. Detailed Risk Analysis

| Risk | Likelihood | Impact | Overall | Description & mitigation | Related finding |
|---|---|---|---|---|---|
| Pickle RCE via `torch.load` | Low | High | **Moderate** | Checkpoints are local/trusted, but weights are in repo; tampered `.pth` executes code on load. Add `weights_only=True` + untrack weights | [SEC-1](#sec-1-unsafe-checkpoint-loading) |
| Reproducibility failure | High | High | **High** | Unpinned deps + CUDA-specific install. Pin `requirements.lock` | [DEP-1](#dep-1-unpinned-dependencies) |
| Regression / correctness | High | High | **High** | No tests; overfitting fix shipped untested. Add tests + CI | [TST-1](#tst-1-no-tests) |
| Repo bloat / data governance | Medium | Medium | **Medium** | 14 `.pth`, 2 `.db`, 28 `.png` tracked. Gitignore + `git rm --cached` | [ARC-2](#arc-2-tracked-binary-artifacts) |
| Portability breakage | Medium | Medium | **Medium** | Hardcoded `../outputs/...` paths. Root-relative `Path` | [CQ-1](#cq-1-hardcoded-relative-paths) |
| Silent overfitting | Medium | Medium | **Medium** | Train loop tracks only train loss. Add val tracking | [CQ-2](#cq-2-no-validation-tracking-in-the-training-loop) |
| Policy non-compliance exposure | Medium | Medium | **Medium** | Rulebase §9/§12/§10 unmet. Remediate per plan | [TST-1](#tst-1-no-tests), [SEC-1](#sec-1-unsafe-checkpoint-loading) |

---

## 11. Overall Project Health

| Dimension | Rating | Notes |
|---|---|---|
| Documentation & traceability | **Strong** | Changelogs, phase notebooks, team doc, rulebase |
| Code quality | **Fair** | Hardcoded paths; no validation in train loop |
| Security | **Weak** | Unsafe `torch.load`; weights committed |
| Dependency health | **Weak** | Unpinned; no dev requirements |
| Architecture consistency | **Fair** | Flat utils + notebook duplication |
| Test coverage | **None** | No tests, no pytest, no CI |
| Performance | **Good enough** | Small dataset; minor improvements only |

Evidence for each dimension is in [§3](#3-code-quality)–[§8](#8-performance-bottlenecks).

---

## 12. Prioritized Action Plan

### P0 — Fix now (security, reproducibility, verifiability)
- **P0.1** Add `weights_only=True` to `src/model_utils.py` and any notebook `torch.load` — addresses [SEC-1](#sec-1-unsafe-checkpoint-loading).
- **P0.2** Add `*.pth`, `*.db`, `*.html`, `*.png` to `.gitignore`; `git rm --cached` the 14 weights + 2 DBs — addresses [ARC-2](#arc-2-tracked-binary-artifacts).
- **P0.3** Pin `requirements.lock`; add `requirements-dev.txt` with `pytest` — addresses [DEP-1](#dep-1-unpinned-dependencies), [DEP-2](#dep-2-runtime-risk).
- **P0.4** Create `tests/` (data_utils, train_utils smoke, eval_utils, model_utils round-trip) — addresses [TST-1](#tst-1-no-tests).

### P1 — Next iteration (raises confidence)
- **P1.1** Centralize `PROJECT_ROOT`; replace `'../outputs/...'` — [CQ-1](#cq-1-hardcoded-relative-paths).
- **P1.2** Add validation + early stopping to `train_model` — [CQ-2](#cq-2-no-validation-tracking-in-the-training-loop).
- **P1.3** Add CI running `pytest` + notebook smoke — [TST-2](#tst-2-no-ci-and-smoke-test-enforcement).

### P2 — Polish (when time permits)
- **P2.1** Add `pyproject.toml` + `ruff`/`mypy` — [CQ-3](#cq-3-no-linting-or-type-check-configuration).
- **P2.2** Adopt the LAB2 package layout; de-duplicate notebook logic — [ARC-1](#arc-1-flat-src-with-duplicated-notebook-logic).
- **P2.3** Populate or remove `config/` — [ARC-3](#arc-3-empty-config-directory).
- **P2.4** Set explicit `num_workers`; cache recompute-heavy results — [PERF-1](#perf-1-single-threaded-data-loading), [PERF-2](#perf-2-recompute-heavy-notebooks).

Each item links back to its finding; the summary table is in [§2](#2-findings-summary).
