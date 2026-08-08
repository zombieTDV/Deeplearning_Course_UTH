# Codebase Audit Report — LAB1 (FashionMNIST Classification)

**Project:** Deep Learning Course — Practice 1 (FashionMNIST MLP / CNN)
**Branch audited:** `LAB1-FashionMNIST-Classification/main` (HEAD `9d99b07`)
**Date:** 2026-08-08
**Method:** Static audit of the branch via git (working tree untouched): file tree, `src/` modules, notebooks, `requirements.txt`, `.gitignore`, tracked artifacts, and the project's own rulebase (`Docs/Rulebase.md`, `agents/rules.md`). Report structure mirrors the reference audit report (`docs/codebase-audit.md`) with additional Compliance and Risk Analysis sections.

> **Note on the reference document:** the Google Docs link was supplied as a placeholder (no URL), and Google Docs/Drive is not accessible from this environment. This report therefore follows the structure, terminology, and detail level of the reference audit already produced locally (`docs/codebase-audit.md`). It is saved locally as `docs/codebase-audit-lab1.md` for you to copy into the Drive folder.

---

## Executive Summary

The LAB1 branch is a **well-documented experimental workflow** for FashionMNIST: a clear phase-based notebook structure (`notebooks/error_analysis/CNN` × 14 phases, `MLP` × 4 phases), comprehensive changelogs, a team-contribution doc, and an explicit AI-assisted development rulebase. Its strengths are **documentation discipline and experiment traceability**.

However, engineering rigor is materially weaker than the LAB2 codebase:

1. **No tests exist** — `pytest` is absent from `requirements.txt` and there is no `tests/` directory (violates Rulebase §9).
2. **Unsafe checkpoint loading** — `src/model_utils.py` calls `torch.load(...)` without `weights_only=True`, and **14 `.pth` weight files + 2 Optuna `.db` databases are tracked in git** (violates Rulebase §12 and basic VCS hygiene).
3. **Unpinned dependencies** — `requirements.txt` lists bare package names (non-reproducible).

Overall health: **Good documentation and process; weak automated verification, dependency hygiene, and security hardening.** The prioritized plan (Section 12) remediates the high-severity items first.

---

## 1. Findings Summary

| ID | Area | Severity | Title |
|---|---|---|---|
| SEC-1 | Security | **High** | `torch.load` without `weights_only=True` (+ weights tracked in repo) |
| DEP-1 | Dependencies | **High** | Unpinned `requirements.txt` (non-reproducible) |
| TST-1 | Testing | **High** | No tests; `pytest` not declared |
| ARC-2 | Architecture | **Medium** | 14 `.pth` + 2 `.db` + 28 PNG tracked in git (binary bloat) |
| CQ-1 | Code quality | **Medium** | Hardcoded relative paths (`../outputs/...`) across `src/` |
| CQ-2 | Code quality | **Medium** | `train_model` tracks only train loss (no validation / early stopping) |
| TST-2 | Testing | **Medium** | No CI / smoke-test enforcement |
| ARC-1 | Architecture | **Medium** | Flat `src/utils` modules + duplicated logic across 18 notebooks |
| DEP-2 | Dependencies | **Low** | Bleeding-edge runtime (torch cu130); no version constraints |
| CQ-3 | Code quality | **Low** | No linting / type-check configuration |
| ARC-3 | Architecture | **Low** | `config/` directory empty; no centralized configuration |
| PERF-1 | Performance | **Low** | `DataLoader` default `num_workers=0` (single-threaded) |
| PERF-2 | Performance | **Low** | Notebooks recompute results each run (no caching) |
| SEC-2 | Security | **Info** | No hardcoded credentials / secrets found (good) |
| CQ-4 | Code quality | **Info** | Documentation & changelogs are a strength (Rulebase §7 compliant) |

---

## 2. Code Quality

### CQ-1 — Hardcoded relative paths — **Medium**
- **Description:** All `src/` utilities write to hardcoded relative paths such as `'../outputs/practice_1/metrics'` (`src/eval_utils.py` `METRICS_DIR`, `src/train_utils.py`, `src/model_utils.py` default path). This breaks if notebooks are run from any directory other than `notebooks/`, and mixes logic with I/O location.
- **Affected:** `src/eval_utils.py`, `src/train_utils.py`, `src/model_utils.py`, `src/data_utils.py`
- **Remediation:** Resolve paths from the project root (`Path(__file__).resolve().parents[...]` or a `PROJECT_ROOT` helper), as done in the LAB2 codebase.

### CQ-2 — No validation tracking in the training loop — **Medium**
- **Description:** `train_utils.train_model` records only training loss per epoch; there is no validation loop, early stopping, or scheduler support, so overfitting is not observable during training and "best epoch" is not tracked.
- **Affected:** `src/train_utils.py`
- **Remediation:** Add a `validate(model, val_loader, ...)` helper and best-model tracking (load-if-present semantics per the notebook rules), mirroring LAB2's `train_model`.

### CQ-3 — No linting / type-check configuration — **Low**
- **Description:** No `pyproject.toml`, `ruff`, `mypy`, or `flake8` configuration exists on the branch.
- **Affected:** repository root
- **Remediation:** Add `pyproject.toml` with `ruff` and enforce in CI.

### CQ-4 — Documentation strength — **Info (positive)**
- **Description:** `CNN_Experiment_Changelog.md`, `MLP_Experiment_Changelog.md`, `Docs/PROJECT_STRUCTURE_*.md`, `Docs/LAB1_TEAM_CONTRIBUTION.md`, and per-phase notebook headers are thorough and satisfy the Rulebase documentation standard.
- **Remediation:** None — maintain.

---

## 3. Security Vulnerabilities

### SEC-1 — Unsafe checkpoint loading — **High**
- **Description:** `src/model_utils.py:12` loads checkpoints with `torch.load(path, map_location=device)` **without `weights_only=True`**, which permits arbitrary pickle-based code execution if a checkpoint is tampered or comes from an untrusted source. Exposure is amplified because **14 `.pth` weight files are committed to the repository** (see ARC-2), so anyone with repo access controls files that will later be `torch.load`-ed.
- **Affected:** `src/model_utils.py`; notebooks calling `load_model` (`phase10_ensemble.ipynb`, `phase4_mlp_optuna_merged.ipynb`, `_optuna_mlp_hp_search.py`)
- **Remediation:** Add `weights_only=True` to every `torch.load` (or route through a single hardened loader); stop tracking `.pth` files (`git rm --cached`, extend `.gitignore` to `*.pth`).

### SEC-2 — Secrets / credentials — **Info (good)**
- **Description:** No hardcoded credentials, API keys, or secrets found in `src/` (grep-verified); no network-exposed service.
- **Remediation:** Continue current policy; never commit `.env` or database files.

---

## 4. Dependency Health

### DEP-1 — Unpinned dependencies — **High**
- **Description:** `requirements.txt` lists bare names (`torch`, `torchvision`, `scikit-learn`, `pandas`, ...). Any environment install is non-reproducible; the README's install guide (`pip install torch torchvision --index-url .../cu130`) already targets a specific (bleeding-edge) CUDA build, so results are not portable across machines.
- **Affected:** `requirements.txt`, `README.MD`
- **Remediation:** Pin exact versions (`pip freeze > requirements.lock`) or constrained ranges; document the tested Python/torch matrix; add `requirements-dev.txt` with `pytest`.

### DEP-2 — Runtime risk — **Low**
- **Description:** No version constraints; the documented install pulls a very recent torch/CUDA 130 build. Behavioral changes across versions can silently alter results.
- **Affected:** `requirements.txt`
- **Remediation:** Same as DEP-1; record actual versions in `requirements.lock`.

---

## 5. Architecture Consistency

### ARC-1 — Flat `src/` + duplicated logic — **Medium**
- **Description:** LAB1 uses a flat `src/{data,eval,model,train,vis}_utils.py` layout (no `src/data`, `src/models`, `src/training`, `src/eval` packages). Core logic (evaluation, confusion matrices, metric writing, model definitions) is also duplicated inside 18 notebooks (14 CNN + 4 MLP phases) rather than imported.
- **Affected:** `src/*.py`, `notebooks/error_analysis/**/*.ipynb`
- **Remediation:** Adopt the LAB2 package layout; make notebooks thin orchestrators importing from `src/`.

### ARC-2 — Tracked binary artifacts — **Medium**
- **Description:** The branch tracks **14 `.pth` checkpoints, 2 Optuna `.db` databases, 5 Plotly `.html`, and 28 `.png`** files (272 tracked files total). `.gitignore` ignores `*.pt` but **not** `*.pth` or `*.db`, so weights and databases are committed. This bloats the repository, causes merge conflicts on regenerated binaries, risks accidental distribution of trained weights, and frequently corrupts binary DBs in git.
- **Affected:** `.gitignore`, `outputs/**`
- **Remediation:** Add `*.pth`, `*.db`, `*.html`, `*.png` (except curated result summaries) to `.gitignore`; `git rm --cached` the binaries; keep only small, human-readable metrics (`.txt`/`.csv`) and a few curated plots tracked.

### ARC-3 — Empty config directory — **Low**
- **Description:** `config/` contains only `.gitkeep`; no centralized configuration despite the Rulebase's architecture expectations.
- **Affected:** `config/`
- **Remediation:** Either populate with a config file consumed by `src/`, or remove the empty directory.

---

## 6. Test Coverage

### TST-1 — No tests — **High**
- **Description:** There is **no `tests/` directory** on the branch and `pytest` is not declared in `requirements.txt`. None of the training, evaluation, or data-loading code is covered; the "fix overfitting" data-augmentation change (commit `1e51fe4`) shipped with no regression test.
- **Affected:** repository root, `requirements.txt`
- **Remediation:** Create `tests/` with unit tests for `data_utils` (split persistence, transforms), `train_utils` (1-epoch smoke), `eval_utils` (metrics correctness), and `model_utils` (save/load round-trip with `weights_only=True`). Add `pytest` to dev requirements.

### TST-2 — No CI / smoke enforcement — **Medium**
- **Description:** No `.github/workflows` or pre-commit; the Rulebase's testing requirement is not enforced by automation, and a smoke-test checklist is not present on this branch.
- **Affected:** repository root
- **Remediation:** Add a CI workflow that installs pinned deps, runs `pytest`, and executes a notebook smoke test on every PR.

---

## 7. Performance Bottlenecks

### PERF-1 — Single-threaded data loading — **Low**
- **Description:** `get_dataloaders` and the notebooks do not set `num_workers`, so PyTorch defaults to `num_workers=0` (single-threaded). FashionMNIST is small, so impact is low, but it becomes a GPU-starvation risk for larger runs.
- **Affected:** `src/data_utils.py`, notebooks
- **Remediation:** Set `num_workers` explicitly (with the Python-3.14 spawn-safe workaround if applicable) and document the choice.

### PERF-2 — Recompute-heavy notebooks — **Low**
- **Description:** Each phase notebook re-trains/re-evaluates and regenerates artifacts; there is no on-disk result cache, so re-runs repeat work. Impact is low because FashionMNIST models are tiny.
- **Affected:** `notebooks/error_analysis/**`
- **Remediation:** Cache extracted metrics/features to `outputs/` and load-if-present, per the notebook independence rules.

---

## 8. Compliance with Policies & Procedures

Assessment against the branch's own `Docs/Rulebase.md` (15 sections) and `agents/rules.md`:

| Policy / procedure | Compliance | Evidence / gap |
|---|---|---|
| §1 Role Definition | **Compliant** | AI/developer responsibilities documented; `agents/rules.md` present |
| §2 Transparency & AI Logging | **Compliant** | `Docs/agents_log.md`, `agents_log.md` exist and are populated |
| §3 Development Workflow | **Partial** | Iterative commits; but binaries/DBs committed alongside code |
| §4 Software Architecture Rules | **Partial** | Flat `src/` utils; hardcoded paths; no package layout |
| §5 Coding Rules | **Partial** | Consistent naming; no linting enforcement; hardcoded paths |
| §6 AI-Generated Code Rules | **Partial** | Logging exists; no explicit AI-generated-code markers per file |
| §7 Documentation Standards | **Compliant** | Changelogs, notebook headers, `Docs/`, `PROJECT_STRUCTURE` |
| §8 Report Writing Rules | **Compliant** | `LAB1_TEAM_CONTRIBUTION.md`, phase changelogs |
| §9 Testing Rules | **Non-compliant** | No `tests/`, no `pytest`, no verification evidence |
| §10 Version Control Rules | **Partial** | Commit volume good; tracked `.pth`/`.db` violates VCS hygiene |
| §11 Technical Decision Records | **Partial** | Changelogs serve as de-facto TDR; no formal TDR format |
| §12 Security Rules | **Non-compliant** | `torch.load` without `weights_only`; weights committed to repo |
| §13 Debugging Rules | **Compliant** | Systematic `error_analysis/` phases (focal loss, augmentation, etc.) |
| §15 Project Completion Checklist | **Partial** | No checklist artifact present in the branch |

**Bottom line:** The documentation, transparency, and debugging policies are well met. The **testing (§9)** and **security (§12)** policies are not met, and **version control hygiene (§10)** is only partially met due to committed binaries.

---

## 9. Detailed Risk Analysis

| Risk | Likelihood | Impact | Overall | Description & mitigation |
|---|---|---|---|---|
| **Pickle RCE via `torch.load`** | Low | High | **Moderate** | Checkpoints are local/trusted today, but weights are in the repo; a tampered `.pth` executes arbitrary code when loaded. Mitigate with `weights_only=True` + untrack weights. |
| **Reproducibility failure** | High | High | **High** | Unpinned deps + CUDA-specific install → results differ across machines. Pin `requirements.lock`. |
| **Regression / correctness** | High | High | **High** | No tests; the overfitting-fix shipped untested. Add unit + smoke tests and CI. |
| **Repo bloat / data governance** | Medium | Medium | **Medium** | 14 `.pth`, 2 `.db`, 5 `.html`, 28 `.png` tracked → large repo, conflicts, accidental weight distribution, corrupt DBs. Gitignore + `git rm --cached`. |
| **Portability breakage** | Medium | Medium | **Medium** | Hardcoded `../outputs/...` paths fail when run from other directories. Root-relative `Path` resolution. |
| **Silent overfitting** | Medium | Medium | **Medium** | `train_model` tracks only train loss; no validation/early stopping in the loop. Add val tracking. |
| **Policy non-compliance exposure** | Medium | Medium | **Medium** | Rulebase §9/§12/§10 unmet; audit/review exposure. Remediate per plan. |

---

## 10. Overall Project Health

| Dimension | Rating | Notes |
|---|---|---|
| Documentation & traceability | **Strong** | Changelogs, phase notebooks, team doc, rulebase |
| Code quality | **Fair** | Hardcoded paths; no validation in train loop; no linting |
| Security | **Weak** | Unsafe `torch.load`; weights committed |
| Dependency health | **Weak** | Unpinned; no dev requirements |
| Architecture consistency | **Fair** | Flat utils + notebook duplication; no central config |
| Test coverage | **None** | No tests, no pytest, no CI |
| Performance | **Good enough** | Small dataset; minor loading/caching improvements only |

**One-line health:** A documentation-strong research branch whose engineering safeguards (tests, security hardening, dependency pinning, VCS hygiene) lag well behind the LAB2 codebase.

---

## 11. Prioritized Action Plan

### P0 — Fix now (security, reproducibility, verifiability)
1. **Harden checkpoint loading** — add `weights_only=True` to `src/model_utils.py` (and any notebook `torch.load`), and stop loading untrusted files. *(SEC-1)*
2. **Untrack binaries** — add `*.pth`, `*.db`, `*.html`, `*.png` to `.gitignore`; `git rm --cached` the 14 weights + 2 DBs (keep human-readable `.txt`/`.csv` metrics). *(ARC-2)*
3. **Pin dependencies** — `pip freeze > requirements.lock`; add `requirements-dev.txt` with `pytest`. *(DEP-1, DEP-2)*
4. **Add a minimal test suite** — `tests/` for `data_utils` (split, transforms), `train_utils` (1-epoch smoke), `eval_utils` (metrics), `model_utils` (save/load round-trip). *(TST-1)*

### P1 — Next iteration (raises confidence)
5. **Root-relative paths** — centralize `PROJECT_ROOT` resolution; replace `'../outputs/...'` throughout `src/`. *(CQ-1)*
6. **Validation + early stopping in `train_model`** with best-model persistence. *(CQ-2)*
7. **Add CI** — GitHub Actions running `pytest` + a notebook smoke script on every PR. *(TST-2)*

### P2 — Polish (when time permits)
8. Adopt the LAB2 package layout (`src/data|models|training|eval`) and de-duplicate notebook logic. *(ARC-1)*
9. Add `pyproject.toml` + `ruff`/`mypy`; enforce in CI. *(CQ-3)*
10. Populate or remove `config/`; set explicit `num_workers`; cache recompute-heavy results. *(ARC-3, PERF-1, PERF-2)*

---

## 12. Notes & Scope

- This audit inspects the **`LAB1-FashionMNIST-Classification/main`** branch state in git (HEAD `9d99b07`); it does not modify the working tree.
- Findings reference LAB1's own artifacts only; LAB2 (`src/`, `tests/`, `agents/rules/*`) is audited separately in `docs/codebase-audit.md`.
- The reference report link (Google Docs) was a placeholder and inaccessible from this environment; this document mirrors the local reference's structure, terminology, and detail level and is saved as `docs/codebase-audit-lab1.md` for upload to the shared Drive folder.
