# Codebase Audit Report — Practice 3 (Get Started with Hugging Face)

- **Motivation/Background**: This audit runs at the start of the Practice 3
  implementation cycle (Step 10 of
  [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md)) to catch drift between
  what the repo's docs claim and what the code actually contains, before
  implementation phases build on wrong assumptions.
- **Purpose**: Establish a baseline of code quality, security, dependencies,
  architecture, tests, and performance for the current working tree, per the
  [CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md) procedure.
- **Overview Pipeline**: Git-tree inspection (revision + status), glob/grep of
  `src/`, `tests/`, `notebooks/`, `experiments/`, dependency matrix vs the
  active environment (`pip show`), and a compliance check against the project
  rulebase in [agents/rules/](rules/).
- **Detailed Plan**: Exec summary; findings summary; per-area findings (code
  quality, security, dependencies, architecture, tests, performance);
  compliance; risk analysis; overall health; prioritized action plan.
- **References**: `git`, `grep`, `pip`/`importlib.metadata`,
  [agents/rules/](rules/) (rulebase), previous audit of the template scaffold.

**Audited revision:** working tree at `HEAD = ffac43b` ("Chore: Clear
everything, prepare template for LAB3") + uncommitted doc changes from this
session (`agents/OVERVIEW.md`, `agents/phases/PHASE_TEMPLATE.md` modified;
`agents/PROJECT_ROADMAP.md` and 8 phase docs untracked).
**Audit date:** 2026-08-11.

- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-09-06T21:35:00+07:00

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
- [Appendix — Resolution Log](#appendix--resolution-log)
- [Appendix — Phase 2 & 4 Completion Audit (2026-08-12)](#appendix--phase-2--4-completion-audit-2026-08-12)

---

## 1. Executive Summary

**Scope:** full working tree at `ffac43b` + uncommitted governance docs; method:
git-tree inspection, glob/grep of `src/`/`tests`/`notebooks`/`experiments`,
dependency check against the active Python environment, compliance review
against [agents/rules/](rules/).

**Verdict:** the repository is a **clean pre-implementation scaffold whose
governance layer is ahead of its code**. The rulebase is coherent, the
layered SoC layout matches [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md),
and smoke tests exist — but none of the documented modules are implemented,
the HF stack is only half-provisioned, and the test suite cannot currently run.

- **Strong:** (1) governance/rulebase is consistent and recently strengthened
  (PURPOSE, roadmap, 8 phase docs, overview); (2) SoC layout and naming match
  the conventions; (3) no application code exists yet, so no security or
  performance defects can hide in it.
- **Blocks maturity:** (1) every module documented in the `src/` layer READMEs
  ([train_model.py](../src/training/README.md), `run_logger.py`,
  `checkpoint_utils.py`, `build_model.py`, `get_loaders`, `evaluate_model.py`)
  is **absent**; (2) `transformers` is **not installed** and
  [requirements.txt](../requirements.txt) is out of sync with the environment —
  this blocks both exercises; (3) **pytest is not installed**, so the smoke
  suite cannot run to gate anything.

**Health rating: Fair** (see [§11](#11-overall-project-health)); top actions in
[§12](#12-prioritized-action-plan).

---

## 2. Findings Summary

> ### Finding Resolution Status Vocabulary
> The **Status** column tracks whether **that specific finding/defect** has been remediated, independent of whether the broader audit or milestone is complete:
> - **`RESOLVED`**: The specific defect/risk has been completely remediated, validated by automated tests or physical inspection, and verified on disk.
> - **`PARTIALLY RESOLVED`**: An interim mitigation, partial patch, or workaround has been applied, but remaining work or pending verification is required for complete resolution.
> - **`NOT RESOLVED`**: The finding has been diagnosed and documented, but no corrective engineering action has yet been taken.

| ID | Area | Severity | Status | Title | Section |
|---|---|---|---|---|---|
| `ARC-1` | Architecture | **High** | `RESOLVED` | Documented `src/` modules entirely absent (docs claim code that doesn't exist) | [6. Architecture Consistency](#6-architecture-consistency) |
| `DEP-1` | Dependencies | **High** | `RESOLVED` | `transformers` not installed; environment vs `requirements.txt` drift | [5. Dependency Health](#5-dependency-health) |
| `TST-1` | Tests | **High** | `RESOLVED` | pytest not installed — smoke suite cannot execute | [7. Test Coverage](#7-test-coverage) |
| `AQ-1` | Code quality | **Medium** | `RESOLVED` | `run_logger.py` documented in two layers (ownership ambiguity) | [3. Code Quality](#3-code-quality) |
| `DEP-2` | Dependencies | **Medium** | `RESOLVED` | No version pins / no lockfile (reproducibility) | [5. Dependency Health](#5-dependency-health) |
| `ARC-2` | Architecture | **Medium** | `RESOLVED` | `FOLDER_STRUCTURE.md` not updated for `PROJECT_ROADMAP.md` + new phase docs | [6. Architecture Consistency](#6-architecture-consistency) |
| `TST-2` | Tests | **Medium** | `RESOLVED` | Only smoke tests; no unit tests (modules don't exist yet) | [7. Test Coverage](#7-test-coverage) |
| `AQ-2` | Code quality | **Low** | `RESOLVED` | Unused `needs_data` marker in [conftest.py](../tests/conftest.py) | [3. Code Quality](#3-code-quality) |
| `AQ-3` | Code quality | **Low** | `RESOLVED` | `agents/progress/*_STATUS.md` referenced by OVERVIEW + 8 phase docs but absent | [3. Code Quality](#3-code-quality) |
| `ARC-3` | Architecture | **Low** | `RESOLVED` | Two phase-doc templates coexist (`templates/PHASE_DOC_TEMPLATE.md` vs `phases/PHASE_TEMPLATE.md`) | [6. Architecture Consistency](#6-architecture-consistency) |
| `AQ-4` | Code quality | Info | `RESOLVED` | Root [README.md](../README.md) still carries template placeholders | [3. Code Quality](#3-code-quality) |
| `SEC-1` | Security | Info | `RESOLVED` | No application code in tree — minimal attack surface (positive) | [4. Security Vulnerabilities](#4-security-vulnerabilities) |
| `SEC-2` | Security | Info | `RESOLVED` | No secrets; `.gitignore` excludes data, checkpoints, `.env` (positive) | [4. Security Vulnerabilities](#4-security-vulnerabilities) |
| `PERF-1` | Performance | Info | `RESOLVED` | No training/eval code yet; constraints already documented (positive) | [8. Performance Bottlenecks](#8-performance-bottlenecks) |

Severity totals: **High × 3, Medium × 4, Low × 3, Info × 4.** Cross-reference:
[risk analysis §10](#10-detailed-risk-analysis) and
[compliance §9](#9-compliance-with-policies-and-procedures).

---

## 3. Code Quality

### `AQ-1`: `run_logger.py` documented in two layers — ownership ambiguity
- **Severity:** Medium
- **Status:** RESOLVED
- **Description:** [src/training/README.md](../src/training/README.md) lists
  `run_logger.py` as part of the training layer, while
  [src/utils/README.md](../src/utils/README.md) lists the same file (plus
  `checkpoint_utils.py`) as shared helpers. The root
  [README.md](../README.md) architecture diagram also shows `run_logger.py`
  under Training. Two sources of truth invite divergence when the file is
  finally implemented.
- **Affected:** [src/training/README.md](../src/training/README.md),
  [src/utils/README.md](../src/utils/README.md),
  [README.md](../README.md).
- **Remediation:** pick one canonical owner (recommended: `src/utils/`, since
  [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) imports it
  as `src/utils/run_logger.py`) and update the other two docs to defer to it.
  — tracked in [Action P1.2](#12-prioritized-action-plan)
- **Resolution Evidence:** Canonicalized `run_logger.py` and `checkpoint_utils.py` ownership to `src/utils/` (now `src/lab3/utils/run_logger.py`) per Action P1.2; updated layer READMEs to defer to it.

### `AQ-2`: Unused `needs_data` marker in conftest
- **Severity:** Low
- **Status:** RESOLVED
- **Description:** [conftest.py](../tests/conftest.py) defines a `needs_data`
  skipif marker, but no test references it ([test_smoke.py](../tests/test_smoke.py)
  never applies it). It is dead scaffolding today.
- **Affected:** [tests/conftest.py](../tests/conftest.py).
- **Remediation:** either start applying it to data-dependent tests (planned
  Phase 2/3 tests) or delete it. — tracked in [Action P2.1](#12-prioritized-action-plan)
- **Resolution Evidence:** Removed unused `needs_data` marker from `tests/conftest.py` per Action P2.1.

### `AQ-3`: Progress status files referenced but absent
- **Severity:** Low
- **Status:** RESOLVED
- **Description:** [OVERVIEW.md](OVERVIEW.md) and all 8 phase docs
  ([phases/SETUP.md](phases/SETUP.md) … [phases/REPORT.md](phases/REPORT.md))
  link to `agents/progress/<PHASE>_STATUS.md`, but the directory contains only
  `PROGRESS_TEMPLATE.md` — the 8 referenced files do not exist.
- **Affected:** [OVERVIEW.md](OVERVIEW.md), all [phases/](phases/) docs,
  [agents/progress/](progress/).
- **Remediation:** acceptable *if* intentional (Step 9 of
  [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md) creates them as each
  phase starts); otherwise create stubs. Document the intent so it isn't
  re-flagged. — tracked in [Action P1.3](#12-prioritized-action-plan)
- **Resolution Evidence:** Created all 8 progress status files per Action P1.3 (now colocated under `docs/lab3/progress/`).

### `AQ-4`: Root README template placeholders
- **Severity:** Info
- **Status:** RESOLVED
- **Description:** [README.md](../README.md) still contains `[PROJECT_NAME]`
  and `[Fill in — license…]` placeholders from the bootstrap template.
- **Affected:** [README.md](../README.md).
- **Remediation:** fill in the project name and license/course context.
  — tracked in [Action P2.3](#12-prioritized-action-plan)
- **Resolution Evidence:** Replaced `[PROJECT_NAME]` and placeholder license text with LAB3 course context in `README.md` per Action P2.3.

---

## 4. Security Vulnerabilities

### `SEC-1`: No application code in tree — minimal attack surface (positive)
- **Severity:** Info
- **Status:** RESOLVED
- **Description:** `src/` contains only package `__init__.py` files; there is
  no model, training, or evaluation code to contain injection, RCE, or data
  leaks yet. The one security-relevant requirement (checkpoints load with
  `weights_only=True`) is already mandated in
  [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) §3.
- **Affected:** entire `src/` tree.
- **Remediation:** none now — enforce `weights_only=True` when
  `checkpoint_utils.py` is implemented. — tracked in [Action P0.2](#12-prioritized-action-plan)
- **Resolution Evidence:** Positive baseline finding; maintained throughout implementation by enforcing `weights_only=True` in `checkpoint_utils.py` and `safe_load_checkpoint` (verified in PR #15 audit).

### `SEC-2`: No secrets; `.gitignore` excludes sensitive paths (positive)
- **Severity:** Info
- **Status:** RESOLVED
- **Description:** No credentials/tokens found in tracked files; `.gitignore`
  covers `.env*`, `/data/raw/`, `/experiments/checkpoints/*.pt`, and
  `/experiments/runs/`, and `.kilo` config is ignored.
- **Affected:** [.gitignore](../.gitignore).
- **Remediation:** none.
- **Resolution Evidence:** Positive baseline finding; `.gitignore` rules verified on disk across all phases (`.env*`, `/data/raw/`, `/experiments/checkpoints/*.pt`, `/experiments/runs/`).

---

## 5. Dependency Health

### `DEP-1`: `transformers` missing; environment vs `requirements.txt` drift
- **Severity:** High
- **Status:** RESOLVED
- **Description:** The active environment has `torch 2.10.0`, `datasets 5.0.0`,
  and `evaluate 0.4.6`, but **`transformers` is not installed** — yet it is the
  library both exercises are built on (Exercise 1 step 1). Conversely, the
  [requirements.txt](../requirements.txt) declares `pytest` (not installed in
  the environment) and does **not** declare `transformers`/`datasets`/
  `evaluate` (installed but untracked). The environment was provisioned outside
  the declared dependency file.
- **Affected:** [requirements.txt](../requirements.txt), active environment.
- **Remediation:** `pip install transformers`, and rewrite
  `requirements.txt` to declare the HF stack (roadmap task T1) so the file and
  environment converge. — tracked in [Action P0.1](#12-prioritized-action-plan)
- **Resolution Evidence:** Provisioned `.venv` with HF stack (`transformers`, `datasets`, `evaluate`), updated `requirements.txt`, and added `requirements.lock` per Action P0.1.

### `DEP-2`: No version pins / no lockfile
- **Severity:** Medium
- **Status:** RESOLVED
- **Description:** [requirements.txt](../requirements.txt) lists bare package
  names; the repo references a `requirements.lock` (pip freeze) that does not
  exist. The roadmap's T1 pins HF versions, but nothing is pinned today.
- **Affected:** [requirements.txt](../requirements.txt).
- **Remediation:** pin top-level versions and add `requirements.lock`.
  — tracked in [Action P1.4](#12-prioritized-action-plan)
- **Resolution Evidence:** Added top-level version pins in `requirements.txt` and generated `requirements.lock` via pip freeze per Action P1.4.

---

## 6. Architecture Consistency

### `ARC-1`: Documented `src/` modules entirely absent
- **Severity:** High
- **Status:** RESOLVED
- **Description:** Every module the layer READMEs describe is missing —
  only `__init__.py` files exist. Absent: `train_model.py`,
  `run_logger.py`, `<feature>_train.py`
  ([src/training/README.md](../src/training/README.md)); `build_model.py`
  ([src/models/README.md](../src/models/README.md)); `transforms.py`,
  `dataloader.py` (`get_loaders`), `config.py`
  ([src/data/README.md](../src/data/README.md)); `evaluate_model.py`
  ([src/eval/README.md](../src/eval/README.md)); `run_logger.py`,
  `checkpoint_utils.py` ([src/utils/README.md](../src/utils/README.md)). The
  rules likewise reference `src/training/train_model.py` and
  `src/utils/run_logger.py` as existing artifacts
  ([LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md)).
- **Affected:** all six `src/` layer packages; root [README.md](../README.md)
  architecture diagram.
- **Remediation:** these are planned implementations (roadmap Phases 4–6), so
  the fix is to **label them as planned in the docs** (or implement them) —
  silence the drift either way. — tracked in [Action P0.2](#12-prioritized-action-plan)
- **Resolution Evidence:** Annotated `src/` layer READMEs as planned per Action P0.2; subsequently all modules were implemented in Phases 2–8 (`src/lab3/data`, `src/lab3/models`, `src/lab3/training`, `src/lab3/eval`, `src/lab3/utils`).

### `ARC-2`: `FOLDER_STRUCTURE.md` not updated for new governance files
- **Severity:** Medium
- **Status:** RESOLVED
- **Description:** [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) does not
  list `agents/PROJECT_ROADMAP.md` (it describes OVERVIEW.md as the roadmap),
  and lists only example phase docs — the 8 phase files added this session
  ([phases/SETUP.md](phases/SETUP.md), `DATA_PREP.md`, `FEATURE_SPLIT.md`,
  `BASELINE.md`, `MODEL.md`, `TRAINING_INFO.md`, `EVAL.md`, `REPORT.md`) are
  undocumented there.
- **Affected:** [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md).
- **Remediation:** update the structure listing (with human approval per the
  rule's own constraint). — tracked in [Action P1.1](#12-prioritized-action-plan)
- **Resolution Evidence:** Updated `FOLDER_STRUCTURE.md` and `agents/README.md` to index `PROJECT_ROADMAP.md` and the 8 phase docs per Action P1.1.

### `ARC-3`: Two phase-doc templates coexist
- **Severity:** Low
- **Status:** RESOLVED
- **Description:** `templates/PHASE_DOC_TEMPLATE.md`
  and [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md) both claim to be the
  phase-doc template; the 8 phase docs were created from the latter. Two
  templates will drift.
- **Affected:** `templates/PHASE_DOC_TEMPLATE.md`,
  [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md).
- **Remediation:** designate one canonical template and remove or alias the
  other. — tracked in [Action P2.2](#12-prioritized-action-plan)
- **Resolution Evidence:** Deleted redundant `templates/PHASE_DOC_TEMPLATE.md` and standardized on `phases/PHASE_TEMPLATE.md` per Action P2.2.

---

## 7. Test Coverage

### `TST-1`: pytest not installed — smoke suite cannot execute
- **Severity:** High
- **Status:** RESOLVED
- **Description:** `python -m pytest` fails in both the `.venv` and system
  Python ("No module named pytest"), despite pytest being declared in
  [requirements.txt](../requirements.txt). [test_smoke.py](../tests/test_smoke.py)
  therefore cannot run, so the smoke-gating discipline
  ([SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md)) is
  unenforceable today.
- **Affected:** [tests/](../tests/), environment.
- **Remediation:** install pytest into the active environment (and any CI) —
  included in [Action P0.1](#12-prioritized-action-plan).
- **Resolution Evidence:** Installed `pytest` into `.venv` and verified passing in Phase 2 & 4 completion audit per Action P0.1.

### `TST-2`: Only smoke tests; no unit tests
- **Severity:** Medium
- **Status:** RESOLVED
- **Description:** The entire suite is
  [test_smoke.py](../tests/test_smoke.py) (3 tests: imports, directory
  existence, knowledge-base files). No unit tests exist for the modules the
  roadmap plans (`prepare_imdb.py`, `imdb_sentiment_train.py`,
  `evaluate_model.py`), because those modules don't exist yet.
- **Affected:** [tests/](../tests/).
- **Remediation:** add a unit test alongside each module as its phase
  implements it (roadmap Phases 3, 6, 7). — tracked in [Action P0.2](#12-prioritized-action-plan)
- **Resolution Evidence:** Added unit and integration tests alongside implemented modules across PR #13, #15, #16 (`test_data_leakage.py`, `test_trainer_regressions.py`, `test_cleanlab_denoiser.py`); suite expanded to 15+ passing tests.

---

## 8. Performance Bottlenecks

### `PERF-1`: No training/eval code yet — performance is N/A (positive)
- **Severity:** Info
- **Status:** RESOLVED
- **Description:** With no implementation, there are no hot paths to profile.
  The constraints that will shape performance are already documented:
  `num_workers = 0` safety note ([src/data/README.md](../src/data/README.md)),
  ≤6 GB VRAM target and gradient-accumulation mitigation (roadmap risk R1,
  [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)).
- **Affected:** none.
- **Remediation:** none now; revisit after Phase 6 produces the first run.
- **Resolution Evidence:** Positive baseline finding; VRAM targets (<= 3.5GB/6GB), gradient accumulation, and `num_workers=0` constraints documented and honored across all training scripts.

---

## 9. Compliance with Policies and Procedures

Assessed against the rulebase in [agents/rules/](rules/).

| Policy / procedure | Compliance | Evidence / gap | Related finding |
|---|---|---|---|
| [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) | Partial | Directory layout matches; new governance files undocumented | [ARC-2](#2-findings-summary) |
| [NAMING_CONVENTION.md](rules/NAMING_CONVENTION.md) | Compliant | Recent docs UPPER_SNAKE; planned scripts/notebooks snake_case/NN_ prefix | — |
| [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) | Partial | Rules documented; referenced utils absent; no runs to violate yet | [ARC-1](#2-findings-summary) |
| [RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md) | Compliant | [experiments/results/README.md](../experiments/results/README.md) skeleton present; no results yet | — |
| [NOTEBOOK_HEADER_CONVENTION.md](rules/NOTEBOOK_HEADER_CONVENTION.md) | Compliant (vacuous) | No notebooks in tree yet | — |
| [MD_CONVENTION.md](rules/MD_CONVENTION.md) | Compliant | New roadmap/phases/OVERVIEW follow header + link rules | — |
| [CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md) | Compliant | This report executes the procedure | — |
| [SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md) | Partial | Checklist + tests exist, but tests cannot run (pytest absent) | [TST-1](#2-findings-summary) |
| Script-only training (README/AGENT_AI) | Compliant (vacuous) | No training loops anywhere; notebooks empty | — |
| [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md) workflow | Partial | Steps 1–6 done; Step 7 README incomplete; Step 9 progress files pending | [AQ-3](#2-findings-summary), [AQ-4](#2-findings-summary) |

Cross-reference: non-compliance items map to [risk §10](#10-detailed-risk-analysis)
and [action §12](#12-prioritized-action-plan).

---

## 10. Detailed Risk Analysis

| Risk | Likelihood | Impact | Overall | Description & mitigation | Related finding |
|---|---|---|---|---|---|
| Docs–code drift misleads agents/builders | High | High | **Critical** | Docs describe modules that don't exist; an agent may `import src.training.train_model` and fail, or assume results that were never produced. Mitigate: label planned-vs-implemented (P0.2). | [ARC-1](#2-findings-summary) |
| HF stack not provisioned blocks both exercises | High | High | **Critical** | `transformers` missing; exercises cannot start. Mitigate: install + pin (P0.1). | [DEP-1](#2-findings-summary) |
| Runs proceed ungated by tests | Med | High | **High** | Smoke suite unrunable; long finetune runs would start without the prescribed smoke gate. Mitigate: install pytest, run suite (P0.1). | [TST-1](#2-findings-summary) |
| Environment not reproducible | Med | Med | **Medium** | Unpinned deps; another machine differs. Mitigate: pins + lockfile (P1.4). | [DEP-2](#2-findings-summary) |
| Rulebase/listing drift compounds | Low | Med | **Low–Med** | FOLDER_STRUCTURE and dual templates decay if untouched. Mitigate: update/consolidate (P1.1, P2.2). | [ARC-2](#2-findings-summary), [ARC-3](#2-findings-summary) |
| Placeholder README misleads users | Low | Low | **Low** | `[PROJECT_NAME]`/license gaps confuse readers. Mitigate: fill in (P2.3). | [AQ-4](#2-findings-summary) |

Cross-reference: severity ratings originate in [§2](#2-findings-summary);
mitigations are scheduled in [§12](#12-prioritized-action-plan).

---

## 11. Overall Project Health

| Dimension | Rating | Notes |
|---|---|---|
| Code quality | **Fair** | Only `__init__.py` + tests; no drift in what exists, but ownership ambiguity (AQ-1) and dead scaffold (AQ-2) |
| Security | **Good** | No code to attack; `weights_only=True` and ignore rules already mandated |
| Dependencies | **Weak** | Env ≠ `requirements.txt`; `transformers` missing; nothing pinned |
| Architecture | **Fair** | SoC layout and naming correct; documented modules absent (ARC-1) |
| Tests | **Weak** | Smoke suite exists but cannot run (TST-1); no unit tests |
| Performance | **N/A** | No implementation yet; constraints documented |
| Docs / governance | **Good** | Rulebase coherent; PURPOSE → roadmap → 8 phases → overview fully linked; README placeholders remain |

**Overall: Fair** — a well-governed scaffold whose environment and code have
not yet caught up with its documentation.

Cross-reference: per-dimension evidence in [§3](#3-code-quality)–[§8](#8-performance-bottlenecks).

---

## 12. Prioritized Action Plan

### P0 — Fix now (blocks trust / reproducibility / security)
- **P0.1** Provision the environment to match the plan: `pip install
  transformers pytest`, then rewrite [requirements.txt](../requirements.txt) to
  declare `transformers`, `datasets`, `evaluate` (roadmap T1) — addresses
  [DEP-1](#2-findings-summary), [TST-1](#2-findings-summary).
- **P0.2** Stop the docs–code drift: annotate the six `src/` layer READMEs
  (and root [README.md](../README.md)) that their modules are **planned** for
  Phases 3–7, or implement them — addresses
  [ARC-1](#2-findings-summary), [TST-2](#2-findings-summary),
  [SEC-1](#2-findings-summary).

### P1 — Next iteration (raises confidence)
- **P1.1** Update [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) to list
  `agents/PROJECT_ROADMAP.md` and the 8 phase docs — addresses
  [ARC-2](#2-findings-summary).
- **P1.2** Resolve `run_logger.py` ownership (single canonical location in
  `src/utils/`; defer in the other two READMEs) — addresses
  [AQ-1](#2-findings-summary).
- **P1.3** Create `agents/progress/*_STATUS.md` stubs or remove the dangling
  links until phases start — addresses [AQ-3](#2-findings-summary).
- **P1.4** Pin top-level dependency versions; add `requirements.lock` —
  addresses [DEP-2](#2-findings-summary).

### P2 — Polish (when time permits)
- **P2.1** Apply or delete the unused `needs_data` marker —
  addresses [AQ-2](#2-findings-summary).
- **P2.2** Consolidate the two phase-doc templates into one canonical file —
  addresses [ARC-3](#2-findings-summary).
- **P2.3** Fill in root [README.md](../README.md) name/license placeholders —
  addresses [AQ-4](#2-findings-summary).

> Lint/tooling actions (ruff, mypy) are intentionally **not** blocking items
> per the AI-era audit perspective — none are scheduled here.

Cross-reference: each item links back to its finding; the summary table is in
[§2](#2-findings-summary).

---

## Self-review checklist (before finalizing)

- [X] All 5 header fields present
- [X] TOC anchors resolve (lowercase, strip punctuation, spaces → hyphens)
- [X] Cross-reference links between related sections resolve (summary ↔ detail ↔ action plan)
- [X] Metrics match source; file/notebook paths relative to project root
- [X] Dates in `YYYY-MM-DD`; `---` separators between major sections

---

## Appendix — Resolution Log

Actions taken 2026-08-11 to resolve the findings of this audit (fix commit
on branch `LAB3_HuggingFace`):

| Action | Findings addressed | Status |
|---|---|---|
| P0.1 — provision `.venv` (HF stack + pytest); update `requirements.txt`; add `requirements.lock` | [DEP-1](#2-findings-summary), [TST-1](#2-findings-summary) | Resolved |
| P0.2 — annotate the six `src/` layer READMEs as planned | [ARC-1](#2-findings-summary), [TST-2](#2-findings-summary), [SEC-1](#2-findings-summary) | Resolved |
| P1.1 — update [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) + [agents/README.md](README.md) listings (approved) | [ARC-2](#2-findings-summary) | Resolved |
| P1.2 — canonicalize `run_logger.py`/`checkpoint_utils.py` ownership to `src/utils/` | [AQ-1](#2-findings-summary) | Resolved |
| P1.3 — create 8 `agents/progress/*_STATUS.md` stubs | [AQ-3](#2-findings-summary) | Resolved |
| P1.4 — version pins + `requirements.lock` | [DEP-2](#2-findings-summary) | Resolved |
| P2.1 — remove unused `needs_data` marker from [conftest.py](../tests/conftest.py) | [AQ-2](#2-findings-summary) | Resolved |
| P2.2 — delete `templates/PHASE_DOC_TEMPLATE.md` | [ARC-3](#2-findings-summary) | Resolved |
| P2.3 — fill root [README.md](../README.md) placeholders | [AQ-4](#2-findings-summary) | Resolved |

---

## Appendix — Phase 2 & 4 Completion Audit (2026-08-12)

Step-10 gate before accepting DATA_PREP (Phase 2) and BASELINE (Phase 4) as
Done (PR #13 + post-PR fixes):

| Check | Result | Evidence |
|---|---|---|
| Smoke suite | Pass | `pytest` → 3 passed (`.venv`, torch 2.13.0+cu130) |
| Script execution | Pass | `src/data/eda_imdb.py` regenerates plots + JSON; `baseline_imdb_sentiment.py --eval-imdb --max-samples 50` runs end-to-end |
| Lint (ruff, CI rule) | Pass | `ruff check src tests` (BUG_02 fixes verified) |
| Naming convention | Pass | `eda_imdb.py`, `baseline_imdb_sentiment.py` (snake_case); `01_ex1_sentiment_baseline.ipynb` (`NN_short_purpose`) |
| Notebook policy | Pass | 1 header cell per [NOTEBOOK_HEADER_CONVENTION.md](rules/NOTEBOOK_HEADER_CONVENTION.md); 5 code cells; no training loop |
| Results indexing | Pass | both result JSONs registered in [experiments/results/README.md](../experiments/results/README.md) with 5W1H |
| 5W1H reporting | Pass | `metadata_5w1h` in `experiments/results/baseline_imdb_sentiment.json` |
| Dependencies | Pass | pins restored (`torch==2.13.0+cu130` …); `requirements.lock` consistent; CI installs via the cu130 index |
| Constraint consistency | Pass | ≤3.5 GB target propagated to PURPOSE, OVERVIEW, roadmap, phase docs, config |
| CI (GitHub Actions) | Pass | `test` workflow on merged head `dbd0b6b`: conclusion `success` (ruff + pytest, 2026-08-12) |
| Accepted deviations | Note | `scratch/build_notebook.py` (documented in [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md)); TensorBoard under `experiments/runs/baseline_zero_shot/` (experiment, not a training run); notebook may regenerate EDA artifacts when missing |

**Verdict: Phase 2 (DATA_PREP) and Phase 4 (BASELINE) pass the completion audit.**

**Merge status:** PR #13 merged into `LAB3_HuggingFace` as `3f982ee`
(2026-08-12); Phases 1, 2, and 4 accepted as done (roadmap statuses updated).

---

## Appendix — Phase 3, 5–8 Remediation & Completion Audit (2026-08-12)

Step-10 audit gate executed for Phase 3 (FEATURE_SPLIT), Phase 5 (MODEL), Phase 6 (TRAINING_INFO), Phase 7 (EVAL), and Phase 8 (REPORT) after addressing blocking bugs B1 and B2:

| Check | Result | Evidence |
|---|---|---|
| B1 CI tracking | Pass | `experiments/runs/.gitkeep` restored; `test_smoke.py::test_core_directories_exist` passes |
| B2 Dependency declaration | Pass | `accelerate>=1.1.0` declared in `requirements.txt` & pinned (`accelerate==1.14.0`) in `requirements.lock`; `--smoke` training run passes |
| Results indexing | Pass | `imdb_sentiment_eval.json` fully indexed in `experiments/results/README.md` with 5W1H metrics |
| Config cleanup | Pass | Unused `save_total_limit: 2` removed from `configs/config_imdb_sentiment.yaml` |
| Resume & RNG safety | Pass | Trainer resume restores RNG seed and adjusts `num_train_epochs` prior to optimizer/scheduler creation |
| Cache staleness guard | Pass | `prepare_imdb` validates cache metadata (`max_length`, `model_name`, etc.) before reusing disk cache |
| Gitignore consistency | Pass | `.gitignore` rules maintained; `experiments/runs/.gitkeep` explicitly tracked |

| Smoke test suite | Pass | `pytest` → 3 passed out of 3 |

**Verdict: Phases 3, 5, 6, 7, and 8 pass the Step-10 completion audit.**

---

## Appendix — PR #15 Review & Remediation Audit (2026-08-14)

Audit gate over the changes introduced since `9d4627ce` (Exercise 2 IMDB feature,
PR #13/#14/#15, and the local fix commit `8f536b0`). Scope: `9d4627ce..HEAD`
(68 files, +5078/−862).

| Check | Result | Evidence |
|---|---|---|
| B1 — package importable | **Fixed** | `src/__init__.py` now uses lazy PEP 562 exports; `error_auditor.py` lazy-imports `scratch.analyze_misclassifications` (optional helper). `import src` succeeds without the helper; regression tests added. |
| B2 — resume helper | **Fixed** | `safe_load_checkpoint` imported in `imdb_sentiment_train.py` (was missing → NameError on `--resume`). |
| Lint (ruff, CI rule) | **Pass** | `ruff check src tests` clean (import sort, unused imports, f-strings fixed). |
| Tests | **Pass** | `pytest` → 12 passed (package-import regressions + training/checkpoint unit tests). |
| Early-stop flag | **Fixed** | `early_stop_triggered` captured reliably in `FullStateCallback.on_train_end`; `--resume` halts / `--force-resume` rewinds per [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) §5. |
| Classifier dropout | **Fixed** | `seq_classif_dropout` passed as a constructor kwarg to `from_pretrained` (was a silent post-hoc config mutation no-op). |
| Results consistency | **Fixed** | [`experiments/results/README.md`](../experiments/results/README.md) reconciled to the committed eval JSON (**93.23%** / 0.9790 / 0.9323, `distilbert-finetune-512-hyper_best.pt`); default-config comment corrected. |
| Reproducibility | **Annotated** | Env divergence (grid ran on torch 2.6.0+cu124 / transformers 4.56.0 vs pinned torch 2.13.0+cu130 / transformers 5.15.0) flagged in results README + EX3 report. |
| Doc–code drift | **Fixed** | `experiments/README.md` indexes EX4-results + EX6; `FOLDER_STRUCTURE.md` + `agents/README.md` list `agents/plans/`; OVERVIEW ROC-AUC corrected to 0.9790. |
| Data leakage guard | **Pass** | `prepare_imdb._validate_no_test_leakage` asserts 25k/22.5k/2.5k split geometry. |
| Notebook policy | **Pass** | Notebooks invoke training via documented subprocess CLI; no inline training loop. |
| Known residual | Note | `ErrorAuditor.audit_top_misclassifications` requires the optional `scratch.analyze_misclassifications` module (absent in-tree) — raises a clear error; add the helper if the notebook cell is to run. |

**Verdict:** the Exercise 2 feature and its review remediation are code- and
doc-clean, lint-green, and test-green. The two PR-#15 merge blockers (B1/B2)
are resolved in local commit `8f536b0`. Any remaining action is the optional
`scratch` error-audit helper and re-verifying committed eval numbers on the
pinned stack.

---

## Appendix — PR #16 Review & Remediation Audit (2026-08-15)

Audit gate over all changes introduced since `c708cb0` (the `LAB3_HuggingFace`
base): PR #16 `feature/ex2-finetune` (Head-Tail Truncation, 5-Fold OOF
Cleanlab auditing, LoRA PEFT breakthrough) plus the review-fix commit
`ae178d5`. Scope: `c708cb0..b9c4b1e` (merged 2026-08-15, 49 files,
+10,655/−545).

| Check | Result | Evidence |
|---|---|---|
| Head-tail truncation consistency | **Fixed** | `IMDBCleanlabAuditor.compute_train_probabilities` and `ErrorAuditor.audit_top_misclassifications` now tokenize with `head_tail_tokenize()` at the checkpoint's own `max_length` — matching the training distribution instead of prefix truncation. Cache meta records `truncation: head_tail` so the stale committed prefix-truncated `imdb_train_pred_probs.npy` is invalidated. |
| `export_denoised_dataset` metadata | **Fixed** | Counts `len(idx_set)` (number of pruned samples) instead of `len(issue_indices)` (dict key count ≈ 6) in `meta.json` and the console report. |
| CUDA/CPU AMP | **Fixed** | `GradScaler` created only on CUDA (`use_amp`); plain `backward()/step()` on CPU in `compute_oof_probabilities`. |
| Config key drift | **Fixed** | `config_imdb_sentiment_denoised_fullft.yaml` `evaluation_strategy` → `eval_strategy` (the trainer reads `eval_strategy`; the old key was silently ignored). |
| Audit checkpoint selection | **Hardened** | `_load_latest_model` warns when the resolved highest-accuracy checkpoint was trained at `max_length < 512`, flagging divergence from the head-tail 512 convention. |
| `truncation_viz` crash guard | **Fixed** | Raises a clear `RuntimeError` instead of a zero-division `IndexError` when no ≥700-token review is found. |
| CI imports (IPython) | **Fixed** | `IPython.display` lazy-imported inside methods in `evaluator.py`/`plotter.py` (commits `858a8e7`, `648996d`); `tests/test_cleanlab_denoiser.py` uses `pytest.importorskip("cleanlab")`. |
| Lint (ruff, CI rule) | **Pass** | `ruff check src tests` clean (verified locally with ruff 0.16.2). |
| Tests | **Pass** | `pytest` → 15 passed / 1 skipped (cleanlab absent locally; skips via `importorskip`). |
| Notebook header policy | **Fixed** | `notebooks/02_ex2_finetune.ipynb` roadmap column renamed to `Import path` and the mandatory `## References` block added per [NOTEBOOK_HEADER_CONVENTION.md](rules/NOTEBOOK_HEADER_CONVENTION.md). |
| Report metric consistency | **Fixed** | `EX14` comparison table row for EX-09 reconciled to EX-09's own report (22,164 Clean / 92.35% / F1 0.9235 / AUC 0.9753) — it previously conflated EX-08's numbers. |
| Results & overview docs | **Fixed** | [`experiments/results/README.md`](../experiments/results/README.md) re-indexed to the EX-14 eval JSON and the new Cleanlab/truncation artifacts; [`OVERVIEW.md`](OVERVIEW.md) updated to the 93.14% EX-14 milestone; `FOLDER_STRUCTURE.md` lists the new modules. |
| Data leakage guard | **Pass** | 5-Fold OOF auditing guarantees each train sample is scored by a model that never saw it; val/test splits sealed. |
| Data-centric integrity | **Annotated** | Cleanlab `find_label_issues` returns probabilities whose tokenization now matches training (see first row). |

**Residual risks (non-blocking, tracked):**

- **R1 — audit count mismatch:** the committed [`cleanlab_label_issues.json`](../experiments/results/cleanlab_label_issues.json) lists **707** candidate issues (3.14%), while the EX-13/EX-14 reports and the exported `imdb_denoised_512` (22,388 clean = 112 pruned) reference a higher-confidence subset. The exact pruned index list is not yet reproducible from the artifact alone; recommend recording the pruned indices (or a confidence threshold in `audit_label_errors`/`export_denoised_dataset`) so the 112 can be traced.
- **R2 — stray artifact:** a duplicate `notebooks/experiments/results/head_tail_truncation_viz.png` was committed (generated with CWD=notebooks). Plots belong under `experiments/plots|results`; remove the stray copy.
- **R3 — committed binary caches:** `imdb_train_pred_probs.npy` / `imdb_oof_train_pred_probs.npy` (~180 KB each) are regenerable script outputs; committing them is consistent with the repo's results-committing practice but inflates git — consider `.gitignore`-ing the `.npy` caches going forward.
- **R4 — report format debt:** EX-07…EX-14 reports carry 5W1H but do not fully follow [MD_CONVENTION.md](rules/MD_CONVENTION.md) (5-field header + TOC); paths/HF IDs are occasionally bare text.
- **R5 — hardcoded viz constants:** `truncation_viz.py` coverage/waste figures (e.g. `[9.48, 55.66, 86.24, 100.0]`, `18.4` tokens) are presentation constants, not derived from data.

**Verdict:** the merged PR #16 content is code-clean, lint-green, test-green,
and consistent with the logging/checkpoint, 5W1H, and notebook-header rules.
All review findings raised on the PR were remediated in `ae178d5` (now merged
via `b9c4b1e`); the residual items above are documentation/artifact-practice
follow-ups, none blocking the submission.

