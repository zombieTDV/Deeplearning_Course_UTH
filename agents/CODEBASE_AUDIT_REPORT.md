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

| ID | Area | Severity | Title | Section |
|---|---|---|---|---|
| `ARC-1` | Architecture | **High** | Documented `src/` modules entirely absent (docs claim code that doesn't exist) | [6. Architecture](#6-architecture-consistency) |
| `DEP-1` | Dependencies | **High** | `transformers` not installed; environment vs `requirements.txt` drift | [5. Dependency Health](#5-dependency-health) |
| `TST-1` | Tests | **High** | pytest not installed — smoke suite cannot execute | [7. Test Coverage](#7-test-coverage) |
| `AQ-1` | Code quality | **Medium** | `run_logger.py` documented in two layers (ownership ambiguity) | [3. Code Quality](#3-code-quality) |
| `DEP-2` | Dependencies | **Medium** | No version pins / no lockfile (reproducibility) | [5. Dependency Health](#5-dependency-health) |
| `ARC-2` | Architecture | **Medium** | `FOLDER_STRUCTURE.md` not updated for `PROJECT_ROADMAP.md` + new phase docs | [6. Architecture](#6-architecture-consistency) |
| `TST-2` | Tests | **Medium** | Only smoke tests; no unit tests (modules don't exist yet) | [7. Test Coverage](#7-test-coverage) |
| `AQ-2` | Code quality | **Low** | Unused `needs_data` marker in [conftest.py](../tests/conftest.py) | [3. Code Quality](#3-code-quality) |
| `AQ-3` | Code quality | **Low** | `agents/progress/*_STATUS.md` referenced by OVERVIEW + 8 phase docs but absent | [3. Code Quality](#3-code-quality) |
| `ARC-3` | Architecture | **Low** | Two phase-doc templates coexist (`templates/PHASE_DOC_TEMPLATE.md` vs `phases/PHASE_TEMPLATE.md`) | [6. Architecture](#6-architecture-consistency) |
| `AQ-4` | Code quality | Info | Root [README.md](../README.md) still carries template placeholders | [3. Code Quality](#3-code-quality) |
| `SEC-1` | Security | Info | No application code in tree — minimal attack surface (positive) | [4. Security](#4-security-vulnerabilities) |
| `SEC-2` | Security | Info | No secrets; `.gitignore` excludes data, checkpoints, `.env` (positive) | [4. Security](#4-security-vulnerabilities) |
| `PERF-1` | Performance | Info | No training/eval code yet; constraints already documented (positive) | [8. Performance](#8-performance-bottlenecks) |

Severity totals: **High × 3, Medium × 4, Low × 3, Info × 4.** Cross-reference:
[risk analysis §10](#10-detailed-risk-analysis) and
[compliance §9](#9-compliance-with-policies-and-procedures).

---

## 3. Code Quality

### `AQ-1`: `run_logger.py` documented in two layers — ownership ambiguity
- **Severity:** Medium
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

### `AQ-2`: Unused `needs_data` marker in conftest
- **Severity:** Low
- **Description:** [conftest.py](../tests/conftest.py) defines a `needs_data`
  skipif marker, but no test references it ([test_smoke.py](../tests/test_smoke.py)
  never applies it). It is dead scaffolding today.
- **Affected:** [tests/conftest.py](../tests/conftest.py).
- **Remediation:** either start applying it to data-dependent tests (planned
  Phase 2/3 tests) or delete it. — tracked in [Action P2.1](#12-prioritized-action-plan)

### `AQ-3`: Progress status files referenced but absent
- **Severity:** Low
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

### `AQ-4`: Root README template placeholders
- **Severity:** Info
- **Description:** [README.md](../README.md) still contains `[PROJECT_NAME]`
  and `[Fill in — license…]` placeholders from the bootstrap template.
- **Affected:** [README.md](../README.md).
- **Remediation:** fill in the project name and license/course context.
  — tracked in [Action P2.3](#12-prioritized-action-plan)

---

## 4. Security Vulnerabilities

### `SEC-1`: No application code in tree — minimal attack surface (positive)
- **Severity:** Info
- **Description:** `src/` contains only package `__init__.py` files; there is
  no model, training, or evaluation code to contain injection, RCE, or data
  leaks yet. The one security-relevant requirement (checkpoints load with
  `weights_only=True`) is already mandated in
  [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) §3.
- **Affected:** entire `src/` tree.
- **Remediation:** none now — enforce `weights_only=True` when
  `checkpoint_utils.py` is implemented. — tracked in [Action P0.2](#12-prioritized-action-plan)

### `SEC-2`: No secrets; `.gitignore` excludes sensitive paths (positive)
- **Severity:** Info
- **Description:** No credentials/tokens found in tracked files; `.gitignore`
  covers `.env*`, `/data/raw/`, `/experiments/checkpoints/*.pt`, and
  `/experiments/runs/`, and `.kilo` config is ignored.
- **Affected:** [.gitignore](../.gitignore).
- **Remediation:** none.

---

## 5. Dependency Health

### `DEP-1`: `transformers` missing; environment vs `requirements.txt` drift
- **Severity:** High
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

### `DEP-2`: No version pins / no lockfile
- **Severity:** Medium
- **Description:** [requirements.txt](../requirements.txt) lists bare package
  names; the repo references a `requirements.lock` (pip freeze) that does not
  exist. The roadmap's T1 pins HF versions, but nothing is pinned today.
- **Affected:** [requirements.txt](../requirements.txt).
- **Remediation:** pin top-level versions and add `requirements.lock`.
  — tracked in [Action P1.4](#12-prioritized-action-plan)

---

## 6. Architecture Consistency

### `ARC-1`: Documented `src/` modules entirely absent
- **Severity:** High
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

### `ARC-2`: `FOLDER_STRUCTURE.md` not updated for new governance files
- **Severity:** Medium
- **Description:** [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) does not
  list `agents/PROJECT_ROADMAP.md` (it describes OVERVIEW.md as the roadmap),
  and lists only example phase docs — the 8 phase files added this session
  ([phases/SETUP.md](phases/SETUP.md), `DATA_PREP.md`, `FEATURE_SPLIT.md`,
  `BASELINE.md`, `MODEL.md`, `TRAINING_INFO.md`, `EVAL.md`, `REPORT.md`) are
  undocumented there.
- **Affected:** [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md).
- **Remediation:** update the structure listing (with human approval per the
  rule's own constraint). — tracked in [Action P1.1](#12-prioritized-action-plan)

### `ARC-3`: Two phase-doc templates coexist
- **Severity:** Low
- **Description:** [templates/PHASE_DOC_TEMPLATE.md](templates/PHASE_DOC_TEMPLATE.md)
  and [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md) both claim to be the
  phase-doc template; the 8 phase docs were created from the latter. Two
  templates will drift.
- **Affected:** [templates/PHASE_DOC_TEMPLATE.md](templates/PHASE_DOC_TEMPLATE.md),
  [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md).
- **Remediation:** designate one canonical template and remove or alias the
  other. — tracked in [Action P2.2](#12-prioritized-action-plan)

---

## 7. Test Coverage

### `TST-1`: pytest not installed — smoke suite cannot execute
- **Severity:** High
- **Description:** `python -m pytest` fails in both the `.venv` and system
  Python ("No module named pytest"), despite pytest being declared in
  [requirements.txt](../requirements.txt). [test_smoke.py](../tests/test_smoke.py)
  therefore cannot run, so the smoke-gating discipline
  ([SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md)) is
  unenforceable today.
- **Affected:** [tests/](../tests/), environment.
- **Remediation:** install pytest into the active environment (and any CI) —
  included in [Action P0.1](#12-prioritized-action-plan).

### `TST-2`: Only smoke tests; no unit tests
- **Severity:** Medium
- **Description:** The entire suite is
  [test_smoke.py](../tests/test_smoke.py) (3 tests: imports, directory
  existence, knowledge-base files). No unit tests exist for the modules the
  roadmap plans (`prepare_imdb.py`, `imdb_sentiment_train.py`,
  `evaluate_model.py`), because those modules don't exist yet.
- **Affected:** [tests/](../tests/).
- **Remediation:** add a unit test alongside each module as its phase
  implements it (roadmap Phases 3, 6, 7). — tracked in [Action P0.2](#12-prioritized-action-plan)

---

## 8. Performance Bottlenecks

### `PERF-1`: No training/eval code yet — performance is N/A (positive)
- **Severity:** Info
- **Description:** With no implementation, there are no hot paths to profile.
  The constraints that will shape performance are already documented:
  `num_workers = 0` safety note ([src/data/README.md](../src/data/README.md)),
  ≤6 GB VRAM target and gradient-accumulation mitigation (roadmap risk R1,
  [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)).
- **Affected:** none.
- **Remediation:** none now; revisit after Phase 6 produces the first run.

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

