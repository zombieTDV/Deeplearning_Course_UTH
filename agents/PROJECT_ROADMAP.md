# PROJECT_ROADMAP.md — Practice 3: Get Started with Hugging Face

- **Motivation/Background**: The locked brief in
  [PURPOSE.md](PURPOSE.md) needs a living plan that makes milestones,
  phases, tasks, dependencies, resources, and timeline visible in one place,
  and that follows the course's ML pipeline reference so every stage is
  accounted for (or explicitly marked N/A).
- **Purpose**: A single roadmap for Practice 3 — Exercise 1 (zero-shot
  sentiment analysis with a Hugging Face pipeline) and Exercise 2 (finetune
  `distilbert-base-uncased` on IMDB) — generated from the roadmap template and
  structured per the pipeline stages in
  [ML_PIPELINE_REFERENCE_v3.md](ML_PIPELINE_REFERENCE_v3.md).
- **Overview Pipeline**: Clarifying interview locked [PURPOSE.md](PURPOSE.md)
  → this roadmap maps the brief onto the ML pipeline's 14 stages → phase docs
  in `phases/` are written per phase (Step 5 of
  [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md)).
- **Detailed Plan**: §1 legend & usage; §2 overview + pipeline alignment;
  §3 milestones; §4 phases; §5 task breakdown; §6 dependencies & critical
  path; §7 resource allocation; §8 timeline & Gantt; §9 risks; §10
  maintenance; appendix self-check.
- **References**: `PURPOSE.md`, `ML_PIPELINE_REFERENCE_v3.md`,
  `templates/PROJECT_ROADMAP_TEMPLATE.md`, `rules/LOGGING_CHECKPOINT_RULES.md`,
  `rules/RESULTS_REPORTING.md`.

**Last updated:** 2026-08-11

---

## Table of Contents

1. [How to Use This Roadmap](#1-how-to-use-this-roadmap)
2. [Project Overview](#2-project-overview)
   - [Pipeline alignment](#pipeline-alignment)
3. [Key Milestones](#3-key-milestones)
4. [Major Phases](#4-major-phases)
5. [Task Breakdown](#5-task-breakdown)
6. [Task Dependencies](#6-task-dependencies)
7. [Resource Allocation](#7-resource-allocation)
8. [Estimated Timeline](#8-estimated-timeline)
   - [8.1 Phase timeline](#81-phase-timeline)
   - [8.2 ASCII Gantt (weeks)](#82-ascii-gantt-weeks)
9. [Risks & Mitigations](#9-risks--mitigations)
10. [Maintenance & Status](#10-maintenance--status)
- [Appendix — Self-Check Checklist](#appendix--self-check-checklist)

---

## 1. How to Use This Roadmap

- **Status legend** (used everywhere below):
  `[ ]` = not started, `[~]` = in progress, `[X]` = done, `[!]` = blocked/on hold.
- **Effort units**: **d** = person-days, **w** = person-weeks.
- **Update the date stamp** in §10 every time this file is touched, and keep
  the "Last updated" line near the top current.
- **Phase docs** referenced in §4 are created one per phase (Step 5 of
  [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md)) from
  [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md); progress is
  tracked in `progress/<PHASE>_STATUS.md`.
- This roadmap follows the course pipeline in
  [ML_PIPELINE_REFERENCE_v3.md](ML_PIPELINE_REFERENCE_v3.md) stage by
  stage; the [Pipeline alignment](#pipeline-alignment) table shows where each
  of the 14 pipeline stages lands (or why it is N/A for this pretrained-NLP
  project).

---

## 2. Project Overview

| Field | Value |
|---|---|
| Project name | Practice 3 — Get Started with Hugging Face (sentiment analysis & finetuning) |
| Problem / motivation | Graded coursework. Without it, no hands-on Hugging Face experience: Hub model loading, tokenization, and the Trainer finetuning workflow. See [PURPOSE.md §2](PURPOSE.md#2-clarifying-answers). |
| Goal (1 sentence) | Complete both exercises end-to-end: run zero-shot sentiment analysis with a pretrained HF pipeline (Ex 1), and finetune `distilbert-base-uncased` on IMDB with the HF `Trainer`, then evaluate (Ex 2). |
| Success criteria | All steps of both exercises run end-to-end from scripts; finetuned model evaluated on the IMDB test split with accuracy (plus confusion matrix and per-class precision/recall/F1) reported under full 5W1H; run artifacts auto-persisted per [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md); no hard accuracy threshold. |
| Non-goals (out of scope) | Other datasets/models, non-binary classification, deployment/serving, hyperparameter sweeps, K-Fold CV, non-English samples. See [PURPOSE.md §2](PURPOSE.md#2-clarifying-answers). |
| Sponsor / product owner | Course instructor (coursework submission) |
| Start date | 2026-08-11 |
| Target end date | 2026-09-07 *(assumed — no deadline given in the brief; adjust in §10)* |
| Key constraints | Multi-GPU team hardware (8 GB + 4 GB machines): target ≤3.5 GB VRAM usage (ceiling 4 GB); English-only samples; HF dependencies (`transformers`, `datasets`, `evaluate`) pinned in `requirements.txt`; script-only training (no training in notebooks). |

### Pipeline alignment

Each of the 14 pipeline stages from the instructor's summary in
[ML_PIPELINE_REFERENCE_v3.md](ML_PIPELINE_REFERENCE_v3.md) maps to a
phase/task below. Stages that do not apply to a pretrained-NLP project are
marked **N/A** with a documented rationale instead of being silently skipped.

| # | Pipeline stage | Reference § | Where it lands in this roadmap | Status |
|---|---|---|---|---|
| 1 | EDA | §3 | Phase 2 — Task T3 | `[ ]` |
| 2 | Cleaning | §4, §5 | Phase 2 — Task T4 (**N/A** for IMDB: pre-cleaned; rationale documented) | `[ ]` |
| 3 | Feature Engineering | §6, §7, §9 | Phase 3 — Task T5 (tokenization = DL-equivalent FE; scaling/encoding **N/A**) | `[ ]` |
| 4 | Data Split | §10 | Phase 3 — Task T6 (leakage golden rules applied) | `[ ]` |
| 5 | Baseline Model | §11 | Phase 4 — Task T7 (Ex 1 zero-shot pipeline + majority-class floor) | `[ ]` |
| 6 | Model Selection | §12 | Phase 5 — Task T8 (`distilbert-base-uncased`, No-Free-Lunch justification) | `[ ]` |
| 7 | Hyperparameter Tuning | §14, §15 | Phase 5 — Task T9 (training args with bias-variance reasoning; no sweep) | `[ ]` |
| 8 | Train Model | — | Phase 6 — Tasks T10–T12 (Trainer finetuning, script-only) | `[ ]` |
| 9 | Evaluation Metrics | §16 | Phase 7 — Task T13 (accuracy, confusion matrix, per-class P/R/F1) | `[ ]` |
| 10 | Cross-Validation | §17 | Phase 7 — Task T14 (**N/A** for coursework: single hold-out; μ±σ across seeds optional) | `[ ]` |
| 11 | Experiments Management | §18 | Phase 7 — Task T14 (single-variable principle, logged configs) | `[ ]` |
| 12 | Statistical Validation | §18.4 | Phase 7 — Task T14 (optional μ±σ across 2–3 seeds) | `[ ]` |
| 13 | Error Analysis | §19 | Phase 8 — Task T15 (root-cause grouping of misclassified reviews) | `[ ]` |
| 14 | Model Interpretability | §20 | Phase 8 — Task T16 (local: sample predictions + confidence) | `[ ]` |

---

## 3. Key Milestones

> A milestone is a *point in time* with an externally verifiable deliverable.

| Milestone | Target date | Deliverable / definition of done | Status | Owner |
|---|---|---|---|---|
| M1 — Environment & framing | 2026-08-13 | HF stack installed & GPU verified (≤3.5 GB target); [SETUP.md](phases/SETUP.md) written; problem framing locked | `[X]` | Learner + AI agent |
| M2 — Ex 1 baseline | 2026-08-17 | Zero-shot sentiment pipeline runs on sample English sentences; baseline + majority-class metrics recorded (reference floor) | `[X]` | AI agent |
| M3 — Data pipeline ready | 2026-08-21 | IMDB loaded, tokenized, split (train/val/test) verified; [FEATURE_SPLIT.md](phases/FEATURE_SPLIT.md) documents leakage rules | `[ ]` | AI agent |
| M4 — Finetuned model | 2026-08-28 | Full finetune run complete; `experiments/runs/<ts>_<run>/` contains checkpoints (`<run>_best.pt`, `<run>_last.pt`), logs, config, history JSONL | `[ ]` | Learner (launch) |
| M5 — Evaluation done | 2026-09-01 | Accuracy + confusion matrix + per-class P/R/F1 reported with full 5W1H; [EVAL.md](phases/EVAL.md) written | `[ ]` | AI agent |
| M6 — Deliverables | 2026-09-04 | Scripts, demo notebook, final report; [OVERVIEW.md](OVERVIEW.md) and `experiments/results/README.md` index updated; submission-ready | `[ ]` | Learner + AI agent |

---

## 4. Major Phases

> Phases mirror the pipeline stages (see [Pipeline alignment](#pipeline-alignment));
> top-to-bottom by workflow; parallel phases noted in "Input (from)".

| Phase | Description | Pipeline § | Input (from) | Output (feeds) | Phase doc | Status |
|---|---|---|---|---|---|---|
| Phase 1 — Setup & Problem Framing | Install HF stack, verify GPU/VRAM, lock paradigm (supervised binary sentiment classification) | Steps 1–2 | `requirements.txt`, GPU | Verified environment; [SETUP.md](phases/SETUP.md) | [SETUP.md](phases/SETUP.md) | `[X]` |
| Phase 2 — Data Survey & Cleaning | EDA on IMDB (label balance, review lengths, samples); confirm cleaning & imbalance stages are N/A | §3–§5, §8 | IMDB via `datasets` | EDA stats + N/A rationale in [DATA_PREP.md](phases/DATA_PREP.md) | [DATA_PREP.md](phases/DATA_PREP.md) | `[X]` |
| Phase 3 — Feature Engineering & Split | Tokenization (FE for DL); IMDB train/test + validation carve; leakage golden rules | §6–§7, §9–§10 | IMDB dataset | Tokenized, split datasets; preprocessing verified | [FEATURE_SPLIT.md](phases/FEATURE_SPLIT.md) | `[ ]` |
| Phase 4 — Baseline Model | Ex 1 zero-shot pipeline on sample sentences + majority-class floor; record baseline metrics | §11 | Phase 1 env | Baseline metrics (reference floor for deltas) | [BASELINE.md](phases/BASELINE.md) | `[X]` |
| Phase 5 — Model Selection & Config | Justify `distilbert-base-uncased`; define training args with bias-variance reasoning | §12–§15 | Phase 2–3 data outputs | [MODEL.md](phases/MODEL.md); `configs/config_imdb_sentiment.yaml` | [MODEL.md](phases/MODEL.md) | `[ ]` |
| Phase 6 — Train Model | Finetune via HF `Trainer` in a script-only CLI with full-state checkpoints, resume, logging | §15 | Phase 5 config | Run artifacts in `experiments/runs/<ts>_<run>/` | [TRAINING_INFO.md](phases/TRAINING_INFO.md) | `[ ]` |
| Phase 7 — Evaluation & Validation | One held-out test evaluation: accuracy, confusion matrix, per-class P/R/F1; single-variable experiments; CV N/A | §16–§18 | Phase 6 checkpoints | Metrics with 5W1H; experiment deltas | [EVAL.md](phases/EVAL.md) | `[ ]` |
| Phase 8 — Error Analysis, Interpretability & Reporting | Root-cause grouping of errors; local interpretability; demo notebook; final 5W1H report | §18–§20 | Phase 7 metrics | Notebook, report, updated indexes | [REPORT.md](phases/REPORT.md) | `[ ]` |

---

## 5. Task Breakdown

> Granular units of work. Effort is a best-effort estimate. Priority:
> **P0** (blocker/critical), **P1** (high), **P2** (medium), **P3** (nice-to-have).

| Task ID | Phase | Description (what + acceptance) | Owner / role | Effort | Dependencies | Priority | Status |
|---|---|---|---|---|---|---|---|
| `T1` | Phase 1 | Install HF stack (`transformers`, `datasets`, `evaluate`) into `.venv`; pin versions in `requirements.txt`; verify imports + GPU/VRAM check (≤3.5 GB target, 4 GB ceiling). **Accept:** imports pass, VRAM reported | Learner | `0.5d` | `—` | `P0` | `[X]` |
| `T2` | Phase 1 | Lock problem framing (supervised binary sentiment classification; Ex 1 baseline vs Ex 2 finetune per [PURPOSE.md](PURPOSE.md)) and write [SETUP.md](phases/SETUP.md). **Accept:** reviewed and approved | AI agent (draft) + Learner (review) | `0.5d` | `T1` | `P0` | `[X]` |
| `T3` | Phase 2 | EDA on IMDB via `datasets`: label balance, review-length stats, sample inspection; record in [DATA_PREP.md](phases/DATA_PREP.md) (pipeline §3). **Accept:** EDA numbers recorded | AI agent | `0.5d` | `T2` | `P1` | `[X]` |
| `T4` | Phase 2 | Confirm cleaning & imbalance stages: IMDB pre-cleaned (missing values/outliers **N/A**) and balanced 50/50 — verify + document rationale (§4–§5, §8). **Accept:** N/A rationale documented | AI agent | `0.5d` | `T3` | `P2` | `[X]` |
| `T5` | Phase 3 | Tokenization pipeline: load `distilbert-base-uncased` tokenizer, map IMDB with padding/truncation (`max_length`); document as DL-equivalent feature engineering (§6–§7, §9). **Accept:** `Dataset.map` runs on train + test | AI agent | `1d` | `T4` | `P0` | `[ ]` |
| `T6` | Phase 3 | Split handling: use IMDB built-in train/test; carve validation split for the `Trainer`; document leakage golden rules (tokenizer is static; test evaluated once, §10). **Accept:** splits verified, no test statistics in preprocessing | AI agent | `0.5d` | `T5` | `P0` | `[ ]` |
| `T7` | Phase 4 | Ex 1 zero-shot baseline: script [baseline_imdb_sentiment.py](../src/experiments/baseline_imdb_sentiment.py) runs HF sentiment pipeline on sample sentences + majority-class baseline on IMDB test; record metrics as reference floor (§11). **Accept:** baseline metrics recorded with 5W1H | AI agent | `0.5d` | `T2` | `P0` | `[X]` |
| `T8` | Phase 5 | Model selection rationale in [MODEL.md](phases/MODEL.md): `distilbert-base-uncased` justified against alternatives on compute/VRAM/accuracy (No Free Lunch, §12). **Accept:** rationale documented | AI agent | `0.5d` | `T3`, `T6` | `P1` | `[ ]` |
| `T9` | Phase 5 | Define training args (learning rate, epochs, batch size, weight decay, `max_length`) with bias-variance reasoning; write `configs/config_imdb_sentiment.yaml` (§13–§15). **Accept:** every hyperparameter logged with each run | AI agent + Learner | `0.5d` | `T8` | `P0` | `[ ]` |
| `T10` | Phase 6 | Finetuning CLI [imdb_sentiment_train.py](../src/training/imdb_sentiment_train.py): HF `Trainer`, full-state checkpoints (`<run>_best.pt`, `<run>_last.pt`), `--epochs/--seed/--resume/--force-resume/--tb/--smoke` per [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md). **Accept:** resume + auto-persistence implemented | AI agent | `2d` | `T6`, `T9` | `P0` | `[ ]` |
| `T11` | Phase 6 | Smoke test finetuning on a tiny subset per [SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md); verify VRAM ≤3.5 GB. **Accept:** smoke passes, VRAM reported | AI agent | `0.5d` | `T10` | `P0` | `[ ]` |
| `T12` | Phase 6 | Full finetune run on GPU; artifacts (checkpoints, logs, config, history JSONL) auto-persisted to `experiments/runs/<ts>_<run>/`. **Accept:** complete run dir per logging rules | Learner (launch) + AI agent (monitor) | `1d` | `T11` | `P0` | `[ ]` |
| `T13` | Phase 7 | Evaluation script [evaluate_model.py](../src/eval/evaluate_model.py): accuracy (primary), confusion matrix, per-class precision/recall/F1 on the held-out test set, evaluated once (§16; balanced classes → ROC-AUC optional). **Accept:** metrics reported with 5W1H per [RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md) | AI agent | `1d` | `T12` | `P0` | `[ ]` |
| `T14` | Phase 7 | Single-variable experiments (e.g. one lr or epoch delta) with optional μ±σ across 2–3 seeds; K-Fold CV explicitly **N/A** and documented (§17–§18). **Accept:** one variable changed per run; deltas vs baseline | Learner | `1d` | `T13` | `P2` | `[ ]` |
| `T15` | Phase 8 | Error analysis: extract misclassified reviews, group by root cause (label noise, rare patterns, etc.) per §19.1; record in [EVAL.md](phases/EVAL.md). **Accept:** root-cause groups documented | AI agent + Learner | `0.5d` | `T13` | `P1` | `[ ]` |
| `T16` | Phase 8 | Local interpretability: sample predictions + confidence review in notebook (§20). **Accept:** example predictions shown | AI agent | `0.5d` | `T13` | `P3` | `[ ]` |
| `T17` | Phase 8 | Demo/analysis notebook `notebooks/01_imdb_sentiment_analysis.ipynb` with header per [NOTEBOOK_HEADER_CONVENTION.md](rules/NOTEBOOK_HEADER_CONVENTION.md); loads artifacts only, no training loop. **Accept:** runs standalone | AI agent | `1d` | `T15`, `T16` | `P1` | `[ ]` |
| `T18` | Phase 8 | Final report with 5W1H results; update [OVERVIEW.md](OVERVIEW.md), `experiments/results/README.md` index, `experiments/runs/registry.json`. **Accept:** submission-ready | Learner + AI agent | `1d` | `T13`, `T15`, `T16`, `T17` | `P0` | `[ ]` |

**Total estimated effort:** ≈ 13.5 person-days.

---

## 6. Task Dependencies

> Meaningful edges only — a task cannot start until its dependencies finish.

| Wait | Depends on | Reason |
|---|---|---|
| `T2` | `T1` | Environment must exist before framing/setup doc can be verified |
| `T3` | `T2` | EDA needs the installed `datasets` library |
| `T4` | `T3` | Cleaning/balance checks run on EDA findings |
| `T5` | `T4` | Tokenization operates on the surveyed dataset |
| `T6` | `T5` | Split/leakage rules apply to the tokenized dataset |
| `T7` | `T2` | Ex 1 baseline needs only the environment (parallel branch to data prep) |
| `T8` | `T3`, `T6` | Model choice is justified by EDA + split design |
| `T9` | `T8` | Training args derive from the chosen model |
| `T10` | `T6`, `T9` | Finetuning CLI consumes tokenized splits and training config |
| `T11` | `T10` | Smoke test runs the finetuning script on a subset |
| `T12` | `T11` | Full run starts only after smoke passes |
| `T13` | `T12` | Evaluation needs the finetuned checkpoints |
| `T14` | `T13` | Experiment deltas are measured with the evaluation pipeline |
| `T15` | `T13` | Error analysis consumes the evaluation's misclassified samples |
| `T16` | `T13` | Interpretability examples use evaluated predictions |
| `T17` | `T15`, `T16` | Notebook presents error analysis + interpretability |
| `T18` | `T13`, `T15`, `T16`, `T17` | Final report consolidates evaluation + analysis + notebook |

**Critical path hint:** the longest dependency chain is
`T1 → T2 → T3 → T4 → T5 → T6 → T8 → T9 → T10 → T11 → T12 → T13 → T15 → T17 → T18`.
This chain determines the minimum project duration and is reflected in the §8
dates. `T7` (baseline) runs in parallel off `T2`; `T14` and `T16` are optional
and do not block the critical path.

---

## 7. Resource Allocation

> "Allocation" = % of a person's time in the relevant period. One human +
> one AI agent; no over-allocation (single human at 50% part-time).

| Person / role | Skills | Allocation | Period | Primary tasks |
|---|---|---|---|---|
| Learner (you) | Python, PyTorch (growing), ML pipeline, HF (learning) | 50% (part-time coursework) | 2026-08-11 → 2026-09-07 | `T1`, `T2` (review), `T9`, `T12`, `T14`, `T18`; reviews all agent output |
| AI coding agent | Python/HF refactoring, docs per repo rules, audits | On-demand (≈100% during working sessions) | 2026-08-11 → 2026-09-07 | `T2`–`T18` support; owns script/doc drafts and [CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md) before phase completion |

> Skills gap note: learner's HF familiarity is the gap this practice closes;
> the AI agent scaffolds the Hugging Face APIs while the learner owns decisions
> and the final submission.

---

## 8. Estimated Timeline

> Derived from §5 effort × §7 allocation, sequenced by §6 dependencies. Calendar
> spans assume the 50% part-time allocation (≈13.5 person-days ≈ 4 weeks).

### 8.1 Phase timeline

| Phase / milestone | Start | End | Duration |
|---|---|---|---|
| Phase 1 — Setup & Problem Framing | 2026-08-11 | 2026-08-13 | 2d |
| M1 — Environment & framing | 2026-08-13 | — | — |
| Phase 2 — Data Survey & Cleaning | 2026-08-14 | 2026-08-18 | 3d |
| Phase 4 — Baseline Model (parallel) | 2026-08-14 | 2026-08-17 | 2d |
| M2 — Ex 1 baseline | 2026-08-17 | — | — |
| Phase 3 — Feature Engineering & Split | 2026-08-18 | 2026-08-21 | 3d |
| M3 — Data pipeline ready | 2026-08-21 | — | — |
| Phase 5 — Model Selection & Config | 2026-08-21 | 2026-08-24 | 2d |
| Phase 6 — Train Model | 2026-08-24 | 2026-08-28 | 4d |
| M4 — Finetuned model | 2026-08-28 | — | — |
| Phase 7 — Evaluation & Validation | 2026-08-28 | 2026-09-01 | 3d |
| M5 — Evaluation done | 2026-09-01 | — | — |
| Phase 8 — Error Analysis & Reporting | 2026-09-01 | 2026-09-04 | 3d |
| M6 — Deliverables | 2026-09-04 | — | — |
| Buffer | 2026-09-04 | 2026-09-07 | 3d |

### 8.2 ASCII Gantt (weeks)

```
Week               W1          W2          W3          W4
             Aug 11–17   Aug 18–24   Aug 25–31   Sep 01–07
Phase 1       ██
Phase 2       ███
Phase 4       ███
Phase 3                   ███
Phase 5                     ███
Phase 6                        ████
Phase 7                            ███
Phase 8                                ███
M6 (deliverables)                               ◆
```

---

## 9. Risks & Mitigations

> Likelihood (L/M/H) × Impact (L/M/H) → priority. A materialized risk moves to
> §5 as a task.

| Risk | Likelihood | Impact | Priority | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 — VRAM exceeds 3.5 GB during finetuning on 4 GB team machines | M | M | High | Gradient accumulation + small batch size (8–16); fp16; monitor with `nvidia-smi` during smoke test (`T11`); fall back to shorter `max_length` or smaller subset | Learner |
| R2 — IMDB download blocked (network/firewall) | M | H | High | Use `datasets` cache; set `HF_ENDPOINT` mirror; pre-cache to `data/external/`; last resort: documented smaller subset | AI agent |
| R3 — HF API drift / version incompatibility (`Trainer`, `pipeline` signatures) | M | M | Medium | Pin versions in `requirements.txt` at `T1`; follow docs matching pinned versions; smoke test before full run | AI agent |
| R4 — Finetuned model underperforms or overfits vs baseline | M | M | Medium | Compare to baseline floor (`T7`); bias-variance diagnosis of train/val gap (§13); adjust one hyperparameter at a time (`T14`) | Learner |
| R5 — Coursework timeline slip (part-time) | M | M | Medium | 3-day buffer in §8; smoke-test-first discipline; cut `T14`/`T16` (optional) if needed without touching critical path | Learner |
| R6 — Label/`id2label` mismatch producing inverted or wrong predictions | L | H | Medium | Verify `id2label`/`label2id` in eval script; run a 3-sample sanity check in the notebook before reporting | AI agent |

---

## 10. Maintenance & Status

- **Last updated:** 2026-08-12
- **Update cadence:** per milestone (per [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md) Step 8–10)
- **Who updates it:** Learner, with AI agent drafting updates
- **Progress note:** Phases 1, 2, and 4 completed via PR #13 (merge `3f982ee`, 2026-08-12); next unstarted phase: Phase 3 — Feature Engineering & Split (T5–T6).

> **How to keep it alive:** at each review, (1) tick off completed
> milestones/tasks, (2) move new work into §5 with an ID, (3) re-check §6
> dependencies and the critical path, (4) re-derive §8 dates, (5) refresh §9
> risk likelihoods, (6) bump the date above. Progress lives per-phase in
> `progress/<PHASE>_STATUS.md`; run the codebase audit
> ([CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md)) **before** marking a
> phase done; smoke tests ([SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md))
> gate long runs. A roadmap that isn't updated is a fiction — keep it living.

**Open assumptions** (to revisit at M1 review): target end date 2026-09-07 is
assumed (no deadline in the brief); `distilbert-base-uncased` + IMDB locked per
[PURPOSE.md §3](PURPOSE.md#3-locked-objective); English-only samples;
K-Fold CV and hyperparameter sweeps remain out of scope.

---

## Appendix — Self-Check Checklist

- [X] Overview §2 is specific (success criterion measurable: end-to-end runs + reported accuracy with 5W1H)
- [X] Every milestone has a verifiable definition of done (M1–M6)
- [X] Every task has a unique ID referenced by §6 and §7 (T1–T18)
- [X] The critical path (§6) is identified and reflected in §8 dates
- [X] No person is over-allocated (§7 — single learner at 50%)
- [X] Top risks have named owners and mitigations (§9 — R1–R6)
- [X] "Last updated" is current (§10)
- [X] Every pipeline stage in [ML_PIPELINE_REFERENCE_v3.md](ML_PIPELINE_REFERENCE_v3.md) is mapped or explicitly marked N/A (see [Pipeline alignment](#pipeline-alignment))
