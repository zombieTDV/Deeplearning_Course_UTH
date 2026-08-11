# OVERVIEW.md — Project Overview & Roadmap

- **Motivation/Background**: A single living overview keeps the project's
  plan, phases, and status visible without re-reading every phase doc.
- **Purpose**: Summarize Practice 3 — dataset, models, approach, phases,
  known constraints, and pointers to status docs.
- **Overview Pipeline**: Derived from [PURPOSE.md](PURPOSE.md) and
  [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md); updated at the start of every
  session/phase.
- **Detailed Plan**: §1 project; §2 plan; §3 phases; §4 known constraints;
  §5 progress pointers.
- **References**: `PURPOSE.md`, `PROJECT_ROADMAP.md`,
  `templates/PROJECT_ROADMAP_TEMPLATE.md`, `phases/<PHASE>.md`.

---

## Project

Practice 3 — Get Started with Hugging Face: Exercise 1 (zero-shot sentiment
analysis with a pretrained HF pipeline) and Exercise 2 (finetune
`distilbert-base-uncased` on IMDB for binary sentiment classification),
delivered as graded coursework.

## Purpose

See [PURPOSE.md](PURPOSE.md) for the original brief and locked objective. The
execution plan lives in [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).

## Plan

- **Dataset:** IMDB (Hugging Face `datasets`), binary sentiment
  (positive/negative), 25k train + 25k test reviews, balanced 50/50.
- **Models:** Ex 1 — pretrained HF sentiment pipeline (zero-shot baseline);
  Ex 2 — `distilbert-base-uncased` finetuned with the HF `Trainer`.
- **Approach:** baseline-first ([§11 of ML_PIPELINE_REFERENCE_v3.md](ML_PIPELINE_REFERENCE_v3.md#11-baseline-thinking))
  → tokenize/split with leakage rules ([§10](ML_PIPELINE_REFERENCE_v3.md#10-train--test-split-and-data-leakage))
  → finetune → single held-out evaluation ([§16](ML_PIPELINE_REFERENCE_v3.md#16-evaluation-metrics))
  with per-class metrics → error analysis ([§19](ML_PIPELINE_REFERENCE_v3.md#19-error-analysis))
  → 5W1H report.
- **Monitoring:** [run_logger.py](../src/utils/run_logger.py) live progress +
  JSONL history + optional TensorBoard (`--tb`); artifacts auto-persisted to
  `experiments/runs/<ts>_<run>/`.
- **Success criteria:** All steps of both exercises run end-to-end from
  scripts; finetuned model evaluated on the IMDB test split with accuracy +
  per-class metrics reported under full 5W1H; run artifacts auto-persisted per
  [rules/LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md).

## Phases

1. [phases/SETUP.md](phases/SETUP.md) — environment, HF stack, problem framing
2. [phases/DATA_PREP.md](phases/DATA_PREP.md) — IMDB EDA, cleaning/imbalance N/A checks
3. [phases/FEATURE_SPLIT.md](phases/FEATURE_SPLIT.md) — tokenization, train/val/test split, leakage rules
4. [phases/BASELINE.md](phases/BASELINE.md) — Ex 1 zero-shot baseline + majority-class floor
5. [phases/MODEL.md](phases/MODEL.md) — model selection rationale, training config
6. [phases/TRAINING_INFO.md](phases/TRAINING_INFO.md) — Trainer finetuning, checkpoints, resume
7. [phases/EVAL.md](phases/EVAL.md) — test evaluation, metrics, optional seed experiments
8. [phases/REPORT.md](phases/REPORT.md) — error analysis, notebook, final 5W1H report

## Known constraints

- GPU with 8 GB VRAM — target ≤6 GB usage (ceiling 8 GB); limits batch size
  and `max_length`.
- English-only sample sentences.
- HF dependencies (`transformers`, `datasets`, `evaluate`) must be added to
  `requirements.txt` and pinned.
- Training runs only from scripts; notebooks load artifacts for analysis only.
- K-Fold CV and hyperparameter sweeps are out of scope (documented N/A).

## Progress

- [progress/SETUP_STATUS.md](progress/SETUP_STATUS.md)
- [progress/DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md)
- [progress/FEATURE_SPLIT_STATUS.md](progress/FEATURE_SPLIT_STATUS.md)
- [progress/BASELINE_STATUS.md](progress/BASELINE_STATUS.md)
- [progress/MODEL_STATUS.md](progress/MODEL_STATUS.md)
- [progress/TRAINING_INFO_STATUS.md](progress/TRAINING_INFO_STATUS.md)
- [progress/EVAL_STATUS.md](progress/EVAL_STATUS.md)
- [progress/REPORT_STATUS.md](progress/REPORT_STATUS.md)
