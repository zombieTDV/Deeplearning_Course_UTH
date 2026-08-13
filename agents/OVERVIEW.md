# OVERVIEW.md — Project Overview & Roadmap

- **Motivation/Background**: A single living overview keeps the project's plan, phases, and status visible without re-reading every phase doc.
- **Purpose**: Summarize Practice 3 — dataset, models, approach, phases, known constraints, and pointers to status docs.
- **Overview Pipeline**: Derived from [PURPOSE.md](PURPOSE.md) and [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md); updated at the start of every session/phase.
- **Detailed Plan**: §1 project; §2 plan; §3 phases; §4 known constraints; §5 progress pointers.
- **References**: `PURPOSE.md`, `PROJECT_ROADMAP.md`, `templates/PROJECT_ROADMAP_TEMPLATE.md`, `phases/<PHASE>.md`.

**Last Updated:** 2026-08-13 (Breakthrough Milestone: **93.23% Test Accuracy** achieved with EXP-06 512 Tokens Expansion)

---

## Project

Practice 3 — Get Started with Hugging Face: Exercise 1 (zero-shot sentiment analysis with a pretrained HF pipeline) and Exercise 2 (finetune `distilbert-base-uncased` on IMDB for binary sentiment classification), delivered as graded coursework.

## Purpose

See [PURPOSE.md](PURPOSE.md) for the original brief and locked objective. The execution plan lives in [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).

## Plan

- **Dataset:** IMDB (Hugging Face `datasets`), binary sentiment (positive/negative), 25k train + 25k test reviews, balanced 50/50.
- **Models:**
  - **Ex 1 Zero-Shot Baseline:** Pretrained HF sentiment pipeline (`distilbert-base-uncased-finetuned-sst-2-english`), achieving **89.07% Test Accuracy** and **0.9587 ROC-AUC**.
  - **Ex 2 Finetuned Model:** `distilbert-base-uncased` finetuned with HF `Trainer` + LLRD + SWA + 512 tokens sequence length expansion (`EXP-06`), achieving **93.23% Test Accuracy**, **0.9323 Macro F1**, and **0.9742 ROC-AUC**.
- **Approach:** Baseline-first ([§11 of ML_PIPELINE_REFERENCE_v3.md](ML_PIPELINE_REFERENCE_v3.md#11-baseline-thinking)) → tokenize/split with leakage rules → finetune → single held-out evaluation with per-class metrics → error analysis → 5W1H report.
- **Modular Code Base:** Fully modularized `src/` package (`src/models`, `src/data`, `src/eval`, `src/training`, `src/utils`).
- **Interactive Notebook Suite:**
  - [`notebooks/01_ex1_sentiment_baseline.ipynb`](../notebooks/01_ex1_sentiment_baseline.ipynb): Zero-Shot Baseline Exercise 1 (89.07% Accuracy).
  - [`notebooks/02_ex2_finetune.ipynb`](../notebooks/02_ex2_finetune.ipynb): Fine-Tuning Exercise 2 & Presets `EXP-00` to `EXP-07` (93.23% Accuracy).

## Major Phases Status

1. [phases/SETUP.md](phases/SETUP.md) — environment, HF stack, problem framing — **Done**
2. [phases/DATA_PREP.md](phases/DATA_PREP.md) — IMDB EDA, cleaning/imbalance N/A checks — **Done**
3. [phases/FEATURE_SPLIT.md](phases/FEATURE_SPLIT.md) — tokenization, train/val/test split, leakage rules — **Done**
4. [phases/BASELINE.md](phases/BASELINE.md) — Ex 1 zero-shot baseline + majority-class floor — **Done**
5. [phases/MODEL.md](phases/MODEL.md) — model selection rationale, training config — **Done**
6. [phases/TRAINING_INFO.md](phases/TRAINING_INFO.md) — Trainer finetuning, LLRD, SWA, checkpoints — **Done**
7. [phases/EVAL.md](phases/EVAL.md) — test evaluation, metrics, 25k sealed test set evaluation — **Done**
8. [phases/REPORT.md](phases/REPORT.md) — error analysis, notebooks, final 5W1H report — **Done**


## Known Constraints & Rules
- **VRAM Ceiling:** GPU memory target $\le 3.5\text{ GB}$ (Hard ceiling $4.0\text{ GB}$). Achieved $2.14\text{ GB}$ peak VRAM.
- **Training Rules:** Agents must NEVER run background training scripts without user consent. Training is interactively executed by the user via Jupyter notebooks.
- **Persisted Artifacts:** All evaluation metrics auto-persisted to `experiments/results/imdb_sentiment_eval.json`.
