# OVERVIEW.md — Project Overview & Roadmap

- **Motivation/Background**: A single living overview keeps the project's plan, phases, and status visible without re-reading every phase doc.
- **Purpose**: Summarize Practice 3 — dataset, models, approach, phases, known constraints, and pointers to status docs.
- **Overview Pipeline**: Derived from [PURPOSE.md](PURPOSE.md) and [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md); updated at the start of every session/phase.
- **Detailed Plan**: §1 project; §2 plan; §3 phases; §4 known constraints; §5 progress pointers.
- **References**: `PURPOSE.md`, `PROJECT_ROADMAP.md`, `templates/PROJECT_ROADMAP_TEMPLATE.md`, `phases/<PHASE>.md`.

**Last Updated:** 2026-08-15 (Peak Milestone: **93.14% Test Accuracy** — EX-14 LoRA PEFT on Cleanlab-denoised IMDB with Head+Tail Truncation; builds on the EXP-06 512-token breakthrough of **93.23%**)

---

## Project

Practice 3 — Get Started with Hugging Face: Exercise 1 (zero-shot sentiment analysis with a pretrained HF pipeline) and Exercise 2 (finetune `distilbert-base-uncased` on IMDB for binary sentiment classification), delivered as graded coursework.

## Purpose

See [PURPOSE.md](PURPOSE.md) for the original brief and locked objective. The execution plan lives in [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).

## Plan

- **Dataset:** IMDB (Hugging Face `datasets`), binary sentiment (positive/negative), 25k train + 25k test reviews, balanced 50/50.
- **Models:**
  - **Ex 1 Zero-Shot Baseline:** Pretrained HF sentiment pipeline (`distilbert-base-uncased-finetuned-sst-2-english`), achieving **89.07% Test Accuracy** and **0.9587 ROC-AUC**.
  - **Ex 2 Finetuned Model (current):** `distilbert-base-uncased` + PEFT LoRA ($r=32, \alpha=64$) trained on the Cleanlab-denoised split with Head+Tail Truncation ($128+384=512$) — EX-14, achieving **93.14% Test Accuracy**, **0.9314 Macro F1**, and **0.9662 ROC-AUC** at **1.11 GB** peak VRAM (committed in [`experiments/results/imdb_sentiment_eval.json`](../experiments/results/imdb_sentiment_eval.json), run `20260815_173320_distilbert-finetune-lora-denoised`).
  - **Ex 2 Historical Milestones:** EXP-06/EXP-07 full fine-tuning with 512-token expansion reached **93.23% Test Accuracy** / 0.9742 ROC-AUC (see [EX6_512_TOKENS_BREAKTHROUGH_REPORT.md](experiments/EX6_512_TOKENS_BREAKTHROUGH_REPORT.md)); EX-08 Cleanlab denoising (**92.66%**) and EX-09 LoRA rank-32 (**92.35%**) preceded the EX-13 Head+Tail upgrade and the EX-14 peak.
- **Approach:** Baseline-first ([§11 of ML_PIPELINE_REFERENCE_v3.md](ML_PIPELINE_REFERENCE_v3.md#11-baseline-thinking)) → tokenize/split with leakage rules → finetune → single held-out evaluation with per-class metrics → error analysis → 5W1H report. **Data-centric AI:** label-error pruning via an independent 5-Fold Out-Of-Fold (OOF) Cleanlab audit ([`src/data/cleanlab_denoiser.py`](../src/data/cleanlab_denoiser.py)) exports the pristine 22,388-sample `imdb_denoised_512` split; **Head+Tail Truncation** (`head_tail_tokenize()`, 128 head + 384 tail tokens) preserves reviewer verdicts on reviews > 512 tokens.
- **Modular Code Base:** Fully modularized `src/` package (`src/models`, `src/data`, `src/eval`, `src/training`, `src/utils`) with lazy PEP 562 exports in [`src/__init__.py`](../src/__init__.py).
- **Interactive Notebook Suite:**
  - [`notebooks/01_ex1_sentiment_baseline.ipynb`](../notebooks/01_ex1_sentiment_baseline.ipynb): Zero-Shot Baseline Exercise 1 (89.07% Accuracy).
  - [`notebooks/02_ex2_finetune.ipynb`](../notebooks/02_ex2_finetune.ipynb): Exercise 2 end-to-end workflow — presets `EXP-00` to `EXP-07`, `EXP-LORA-DENOISED`; EDA, HTML-noise cleaning, Head+Tail truncation visualization, 5-Fold OOF Cleanlab audit, LoRA training, sealed-test evaluation, SWA inference, and misclassification audit (peak: 93.14% Accuracy).

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
- **VRAM Ceiling:** GPU memory target $\le 3.5\text{ GB}$ (Hard ceiling $4.0\text{ GB}$). Peak achieved: **1.11 GB** (EX-14 LoRA) — well within budget.
- **Training Rules:** Agents must NEVER run background training scripts without user consent. Training is interactively executed by the user via Jupyter notebooks.
- **Persisted Artifacts:** All evaluation metrics auto-persisted to `experiments/results/imdb_sentiment_eval.json`; Cleanlab audit artifacts to `experiments/results/cleanlab_label_issues.json` and `data/processed/imdb_denoised_512/`.
