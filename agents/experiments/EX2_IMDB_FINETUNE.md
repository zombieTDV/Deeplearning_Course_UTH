# EX2_IMDB_FINETUNE.md — Feature Plan (agents/experiments)

---

## Header

- **Title:** Exercise 2 — Finetuning `distilbert-base-uncased` on IMDB for Binary Sentiment Classification
- **Date created:** 2026-08-12
- **Last updated:** 2026-08-12
- **Description:** Plan for implementing Exercise 2: script-only HF `Trainer` finetuning of `distilbert-base-uncased` on the IMDB binary sentiment dataset, with full-state checkpoints, resume, logging, single held-out evaluation, and final 5W1H report.
- **Status:** Done
- **Experiment ID:** EXP-EX2-FINETUNE
- **Branch:** `feature/ex2-imdb-finetune`

---

## Hugging Face Hub Resources Used

- **Base Model (finetune target):** [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- **Dataset:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Reference floor (from Ex 1):** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) — zero-shot test accuracy 89.07%

---

## Objective

Deliver Exercise 2 end-to-end per the coursework brief and repo rules:

1. Verify `transformers`, `datasets`, `evaluate` (already installed in Phase 1).
2. Justify `distilbert-base-uncased` model choice (No Free Lunch, §12) and pin training hyperparameters in `configs/config_imdb_sentiment.yaml`.
3. Implement the script-only finetuning CLI `src/training/imdb_sentiment_train.py` wrapping HF `Trainer` + `TrainingArguments` (Exercise 2 steps 5–6).
4. Persist full-state checkpoints (`<run>_best.pt`, `<run>_last.pt`), logs, config, and history JSONL to `experiments/runs/<ts>_<run>/` with resume support (`--resume`, `--force-resume`).
5. Smoke-test on a tiny subset; verify VRAM ≤3.5 GB (ceiling 4 GB).
6. Evaluate the finetuned model exactly once on the held-out IMDB test split (accuracy, confusion matrix, per-class precision/recall/F1) with full 5W1H.
7. Error analysis, demo notebook, and final 5W1H report; update OVERVIEW/indexes.

---

## Single Variable Changed / Held Constant

- **Changed:** Fine-tuning of `distilbert-base-uncased` on IMDB (supervised) vs the Ex 1 zero-shot pretrained pipeline floor (89.07% test accuracy).
- **Held Constant:** IMDB train/test splits (25k/25k from [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)), binary sentiment labels, evaluation metric (accuracy as headline + per-class P/R/F1), evaluation device (GPU), seed within a run. Single-variable principle for any experiment delta (Task T14).

---

## Step-by-Step Implementation Plan

### Step 1: Model Selection & Training Config (Phase 5, Tasks T8–T9)
- Document model rationale in `agents/phases/MODEL.md`: DistilBERT ≈ 66M params, ~40% smaller than BERT-base with near-parity accuracy; fits ≤3.5 GB VRAM on 4 GB team machines.
- Define training args with bias-variance reasoning: lr ≈ 2e-5, epochs 2–3, batch size tuned to VRAM (e.g. 16 with gradient accumulation), weight decay (AdamW default 0.01), `max_length` from Phase 3.
- Write `configs/config_imdb_sentiment.yaml`; every value logged with every run (§18.3 single-variable principle).

### Step 2: Implement Finetuning CLI (Phase 6, Task T10)
- Path: `src/training/imdb_sentiment_train.py`
- Features:
  - CLI flags: `--epochs`, `--seed`, `--resume`, `--force-resume`, `--tb`, `--smoke`.
  - HF `Trainer` + `TrainingArguments` finetuning on tokenized IMDB (Exercise 2 steps 5–6).
  - Full-state checkpoints (model, optimizer, scheduler, RNG, history, best metrics, config, timestamp, commit) per `LOGGING_CHECKPOINT_RULES.md`.
  - `_last.pt` written every epoch (resume source); `_best.pt` on best tracked metric.
  - Resume loads `_last.pt` and continues at `epoch + 1`; `--force-resume` rewinds from `_best.pt`; checkpoints load with `weights_only=True`.
  - fp16 + gradient accumulation to keep VRAM ≤3.5 GB (risk R1).

### Step 3: Smoke Test & Full Run (Phase 6, Tasks T11–T12)
- Run smoke test on a tiny subset per `templates/SMOKE_TEST_CHECKLIST.md`; verify VRAM ≤3.5 GB.
- Launch full finetune on GPU (produces the Exercise 2 finetuned model).
- Artifacts auto-persisted to `experiments/runs/<ts>_<run>/`: `checkpoints/`, `logs/`, `metrics/`, optional `tensorboard/`.

### Step 4: Evaluation & Validation (Phase 7, Tasks T13–T14)
- Path: `src/eval/evaluate_model.py`
- Load `<run>_best.pt` (`weights_only=True`); rebuild model + tokenizer; verify `id2label`/`label2id` (risk R6).
- Run inference on the held-out IMDB test split exactly once; compute accuracy, confusion matrix, per-class precision/recall/F1.
- Report with full 5W1H per `RESULTS_REPORTING.md`; save `experiments/results/imdb_sentiment_eval.json` + confusion-matrix plot in `experiments/plots/`.
- Optional: single-variable experiment (one lr or epoch delta) and μ±σ across 2–3 seeds; K-Fold CV explicitly N/A.

### Step 5: Error Analysis, Interpretability & Reporting (Phase 8, Tasks T15–T18)
- Extract misclassified reviews; group by root cause (label noise, rare patterns, long reviews, etc.); prioritize by frequency × severity.
- Local interpretability: sample predictions with confidence scores.
- Demo notebook `notebooks/01_imdb_sentiment_analysis.ipynb` (header per `NOTEBOOK_HEADER_CONVENTION.md`; loads artifacts only, no training).
- Final 5W1H report with results vs baseline delta; update `OVERVIEW.md`, `experiments/results/README.md`, `experiments/runs/registry.json`.

---

## Verification & Smoke Tests

- `python -m src.training.imdb_sentiment_train --smoke` (tiny subset, VRAM ≤3.5 GB check).
- `python -m src.training.imdb_sentiment_train --epochs 3 --seed 42 --tb` (full run).
- `python -m src.training.imdb_sentiment_train --resume` (exact resume from `_last.pt`).
- `python -m src.eval.evaluate_model --checkpoint experiments/runs/<ts>_<run>/checkpoints/<run>_best.pt`.
- Run unit/smoke tests with `python -m pytest tests/ -v`.
- Run codebase audit per `rules/CODEBASE_AUDIT.md` before marking each phase done.

---

## Results Table

| Metric | Ex 1 Zero-Shot Baseline | Ex 2 Finetuned Model | Delta / Notes |
|---|---|---|---|
| IMDB Test Accuracy | 89.07% | **91.26%** | **+2.19% improvement** (22,815/25,000 correct) |
| Confusion Matrix | N/A | `[[11268, 1232], [953, 11547]]` | Balanced error distribution (TN=11,268, TP=11,547) |
| Precision / Recall / F1 (per class) | N/A | Neg: 0.922 / 0.901 / 0.912<br>Pos: 0.904 / 0.924 / 0.914 | Macro F1 = 0.9126 |
| Peak VRAM | N/A | **1,583 MB allocated / 1,696 MB reserved** | **Passed ≤3.5 GB target** |
| Training Time | N/A | 12m 30s (3 epochs, 4,221 steps) | Best checkpoint at Epoch 1 (step 1407) |

---

## Links & References

- Branch: `feature/ex2-imdb-finetune`
- Training script: [src/training/imdb_sentiment_train.py](../../src/training/imdb_sentiment_train.py)
- Eval script: [src/eval/evaluate_model.py](../../src/eval/evaluate_model.py)
- Config: [configs/config_imdb_sentiment.yaml](../../configs/config_imdb_sentiment.yaml)
- Phase docs: [../phases/MODEL.md](../phases/MODEL.md), [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md), [../phases/EVAL.md](../phases/EVAL.md), [../phases/REPORT.md](../phases/REPORT.md)
- Status docs: [../progress/MODEL_STATUS.md](../progress/MODEL_STATUS.md), [../progress/TRAINING_INFO_STATUS.md](../progress/TRAINING_INFO_STATUS.md), [../progress/EVAL_STATUS.md](../progress/EVAL_STATUS.md), [../progress/REPORT_STATUS.md](../progress/REPORT_STATUS.md)
- Roadmap: [../PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md)
- Rules: [../rules/LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md), [../rules/RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md), [../rules/PYTORCH_FRAMEWORK_RULES.md](../rules/PYTORCH_FRAMEWORK_RULES.md), [../rules/MD_CONVENTION.md](../rules/MD_CONVENTION.md)
