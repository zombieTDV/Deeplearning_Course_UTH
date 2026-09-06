# EX1_SENTIMENT_BASELINE.md — Feature Plan (agents/experiments)

---

## Header

- **Title:** Exercise 1 — Zero-Shot Sentiment Analysis Baseline
- **Created**: 2026-08-12T00:00:00+07:00
- **Last Updated**: 2026-08-12T00:00:00+07:00
- **Description:** Plan for implementing Exercise 1: zero-shot Hugging Face sentiment analysis pipeline, tokenization demonstration, and baseline evaluation on IMDB test set.
- **Status:** Done
- **Experiment ID:** EXP-EX1-BASELINE
- **Branch:** `feature/ex1-sentiment-baseline`

---

## Hugging Face Hub Resources Used

- **Pretrained Pipeline Model:** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- **Target Finetuning Base Model:** [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- **Dataset:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)

---

## Objective

Deliver Exercise 1 end-to-end according to coursework brief and project rules:
1. Verify Hugging Face dependencies (`transformers`, `datasets`, `evaluate`).
2. Implement a zero-shot sentiment analysis CLI script (`src/experiments/baseline_imdb_sentiment.py`).
3. Demonstrate single-sentence tokenization & pipeline prediction on sample text.
4. Evaluate zero-shot accuracy on IMDB test split alongside majority-class baseline floor.
5. Auto-persist results to `experiments/results/baseline_imdb_sentiment.json` with 5W1H context.

---

## Single Variable Changed / Held Constant

- **Changed:** Pretrained zero-shot Hugging Face model ([`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)) vs Majority Baseline Floor.
- **Held Constant:** IMDB test split (25,000 samples from [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)), evaluation metric (Accuracy), evaluation device (GPU).

---

## Step-by-Step Implementation Plan

### Step 1: Environment & Dependency Verification (Task T1)
- Verify `transformers`, `datasets`, `evaluate`, `scikit-learn`, `torch` are functional in `.venv`.
- Confirm VRAM usage constraint: Target ≤3.5GB (ceiling 4GB for compatibility across team hardware).

### Step 2: Write Baseline CLI Script (Task T7)
- Path: `src/experiments/baseline_imdb_sentiment.py`
- Features:
  - Argument parsing: `--samples` (sample text input), `--eval-imdb` (run test set evaluation), `--batch-size` (default 32), `--max-samples` (optional evaluation cap).
  - Exercise 1 demo: Tokenize sample sentence, print tokens & IDs, run `pipeline("sentiment-analysis")`.
  - Majority-class baseline calculator: Calculate accuracy if always predicting majority label.
  - Test set zero-shot evaluator: Batch predict on [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) test set with label mapping.

### Step 3: Result Auto-Persistence & 5W1H Logging
- Output path: `experiments/results/baseline_imdb_sentiment.json`
- Metadata: Date/timestamp, model ID, evaluation split size, zero-shot test accuracy, majority-class accuracy, hardware/environment metadata.
- Index entry added to `experiments/results/README.md`.

### Step 4: Verification & Smoke Test
- Run `python -m src.experiments.baseline_imdb_sentiment --samples "This movie was fantastic!"`
- Run unit/smoke tests with `python -m pytest tests/ -v`.

---

## Results Table

| Metric | Majority Baseline | Zero-Shot HF Pipeline ([`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)) |
|---|---|---|
| IMDB Test Accuracy | 50.00% | **89.07%** (22,268 / 25,000 correctly classified) |
| **ROC-AUC Score** | 0.5000 | **0.9587** |
| ROC Curve Plot | N/A | Saved to [`experiments/plots/baseline_zero_shot_roc_curve.png`](../../experiments/plots/baseline_zero_shot_roc_curve.png) |
| PyTorch TensorBoard Logs | N/A | Saved to [`experiments/runs/baseline_zero_shot/tensorboard/`](../../experiments/runs/baseline_zero_shot/tensorboard/) |
| Total Eval Time (GPU) | N/A | 274.77 seconds |
| Sample Inference Latency | N/A | 380.49 ms |

---

## Links & References

- Branch: `feature/ex1-sentiment-baseline`
- Code Script: [src/experiments/baseline_imdb_sentiment.py](../../src/experiments/baseline_imdb_sentiment.py)
- Results Artifact: [experiments/results/baseline_imdb_sentiment.json](../../experiments/results/baseline_imdb_sentiment.json)
- Phase doc: [../phases/BASELINE.md](../phases/BASELINE.md)
- Status doc: [../progress/BASELINE_STATUS.md](../progress/BASELINE_STATUS.md)
- Rules: [../rules/RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md), [../rules/PYTORCH_FRAMEWORK_RULES.md](../rules/PYTORCH_FRAMEWORK_RULES.md), [../rules/MD_CONVENTION.md](../rules/MD_CONVENTION.md)
