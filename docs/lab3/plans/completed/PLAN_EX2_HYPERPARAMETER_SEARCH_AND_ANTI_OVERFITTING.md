**Status:** [DONE] Completed & Archived

# PLAN: EX2 Systematic Hyperparameter Tuning & Advanced Anti-Overfitting Search

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---


**Goal**: Establish a systematic experimental protocol to eliminate overfitting in Exercise 2 (finetuning `distilbert-base-uncased` on IMDB), discover optimal hyperparameters, enforce early stopping, and maximize generalization accuracy while maintaining strict VRAM budget ($\le 3.5\text{ GB}$).

---

## 1. Baseline vs Current Status Diagnosis

| Metric / Aspect | Initial Baseline Run (`152704`) | Anti-Overfitting Run (`175932`) | Target Goal for Experiment Sweep |
|---|---|---|---|
| **Learning Rate ($lr$)** | $2.0 \times 10^{-5}$ | $1.5 \times 10^{-5}$ | $1.0 \times 10^{-5}$ to $3.0 \times 10^{-5}$ (Grid / Optuna) |
| **Weight Decay** | $0.01$ | $0.05$ | $0.05$ – $0.15$ ($L_2$ regularization) |
| **Classifier Dropout** | $0.10$ | $0.20$ | $0.20$ – $0.30$ |
| **Early Stopping** | None | None (4 fixed epochs) | `EarlyStoppingCallback(patience=3)` on `eval_loss` |
| **Layer Freezing** | None (All 6 layers trained) | None | Freeze Bottom 2/4 Layers vs Full Finetuning |
| **Best Val Loss** | $0.2584$ (Ep 1) $\rightarrow$ $0.4161$ (Ep 3) | $0.2671$ (Step 1200) $\rightarrow$ $0.4819$ (Ep 4) | $< 0.2400$ (Stable plateau) |
| **Test Accuracy** | $89.32\%$ | **$91.19\%$** | **$\ge 92.00\%$** |

---

## 2. Experimental Grid & Controlled Variables

Following **Single-Variable & Controlled Search Principles** (Pipeline §18), experiments are structured into 4 distinct phases:

```
[Phase A: Early Stopping & Scheduler] 
          │
          ▼
[Phase B: Regularization & Dropout Sweep]
          │
          ▼
[Phase C: Layer Freezing & LLRD Experiment]
          │
          ▼
[Phase D: Final Candidate Evaluation & Verification]
```

### Experiment Matrix (Single-Variable Grid)

| Exp ID | Hypothesis / Objective | Variable under test | Control parameters | Metrics recorded |
|:---:|:---|:---|:---|:---|
| **`EXP-01`** | **Early Stopping Gate**: Prevent training past minimum `eval_loss` | Add `EarlyStoppingCallback(patience=3)` | $lr=1.5e-5$, $wd=0.05$, $ep=4$ | Min `eval_loss`, Step stopped, Test Acc |
| **`EXP-02`** | **LR Scheduler Impact**: Cosine decay vs Linear warmup | `lr_scheduler_type`: `cosine` vs `linear` | $lr=1.5e-5$, $wd=0.05$, patience=3 | Loss curve smoothness, Val Acc |
| **`EXP-03`** | **High Weight Decay**: Penalize overconfident weight norms | `weight_decay`: `[0.05, 0.10, 0.15]` | $lr=1.5e-5$, patience=3 | Train/Val loss gap, Generalization gap |
| **`EXP-04`** | **Classifier Dropout Sweep**: Extra noise on classification head | `attention_dropout`/`seq_classif_dropout`: `[0.2, 0.3]` | $lr=1.5e-5$, $wd=0.10$ | Val F1, Overfitting step delta |
| **`EXP-05`** | **Bottom Layer Freezing**: Reduce trainable parameters ($66M \rightarrow 25M$) | Freeze Embeddings + Transformer Layers 0–2 | $lr=2.0e-5$, $wd=0.05$ | VRAM usage, Training speed, Test Acc |

---

## 3. Step-by-Step Implementation Steps

### Step 1: Upgrade Training CLI (`src/training/imdb_sentiment_train.py`)
1. Add `EarlyStoppingCallback(early_stopping_patience=args.patience)` to HF `Trainer`.
2. Add CLI flags:
   - `--early-stopping-patience` (default: 3)
   - `--lr-scheduler-type` (`linear`, `cosine`)
   - `--freeze-layers` (number of bottom transformer layers to freeze, e.g. 0, 2, 4)
   - `--classifier-dropout` (dropout rate for sequence classifier head)
3. Ensure run logging records all experimental hyperparameters into `metrics/<run_name>_config.json`.

### Step 2: Automated Experiment Runner Script (`scratch/run_experiment_grid.py`)
Create a lightweight runner script to execute the 5 experiments sequentially:
```bash
python -m src.training.imdb_sentiment_train --run-name exp01-early-stop --patience 3
python -m src.training.imdb_sentiment_train --run-name exp02-cosine-decay --lr-scheduler-type cosine --patience 3
python -m src.training.imdb_sentiment_train --run-name exp03-wd-010 --weight-decay 0.10 --patience 3
python -m src.training.imdb_sentiment_train --run-name exp04-dropout-030 --weight-decay 0.10 --classifier-dropout 0.30 --patience 3
python -m src.training.imdb_sentiment_train --run-name exp05-freeze-bottom2 --freeze-layers 2 --patience 3
```

### Step 3: Comparative Evaluation & Selection
1. Evaluate each experiment's best checkpoint (`<run>_best.pt`) on the sealed IMDB test set using `src/eval/evaluate_model.py`.
2. Generate comparative 5W1H metrics table in `experiments/results/imdb_experiments_summary.json`.

### Step 4: Visual Comparison & Notebook Update
1. Create multi-curve overlay plot (`experiments/plots/exp_loss_comparison.png`) comparing `val_loss` across all 5 experiment runs.
2. Update `notebooks/02_ex2_finetune.ipynb` with an Experiment Comparison Section displaying the head-to-head metrics table and overlay plot.

---

## 4. Verification & Constraints Checklist

- [ ] **VRAM Constraint**: All runs must maintain peak VRAM $\le 3.5\text{ GB}$ (monitored via `ResourceMonitor`).
- [ ] **Script-Only Execution**: All experiment runs launched via CLI scripts; notebooks remain read-only consumers.
- [ ] **Reproducibility**: Explicit seed (`seed: 42`) set for all runs.
- [ ] **RNG & Resumability**: `ckpt["rng"]` restored on resume.
- [ ] **Audit Gate**: Run Step-10 codebase audit (`CODEBASE_AUDIT.md`) after search completion.
