# EX3_HYPERPARAMETER_SEARCH_ANTI_OVERFITTING.md — Experiment Specification

---

## Header

- **Experiment Title:** Exercise 2 Hyperparameter Tuning & Advanced Anti-Overfitting Search
- **Experiment ID:** `EXP-EX2-HYPERPARAMETER-SEARCH`
- **Created**: 2026-08-12T00:00:00+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00
- **Author / Role:** bush-le + AI agent → Coursework submission
- **Status:** Planned
- **Target Component:** [`src/training/imdb_sentiment_train.py`](../../src/training/imdb_sentiment_train.py), [`configs/config_imdb_sentiment.yaml`](../../configs/config_imdb_sentiment.yaml)

---

## 1. Executive Summary & Objective

The objective of this experimental suite is to systematically eliminate validation loss divergence (overfitting) observed during DistilBERT finetuning on IMDB, identify the optimal hyperparameter combination ($lr$, weight decay, classifier dropout, layer freezing), and maximize held-out test accuracy ($\ge 92.00\%$) while maintaining strict VRAM budget ($\le 3.5\text{ GB}$).

---

## 2. Baseline & Current Performance Floor

| Metric | Ex 1 Zero-Shot | Ex 2 Initial Run (`152704`) | Ex 2 Anti-Overfitting Run (`175932`) | Target Goal (`EXP-01` to `EXP-05`) |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 89.07% | 89.32% | **91.19%** | **$\ge 92.00\%$** |
| **ROC-AUC** | 0.9587 | — | **0.9699** | **$\ge 0.9750$** |
| **Min Val Loss** | N/A | 0.2584 (Ep 1) | **0.2671 (Step 1200)** | **$< 0.2400$** |
| **Val Loss at Ep 4** | N/A | 0.4161 (Ep 3) | 0.4819 (Ep 4) | **$< 0.2800$ (Stable plateau)** |
| **Peak VRAM** | < 1.0 GB | 1.7 GB | 264.62 MB | **$\le 3.5\text{ GB}$** |

---

## 3. Experimental Grid & Controlled Variables

Each trial applies the **Single-Variable Principle** (Pipeline §18):

```
┌─────────────────────────────────────────────────────────────┐
│ EXP-01: Early Stopping Gate (patience=3)                   │
├─────────────────────────────────────────────────────────────┤
│ EXP-02: Cosine vs Linear LR Scheduler                      │
├─────────────────────────────────────────────────────────────┤
│ EXP-03: High Weight Decay Sweep (0.05, 0.10, 0.15)         │
├─────────────────────────────────────────────────────────────┤
│ EXP-04: Classifier Dropout Regularization (0.20, 0.30)     │
├─────────────────────────────────────────────────────────────┤
│ EXP-05: Bottom 2-Layer Freezing (66M -> 25M Params)         │
└─────────────────────────────────────────────────────────────┘
```

| Trial ID | Hypothesized Effect | Parameter Variable | Fixed Controls | Target Deliverable |
|:---:|:---|:---|:---|:---|
| `EXP-01` | Early stopping halts training at min `val_loss` before overfitting begins | `early_stopping_patience: 3` | $lr=1.5e-5$, $wd=0.05$, $ep=4$ | Automatically saved `<run>_best.pt` |
| `EXP-02` | Cosine annealing smooths LR decay, reducing loss oscillation | `lr_scheduler_type: "cosine"` | $lr=1.5e-5$, $wd=0.05$, patience=3 | Loss curve comparison plot |
| `EXP-03` | Higher $L_2$ weight decay penalizes extreme weights | `weight_decay: [0.05, 0.10, 0.15]` | $lr=1.5e-5$, patience=3 | Generalization gap analysis |
| `EXP-04` | Higher dropout adds noise to classifier head, preventing memorization | `seq_classif_dropout: [0.2, 0.3]` | $lr=1.5e-5$, $wd=0.10$ | Macro F1 comparison |
| `EXP-05` | Freezing lower layers preserves general feature representations | `freeze_bottom_layers: 2` | $lr=2.0e-5$, $wd=0.05$ | VRAM & speed benchmark |

---

## 4. Execution Workflow & Verification Protocol

1. **CLI Script Extensions** ([`src/training/imdb_sentiment_train.py`](../../src/training/imdb_sentiment_train.py)):
   - Implement CLI arguments: `--early-stopping-patience`, `--lr-scheduler-type`, `--weight-decay`, `--classifier-dropout`, `--freeze-layers`.
2. **Automated Trial Execution**:
   - Run trials sequentially via script; save checkpoints, logs, and configs to `experiments/runs/<ts>_<exp_name>/`.
3. **Sealed Test Set Evaluation**:
   - Run [`src/eval/evaluate_model.py`](../../src/eval/evaluate_model.py) on each trial's `<run>_best.pt`.
4. **5W1H Results Indexing**:
   - Register all metrics in [`experiments/results/README.md`](../../experiments/results/README.md).
5. **Notebook Integration**:
   - Render multi-experiment loss curves and comparison table in [`notebooks/02_ex2_finetune.ipynb`](../../notebooks/02_ex2_finetune.ipynb).

---

## 5. Related Knowledge Base Links

- Master Plan: [`agents/plans/PLAN_EX2_HYPERPARAMETER_SEARCH_AND_ANTI_OVERFITTING.md`](../plans/PLAN_EX2_HYPERPARAMETER_SEARCH_AND_ANTI_OVERFITTING.md)
- Training Rules: [`agents/rules/LOGGING_CHECKPOINT_RULES.md`](../rules/LOGGING_CHECKPOINT_RULES.md)
- Reporting Rules: [`agents/rules/RESULTS_REPORTING.md`](../rules/RESULTS_REPORTING.md)
- Experiment Index: [`agents/experiments/README.md`](README.md)
