# EXPERIMENT REPORT: EX-04 Anti-Overfitting Tuning Results & Failure Mode Diagnostics

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---


**Document ID:** `EX4-RESULTS-AND-DIAGNOSTICS-REPORT`  
**Date:** 2026-08-13  
**Author:** bush-le + Antigravity AI Pair Programmer  
**Dataset:** IMDB Movie Reviews (25,000 train / 25,000 sealed test)  
**Model Architecture:** `distilbert-base-uncased` (66.9M parameters)  
**Status:** Completed & Diagnosed  

---

## 📌 Executive Summary

This report documents the empirical evaluation of Experiment EX-04 (Tuned Anti-Overfitting Recipe with Early Stopping, Cosine Scheduler, Weight Decay 0.10, Classifier Dropout 0.30, and Label Smoothing 0.10) compared against the Baseline Fine-Tuned Model (EX-00 / 91.19% Accuracy).

### Key Finding
When evaluating the optimal checkpoint selected at Step 2700 (Epoch 1.92) / Step 3000 (Epoch 2.13), the EX-04 tuned model achieves **91.12% Test Accuracy** and **0.9112 Macro F1**, restoring performance back to baseline levels ($91.19\%$).

Diagnostic investigation confirmed the **Early Stopping Metric Mismatch & Recovery**:
- Previously, early stopping monitored `eval_loss` with patience 3, halting prematurely at Step 1200 (Epoch 0.85) where `eval_accuracy` was only $88.32\%-90.34\%$.
- Switching metric selection to `eval_accuracy` / `eval_f1` allows training to progress into Epoch 2.0+, reaching peak validation performance ($90.52\%$ val acc, $0.9049$ val F1) and **91.12% test accuracy**.
- **Notebook 02 Visualization Update (Cell 4):** Chart 2 has been updated to feature a dual-axis trajectory plotting both **Validation Accuracy (%)** (primary left Y-axis) and **Validation Macro F1 Score** (secondary right Y-axis) across epochs.

---

## 📊 Comparative Performance Matrix

| Metric / Parameter | Baseline (EX-00 Canonical) | Tuned Variant (EX-04 Initial Step 1200) | Tuned Variant (EX-04 Best Acc Step 2700) | Delta ($\Delta$ vs Baseline) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Test Accuracy** | **91.19%** | 90.34% | **91.12%** | $-0.07\%$ | Target Reached |
| **Test Macro F1** | **0.9118** | 0.9034 | **0.9112** | $-0.0006$ | Target Reached |
| **ROC-AUC Score** | **0.9699** | 0.9677 | **0.9677** | $-0.0022$ | Stable |
| **Selected Checkpoint Step** | Step 2700 (Epoch 1.92) | Step 1200 (Epoch 0.85) | **Step 2700 (Epoch 1.92)** | $0$ steps | Optimal |
| **Learning Rate ($\text{LR}$)** | $2.0 \times 10^{-5}$ (Linear) | $1.5 \times 10^{-5}$ (Cosine) | $1.5 \times 10^{-5}$ (Cosine) | $-0.5 \times 10^{-5}$ | Slower Decay |
| **Weight Decay** | 0.01 | 0.10 | 0.10 | $+0.09$ | Regularized |
| **Classifier Dropout** | 0.20 | 0.30 | 0.30 | $+0.10$ | Regularized |
| **Label Smoothing** | 0.00 | 0.10 | 0.10 | $+0.10$ | Soft Targets |
| **Early Stop Metric** | N/A | `eval_loss` (Premature Halt) | `eval_accuracy` / `eval_f1` | Matched | Fixed |
| **Peak VRAM Allocated** | 264.6 MB | 264.6 MB | 264.6 MB | $0.0\text{ MB}$ | Budget OK ($\le 3.5\text{ GB}$) |

---

## 🔬 Root Cause Diagnostics & Failure Analysis

```
Validation Accuracy (%) & Macro F1
 ^
91% |                            /---\ (Baseline & EX-04 Step 2700 Peak: ~90.5-90.8% -> Test: 91.12%)
    |                           /     \
90% |             /------------/       \
    |            /
89% |    /------/  <--- Old eval_loss Early Stop Premature Halt (Step 1200 -> Test: 90.34%)
    |   /
88% |  /
    +----------------------------------------------------> Epochs / Steps
       0.2    0.5    0.8    1.2    1.5    1.9    2.3
```

### 1. The Loss vs Accuracy Divergence Trap
In transformer fine-tuning on IMDB:
- **Validation Loss** measures cross-entropy probability error. It drops quickly to $\sim 0.27$ at Step 1200 (Epoch 0.85) and then begins a slight upward drift ($\sim 0.30 - 0.38$) as non-zero probabilities on hard ambiguous reviews get penalized.
- **Validation Accuracy & F1** measure decision boundary correctness (`argmax`). Accuracy & F1 continue improving from $88.32\%$ at Step 1200 up to **$90.52\%$ at Step 2700–3000** (Epoch 1.92–2.13).
- Setting `metric_for_best_model: "eval_accuracy"` or `"eval_f1"` ensures HF `Trainer` captures the peak classification performance before ending training.

### 2. Dual-Axis Trajectory Visualization (Notebook 02 Cell 4)
Cell 4 in `notebooks/02_ex2_finetune.ipynb` now plots dual curves on Panel 2:
- Left Y-axis (Green solid line): **Validation Accuracy (%)**
- Right Y-axis (Orange dashed line): **Validation Macro F1 Score**

---

## 💡 Lessons Learned

1. **Never use `eval_loss` as the Early Stopping metric for classification task accuracy.** Best model selection MUST be anchored on `eval_accuracy` or `eval_f1` with `greater_is_better: True`.
2. **Dual-Axis plots reveal metric synchronization.** Tracking both Accuracy and F1 Score on twin Y-axes confirms balanced class precision/recall progression during fine-tuning.
3. **Canonical Baseline (91.19%) is robust.** $2.0 \times 10^{-5}$ learning rate with linear decay and moderate weight decay ($0.01$) remains the benchmark recipe for `distilbert-base-uncased` on IMDB.

---

## 📄 Compliance & Verification
- **Logging & Checkpoint Rules:** Compliant with [`LOGGING_CHECKPOINT_RULES.md`](../rules/LOGGING_CHECKPOINT_RULES.md).
- **Results Index:** Recorded in [`experiments/results/imdb_sentiment_eval.json`](../../experiments/results/imdb_sentiment_eval.json).

