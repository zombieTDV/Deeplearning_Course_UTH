# PLAN: Advanced Anti-Overfitting & Accuracy Enhancements for Exercise 2

**Document ID:** `PLAN-EX2-ADVANCED-ANTI-OVERFITTING`  
**Date:** 2026-08-12  
**Author:** bush-le + Antigravity AI Agent  
**Status:** Ready for Implementation & Notebook Integration  

---

## 📌 Executive Summary

To further suppress validation loss rebound and push test set accuracy beyond **91.5%**, this document outlines 5 advanced anti-overfitting techniques that are not yet integrated into [`notebooks/02_ex2_finetune.ipynb`](../../notebooks/02_ex2_finetune.ipynb) or [`src/training/imdb_sentiment_train.py`](../../src/training/imdb_sentiment_train.py).

---

## 🔬 5 Advanced Anti-Overfitting Techniques to Integrate

### 1. Label Smoothing Regularization (`label_smoothing_factor: 0.10`)
- **Mechanism:** Replaces hard $0/1$ binary targets ($[1.0, 0.0]$) with soft target distributions ($[0.95, 0.05]$).
- **Hypothesis:** Prevents cross-entropy loss from penalizing non-zero probabilities heavily, suppressing logit overconfidence and preventing post-epoch validation loss spikes.
- **Implementation:** Add `--label-smoothing-factor` CLI flag to `imdb_sentiment_train.py` and pass `label_smoothing_factor=0.10` to HF `TrainingArguments`.

### 2. Layer-wise Learning Rate Decay (LLRD)
- **Mechanism:** Assigns exponentially decaying learning rates from upper classifier layers down to bottom embedding layers:
  - Classifier Head: $\text{LR} = 2.0 \times 10^{-5}$
  - Upper Transformer Layers (4–6): $\text{LR} = 1.2 \times 10^{-5}$
  - Lower Transformer Layers (1–3): $\text{LR} = 6.0 \times 10^{-6}$
  - Embeddings: $\text{LR} = 3.0 \times 10^{-6}$
- **Hypothesis:** Preserves general syntactic & semantic representations in low-level layers while adapting high-level sentiment features.
- **Implementation:** Custom parameter group creation in `_train()` optimizer setup.

### 3. On-The-Fly Text Data Augmentation (Synonym Replacement / EDA)
- **Mechanism:** Randomly replaces $10\%$ of non-stopwords with WordNet synonyms during tokenization for the training split.
- **Hypothesis:** Forces the model to learn invariant semantic sentiment concepts rather than memorizing exact word sequences.
- **Implementation:** Integrated data transform function in `src/data/prepare_imdb.py`.

### 4. Gradient Clipping (`max_grad_norm: 1.0`)
- **Mechanism:** Clips gradient norms to a maximum threshold of $1.0$.
- **Hypothesis:** Stabilizes weight updates during early warmup steps when high dropout ($0.30$) and weight decay ($0.10$) are active.
- **Implementation:** Pass `max_grad_norm=1.0` to HF `TrainingArguments`.

### 5. Stochastic Weight Averaging (SWA / Checkpoint Averaging)
- **Mechanism:** Averages the model weights of the top 3 best evaluation checkpoints (e.g. Step 900, 1200, 1500).
- **Hypothesis:** Smoothes out the loss landscape surface near local minima, improving test set generalization without incurring extra inference latency.
- **Implementation:** Helper function in `src/utils/checkpoint_utils.py` to average PyTorch `state_dict`s.

---

## 🗺️ Integration Roadmap for Notebook 02

| Step | Notebook Section | New Feature / Code Block |
|:---:|:---|:---|
| **Step 2.1** | Training Launcher | Add Label Smoothing (`--label-smoothing-factor 0.10`) & LLRD parameter switches. |
| **Step 2.2** | Hyperparameter Search | Add interactive grid toggles for Label Smoothing vs Baseline. |
| **Step 3.1** | Curve Comparison | Plot Loss curves comparing Standard Finetune vs Label Smoothed Finetune. |
| **Step 5.1** | SWA Ensemble Predictor | Load SWA averaged checkpoint for interactive text predictions. |

---

## 🛠️ Verification & Test Plan

1. **CLI Flag Testing:** Run `python -m src.training.imdb_sentiment_train --label-smoothing-factor 0.10 --smoke` to verify flag parsing.
2. **Notebook Regeneration:** Run `python scratch/build_notebook_ex2.py` to embed new interactive cells into `notebooks/02_ex2_finetune.ipynb`.
3. **Repository Verification:** Run `pytest` (3/3 pass) and `ruff check` (0 linter errors).
