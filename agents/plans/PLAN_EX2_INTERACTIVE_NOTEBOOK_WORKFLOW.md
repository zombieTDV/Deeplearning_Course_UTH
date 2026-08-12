# PLAN: Practice 3 Exercise 2 Interactive Notebook & End-to-End Workflow

**Document ID:** `PLAN-EX2-INTERACTIVE-NOTEBOOK`  
**Date:** 2026-08-12  
**Author:** bush-le + Antigravity AI Agent  
**Status:** Approved for Implementation  

---

## 📌 Executive Summary & Intent

The user requested a 100% self-contained, interactive, end-to-end Jupyter Notebook experience in [`notebooks/02_ex2_finetune.ipynb`](../../notebooks/02_ex2_finetune.ipynb). Users must be able to open this single notebook and execute every phase of the machine learning lifecycle:
1. **Data Inspection & Visualization:** Inspect train/val/test splits, token length distributions, and label balance.
2. **Direct In-Notebook Training:** Execute model training directly from a notebook cell using the optimal anti-overfitting configuration (`label_smoothing_factor=0.10`, `classifier_dropout=0.30`, `weight_decay=0.10`, `early_stopping_patience=3`, `lr_scheduler_type="cosine"`).
3. **Live & Interactive Metric Plots:** Render real-time TensorBoard logs and dual-panel Loss/Accuracy history charts.
4. **Direct Test Set Evaluation:** Evaluate the trained model on 25,000 sealed test reviews, displaying 5W1H metrics, Confusion Matrix, and ROC-AUC curves.
5. **Interactive Custom Inference:** Test custom user text inputs in real-time.
6. **Root Cause Error Analysis:** Extract and inspect false positive & false negative misclassifications.

---

## 🎯 Detailed Step-by-Step Notebook Architecture

| Step # | Cell Type | Title / Focus | Implementation Details |
|:---:|:---:|:---|:---|
| **0** | Markdown + Code | **Environment Setup & Imports** | Import PyTorch, Hugging Face `transformers`, `src.training.imdb_sentiment_train`, set up seed & CUDA device context. |
| **1** | Markdown + Code | **Dataset Exploration & Visualization** | Load raw IMDB dataset; plot label distribution bar charts and review word/token length histograms; show raw text samples. |
| **2** | Markdown + Code | **Interactive In-Notebook Training Launcher** | Direct Python function call to `_train()` with configurable hyperparameters (`label_smoothing_factor=0.10`, `dropout=0.30`, `wd=0.10`, `patience=3`). |
| **3** | Markdown + Code | **TensorBoard & Training Curves** | Embed live `%tensorboard` widget + matplotlib dual-panel Training vs Validation Loss and Accuracy/F1 curves. |
| **4** | Markdown + Code | **Direct Test Set Evaluation** | Invoke `evaluate_model` directly from code; plot 5W1H metrics, Confusion Matrix heatmap, and ROC-AUC curve. |
| **5** | Markdown + Code | **Interactive Inference Predictor** | Interactive Python function allowing users to pass custom review text and receive positive/negative probabilities. |
| **6** | Markdown + Code | **Baseline Head-to-Head & Error Analysis** | Compare Ex 1 Zero-Shot vs Ex 2 Finetuned DistilBERT; extract top misclassified examples for qualitative error audit. |

---

## 🛠️ Verification & Acceptance Criteria

1. **Notebook Execution:** Executing `python scratch/build_notebook_ex2.py` must cleanly regenerate `notebooks/02_ex2_finetune.ipynb`.
2. **Code Health:** `pytest` (3/3 tests pass) and `ruff check` (0 errors).
3. **Interactive Functionality:** All cells in `02_ex2_finetune.ipynb` can run sequentially from top to bottom without relying on external terminal commands.
