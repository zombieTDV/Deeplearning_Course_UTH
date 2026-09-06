# EXPERIMENT SPECIFICATION: Exercise 2 Advanced Anti-Overfitting Suite

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---


**Document ID:** `EXP-EX2-ADVANCED-SPEC`  
**Date:** 2026-08-12  
**Author:** bush-le + Antigravity AI Agent  
**Status:** Planned  

---

## 📌 Overview

This document specifies 4 advanced experimental trials designed to enhance generalization and prevent validation loss rebound in Exercise 2 (`distilbert-base-uncased` fine-tuning on IMDB).

---

## 🧪 Experimental Variants (`ADV-01` to `ADV-04`)

| Trial ID | Technique | Hyperparameters | Target Benefit |
|:---:|:---|:---|:---|
| **`ADV-01`** | **Label Smoothing** | `label_smoothing_factor: 0.10` | Suppresses logit overconfidence & lowers val loss rebound |
| **`ADV-02`** | **Layer-wise LR Decay (LLRD)** | `lr_top: 2e-5, lr_bottom: 5e-6` | Protects low-level pre-trained representations |
| **`ADV-03`** | **Gradient Clipping + SWA** | `max_grad_norm: 1.0, swa_top_k: 3` | Stabilizes training dynamics & smoothes loss landscape |
| **`ADV-04`** | **Full Advanced Combo** | Label Smoothing + LLRD + SWA + Dropout 0.30 | Maximum accuracy target ($> 91.5\%$) |

---

## 📊 Target Metrics & Verification Ceiling

- **Target Test Accuracy:** $\ge 91.50\%$
- **Target Val Loss Floor:** $\le 0.2550$
- **VRAM Constraint:** $\le 3.5\text{ GB}$ (Ceiling $4.0\text{ GB}$)
