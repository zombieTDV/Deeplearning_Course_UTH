# MLP Changelog

- **Motivation/Background**: This document tracks MLP experiments for error analysis on Fashion-MNIST. The MLP serves as a non-convolutional baseline to understand how fully-connected architectures handle fine-grained class discrimination (especially Shirt vs. T-shirt/top vs. Pullover vs. Coat), and to identify architectural limitations compared to CNNs.
- **Purpose**: Document all MLP-specific experiments, results, and findings following a single-variable principle to systematically evaluate how width and depth affect classification performance.
- **Overview Pipeline**: The document was created through iterative experimentation: Phase 1 established a diagnostic baseline (10-epoch and 30-epoch training with logit bias sweep), followed by Experiment 2 (wider MLP 512→256→10) and Experiment 3 (deeper MLP 512→256→128→10), with each step varying one architectural factor while holding all others constant.
- **Detailed Plan**: Phase 1 — MLP Diagnostic Baseline (baseline 10-epoch training, extended 30-epoch training with CosineAnnealingLR, logit bias sweep); Experiment 2 — Wider MLP (512→256→10); Experiment 3 — Deeper MLP (512→256→128→10); Cross-Experiment Summary.
- **References**: PyTorch, torchvision (Fashion-MNIST), scikit-learn (accuracy_score, classification metrics), matplotlib, seaborn, NumPy, pandas, Adam optimizer, CosineAnnealingLR scheduler.

## Table of Contents

- [Phase 1 — MLP Diagnostic Baseline](#phase-1--mlp-diagnostic-baseline)
  - [Changes](#changes)
  - [Results — Baseline (10 epochs)](#results--baseline-10-epochs)
  - [Results — Extended (30 epochs)](#results--extended-30-epochs)
  - [Logit Bias Sweep](#logit-bias-sweep)
  - [Fixes](#fixes)
- [Experiment 2 — Architecture: Wider MLP](#experiment-2--architecture-wider-mlp-51225610)
  - [Results](#results-1)
  - [Key findings](#key-findings)
- [Experiment 3 — Architecture: Deeper MLP](#experiment-3--architecture-deeper-mlp-51225612810)
  - [Results](#results-2)
  - [Key findings](#key-findings-1)
- [Cross-Experiment Summary](#cross-experiment-summary)
  - [Architectural ceiling](#architectural-ceiling)

---

## Phase 1 — MLP Diagnostic Baseline

- **Date**: 2026-07-26
- **Model**: MLP 784→256→128→10, ReLU, Dropout 0.2 — **235K parameters**
- **Notebook**: `notebooks/error_analysis/MLP/phase1_mlp_diagnostics.ipynb`
- **Outputs**: `outputs/error_analysis/MLP/phase1_diagnostics/`

### Changes
- Initial baseline diagnostic notebook created for MLP error analysis, mirroring the CNN Phase 1 template
- Three sub-phases in one notebook: A1 (baseline 10 epochs), A2 (extended 30 epochs + CosineAnnealingLR), A3 (logit bias sweep)
- All metrics saved as raw `.txt` files with `_10` and `_30` suffixes

### Results — Baseline (10 epochs)

| Class | TPR | Precision |
|-------|:---:|:---------:|
| Shirt | **0.699** | 0.718 |
| Pullover | 0.807 | 0.805 |
| T-shirt/top | 0.809 | 0.853 |
| **Accuracy** | **88.16%** | |

### Results — Extended (30 epochs)

| Metric | 10 epochs | 30 epochs | Δ |
|--------|:--------:|:---------:|:-:|
| Accuracy | 88.16% | **90.08%** | **+1.92pp** |
| Shirt TPR | 0.699 | **0.707** | +0.008 |
| Macro PR-AUC | — | 0.9517 | — |

Extended training helped all classes except Shirt, which improved only +0.008 TPR (vs Sandal +0.056). This indicated Shirt confusion is capacity-bound, not convergence-bound.

### Logit Bias Sweep

| Bias | Acc% | Shirt TPR | Shirt Prec |
|:----:|:---:|:---------:|:---------:|
| 0.0 | **90.08** | 0.707 | 0.751 |
| +0.5 | 89.89 | **0.758** | 0.699 |
| +1.0 | 89.44 | 0.806 | 0.646 |

Best practical trade-off: bias=+0.5 (89.89% acc, 0.758 Shirt TPR). Shirt TPR cannot exceed ~0.76 without dropping accuracy below 89.9%.

### Fixes
- Cell 0 converted from `markdown_cell` to `code_cell` (was dead code — never executed)
- `OUT_DIR` and `DATA_DIR` changed from fragile relative paths to `PROJ_ROOT`-based absolute paths using `_find_root()` CWD detection
- `from sklearn.metrics import accuracy_score` moved out of bias-sweep loop to top-level imports
- `download=False` removed (`download=True` restored) to ensure data availability

---

## Experiment 2 — Architecture: Wider MLP (512→256→10)

> **Variable changed**: Hidden layer widths doubled (256→512, 128→256)
> **Held constant**: depth (2 hidden layers), dropout (0.2), training (30 epochs, Adam lr=0.001, CosineAnnealingLR), loss (CrossEntropy)

- **Date**: 2026-07-26
- **Model**: MLP 784→512→256→10, ReLU, Dropout 0.2 — **540K parameters (2.3× Phase 1)**
- **Notebook**: `notebooks/error_analysis/MLP/phase2_mlp_wider.ipynb`
- **Outputs**: `outputs/error_analysis/MLP/wider/`

### Results

| Class | Phase 1 TPR | Wider TPR | Δ |
|-------|:----------:|:---------:|:-:|
| Shirt | 0.707 | **0.728** | **+0.021** |
| T-shirt/top | 0.855 | 0.861 | +0.006 |
| Trouser | 0.972 | 0.978 | +0.006 |
| Pullover | 0.841 | 0.838 | −0.003 |
| Coat | 0.856 | 0.853 | −0.003 |

| Metric | Phase 1 | Wider | Δ |
|--------|:------:|:-----:|:-:|
| Accuracy | 90.08% | **90.26%** | +0.18pp |
| Macro PR-AUC | 0.9517 | **0.9526** | +0.0008 |
| Shirt PR-AUC | 0.8232 | 0.8307 | +0.0075 |
| Shirt errors | 293 | **272** | −21 |

### Key findings
- Wider capacity helped Shirt most (+0.021 TPR) but at 2.3× the parameters — poor ROI
- Pullover and Coat slightly regressed (−0.003 each)
- Confirms Phase 1 hypothesis: Shirt benefits from capacity, but diminishing returns are steep
- 540K params still underperform a 260K-param CNN (92.50% acc, 0.847 Shirt TPR)

---

## Experiment 3 — Architecture: Deeper MLP (512→256→128→10)

> **Variable changed**: Added third hidden layer (2→3 hidden layers)
> **Held constant**: width (512→256→128), dropout (0.2), training (30 epochs, Adam lr=0.001, CosineAnnealingLR), loss (CrossEntropy)

- **Date**: 2026-07-26
- **Model**: MLP 784→512→256→128→10, ReLU, Dropout 0.2 — **~572K parameters (2.4× Phase 1)**
- **Notebook**: `notebooks/error_analysis/MLP/phase3_mlp_deeper.ipynb`
- **Outputs**: `outputs/error_analysis/MLP/deeper/`

### Results

| Metric | Phase 1 | Deeper | Δ |
|--------|:------:|:-----:|:-:|
| Accuracy | 90.08% | 90.13% | +0.05pp |
| Macro PR-AUC | 0.9517 | 0.9518 | +0.0001 |
| Shirt TPR | 0.707 | 0.714 | +0.007 |
| Shirt Precision | 0.751 | **0.765** | **+0.014** |
| Shirt PR-AUC | 0.8232 | **0.8334** | **+0.0102** |
| Coat TPR | 0.856 | 0.836 | −0.020 |

### Key findings
- Deeper architecture barely beats baseline (+0.05pp acc) despite 2.4× parameters
- Coat substantially regressed (−0.020 TPR) — third layer hurts coat discrimination
- Shirt→Coat confusion decreased (−15 errors) but Shirt→T-shirt/top increased (+5)
- Precision improved more than recall — model makes fewer but higher-confidence Shirt predictions
- Extra layer adds depth without proportional benefit; width (Exp 2) outperforms depth (Exp 3)

---

## Cross-Experiment Summary

| Experiment | Params | Acc% | Shirt TPR | Macro PR-AUC | Best For |
|:----------:|:-----:|:----:|:---------:|:------------:|----------|
| Phase 1 (baseline) | 235K | 90.08 | 0.707 | 0.9517 | Baseline reference |
| Exp 2 (Wider) | 540K | **90.26** | **0.728** | **0.9526** | Highest accuracy & Shirt TPR |
| Exp 3 (Deeper) | 572K | 90.13 | 0.714 | 0.9518 | Shirt precision (0.765) |

### Architectural ceiling
No MLP configuration tested reaches CNN-level Shirt discrimination (0.847 TPR at 92.50% acc). The gap is fundamental: convolution provides translation-invariant spatial feature extraction that fully-connected layers cannot replicate regardless of width or depth.
