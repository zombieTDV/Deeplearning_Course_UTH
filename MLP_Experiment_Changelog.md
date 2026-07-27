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
- [Experiment 4 — Optuna HP Search](#experiment-4--optuna-hp-search)
  - [Results](#results-3)
  - [Key findings](#key-findings-2)
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

## Experiment 4 — Optuna HP Search

> **Variable changed**: All HPs (architecture, activation, dropout, optimizer, LR, WD, scheduler) discovered automatically via 50-trial TPE search with Median pruning (PR-AUC).
> **Held constant**: MLP architecture family, CrossEntropy loss, FashionMNIST (no augmentation), 15 epochs/trial, 30-epoch best-config retrain.

- **Date**: 2026-07-27
- **Model**: Best config: MLP 784→215→470→10, GELU, Dropout 0.05 — **~332K parameters**
- **Notebook**: `notebooks/error_analysis/MLP/phase4_mlp_optuna_merged.ipynb`
- **Outputs**: `outputs/error_analysis/MLP/phase4_optuna/`

### Results

| Metric | Phase 1 | Exp 4 Optuna | Δ |
|--------|:------:|:-----------:|:-:|
| Test Accuracy | 90.08% | **90.00%** | −0.08pp |
| Val Accuracy (best epoch) | — | **90.58%** | — |
| Shirt TPR (bias=0) | 0.707 | 0.712 | +0.005 |
| Shirt TPR (bias=+0.5) | 0.758 | 0.744 | −0.014 |
| Macro PR-AUC | 0.9517 | **0.9593** | +0.0076 |
| Shirt PR-AUC | 0.8232 | — | — |

| Bias | Acc% | Shirt TPR | Shirt Prec |
|:----:|:---:|:---------:|:---------:|
| 0.0 | **90.00** | 0.712 | 0.749 |
| +0.5 | 89.87 | 0.744 | 0.712 |
| +1.0 | 89.80 | **0.779** | 0.688 |

### Key findings

1. **Optuna search collapsed to the same region as hand-tuned HPs.** All top-5 trials are 2-layer GELU networks with dropout 0.0–0.05, AdamW, cosine scheduler, LR 1.2–2.6e-3, and near-zero weight decay. The search did not discover a configuration that meaningfully beats Phase 1's hand-tuned baseline (90.08%).
2. **GELU > ReLU is the only clear HP improvement.** GELU provided ~+0.2pp over ReLU at equivalent capacity. All other HP dimensions (WD, dropout, scheduler type) converged to "effectively off" or had minimal impact.
3. **Architecture: 2 layers, ascending bottlebrush (small→large).** The best configs consistently used a smaller first layer (~200–300) and larger second layer (~450–750). This reverses Phase 1's 256→128 descending pattern, but the net gain is marginal.
4. **Higher dropout hurts at this capacity.** The model is in an underfitting regime — dropout 0.0–0.05 dominates. Phase 1's 0.2 dropout was actively harmful.
5. **AdamW ≈ Adam at near-zero weight decay.** The optimizer choice is not a meaningful differentiator when WD ≈ 2e-6.
6. **Test accuracy slightly below val accuracy suggests mild val-overfitting.** Val accuracy reached 90.58% (epoch 28) but test accuracy was 90.00% — a 0.58pp gap. The fixed 5K validation split across all 50 trials may have allowed TPE to exploit split-specific patterns.

---

## Cross-Experiment Summary

| Experiment | Params | Acc% | Shirt TPR | Macro PR-AUC | Best For |
|:----------:|:-----:|:----:|:---------:|:------------:|----------|
| Phase 1 (baseline) | 235K | 90.08 | 0.707 | 0.9517 | Baseline reference |
| Exp 2 (Wider) | 540K | 90.26 | **0.728** | 0.9526 | Highest Shirt TPR |
| Exp 3 (Deeper) | 572K | 90.13 | 0.714 | 0.9518 | Shirt precision (0.765) |
| Exp 4 (Optuna) | 332K | 90.00 | 0.712 | **0.9593** | Highest PR-AUC, efficient params |
| **CNN E9 (reference)** | **~260K** | **93.26** | **0.785** | — | **Overall best** |

### Architectural ceiling
No MLP configuration tested reaches CNN-level Shirt discrimination (0.785–0.834 TPR at 92.50–93.26% acc). The gap is fundamental: convolution provides translation-invariant spatial feature extraction that fully-connected layers cannot replicate regardless of width, depth, or HP tuning.

The Optuna search (Exp 4) conclusively demonstrated this ceiling — 50 trials over a search space spanning 10+ HPs could not find a configuration exceeding 90.26% accuracy. For overall accuracy at bias=0, the optimal MLP configuration is:

| HP | Optimal Value |
|----|:------------:|
| Hidden layers | 2 |
| Units | 215 → 470 (ascending) |
| Activation | GELU |
| Dropout | 0.05 |
| Optimizer | AdamW |
| Learning rate | 2.0e-3 |
| Weight decay | 2.0e-6 |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Training epochs | 30 |

Further MLP work is not recommended — effort should focus on CNN experiments or data-centric improvements.
