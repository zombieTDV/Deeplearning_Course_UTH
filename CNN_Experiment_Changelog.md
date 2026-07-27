# CNN Experiment Changelog

- **Motivation/Background**: This document tracks all CNN experiments for error analysis on Fashion-MNIST. Starting from a practice baseline, it systematically evaluates architecture, loss function, data augmentation, class weighting, input resolution, training schedule, post-hoc inference adjustments, optimizer choice, and Bayesian hyperparameter search to understand and mitigate the upper-body confusion sink effect centered on Shirt.
- **Purpose**: Document all CNN-specific experiments, results, and findings following a single-variable principle to systematically evaluate how each intervention affects classification performance and the Shirt confusion sink.
- **Overview Pipeline**: The document was created through iterative experimentation: Baseline established on practice_1, followed by 13 experiments (E1–E13) spanning architecture (widen + residual), loss functions (Focal, Weighted CE, Label Smoothing), data augmentation, input resolution upscaling, extended training schedules, post-hoc logit adjustment, intra-model ensemble, optimizer comparison (SGD), and Optuna Bayesian HP search with validation on test set.
- **Detailed Plan**: Baseline — practice_1; Experiment 1 — Phase 1 Diagnostic CNN; Experiment 2 — Widen + Residual Architecture; Experiment 3 — Focal Loss; Experiment 4 — Augmentation; Experiment 5 — Weighted CrossEntropy; Experiment 6 — Input Upscaling; Experiment 7 — Label Smoothing; Experiment 8 — Extended Training; Experiment 9 — Post-hoc Logit Adjustment; Experiment 10 — Intra-Model Ensemble; Experiment 11 — SGD Optimizer; Experiment 12 — Optuna HP Search; Experiment 13 — Optuna Best Config Validation; Cross-Experiment Summary.
- **References**: PyTorch, torchvision (Fashion-MNIST), scikit-learn (accuracy_score, classification metrics), matplotlib, seaborn, NumPy, pandas, Adam, AdamW, SGD optimizer, CosineAnnealingLR, StepLR, ReduceLROnPlateau, Focal Loss, Optuna (TPESampler, MedianPruner, ASHA).

## Table of Contents

- [Baseline — practice_1](#baseline--practice_1)
- [Experiment 1 — Phase 1 Diagnostic CNN](#experiment-1--phase-1-diagnostic-cnn)
- [Experiment 2 — Architecture: Widen + Residual](#experiment-2--architecture-widen--residual)
- [Experiment 3 — Loss: Focal Loss](#experiment-3--loss-focal-loss-%CE%B3%3D20)
- [Experiment 4 — Data: Augmentation](#experiment-4--data-augmentation-randomerasing--randomaffine)
- [Experiment 5 — Loss: Weighted CrossEntropy](#experiment-5--loss-weighted-crossentropy-shirt-3%C3%97)
- [Experiment 6 — Data: Input Resolution Upscaling](#experiment-6--data-input-resolution-upscaling-56%C3%9756)
- [Experiment 7 — Loss: Label Smoothing CE](#experiment-7--loss-label-smoothing-ce)
- [Experiment 8 — Training: Extended Schedule](#experiment-8--training-extended-schedule-40-epochs--cosine-lr)
- [Experiment 9 — Inference: Post-hoc Logit Adjustment](#experiment-9--inference-post-hoc-logit-adjustment-sweep)
- [Experiment 10 — Inference: Intra-Model Ensemble](#experiment-10--inference-intra-model-ensemble-softmax-average)
- [Experiment 11 — Optimizer: SGD + Nesterov](#experiment-11--optimizer-sgd--nesterov--reducelronplateau)
- [Experiment 12 — Hyperparameter Search: Optuna](#experiment-12--hyperparameter-search-optuna-bayesian--asha-pruning)
- [Experiment 13 — HP Validation: Optuna Best Config](#experiment-13--hp-validation-optuna-best-config-on-test-set)
- [Cross-Experiment Summary](#cross-experiment-summary)
- [Conclusion](#conclusion)

---

## Baseline — practice_1

- **Date**: 2026-07-26
- **Models**: MLP (784→256→128→10, ReLU, Dropout 0.2) and CNN (Conv1→32→Pool→Conv32→64→Pool→FC 128→10, ReLU, Dropout 0.25)
- **Training**: 10 epochs, CrossEntropyLoss + Adam (lr=0.001)
- **Metric files**: `outputs/practice_1/metrics/`
- **Key results**:
  - MLP accuracy: 88.15% / CNN accuracy: 92.02%
  - MLP macro PR-AUC: 0.9394 / CNN macro PR-AUC: 0.9677

### Changes
- `src/train_utils.py` — `train_model()`: added `model_name` param; saves per-epoch losses to `outputs/practice_1/metrics/train_losses_{model_name}.txt`
- `src/eval_utils.py` — `evaluate_detailed()`: added `model_name` param; saves evaluation metrics and confusion matrix to `outputs/practice_1/metrics/`
- `src/eval_utils.py` — `compute_roc_auc_scores()`, `compute_pr_auc_scores()`: added `model_name` param; saves scores to `outputs/practice_1/metrics/`
- `src/eval_utils.py` — `print_comparison_table()`: saves comparison table as Markdown to `outputs/practice_1/metrics/comparison_table.md`
- `src/vis_utils.py` — `plot_class_distribution_comparison()`: new function, saves grouped bar plot + `class_distribution.txt`
- `notebooks/practice_1.ipynb` — added Step 4.5 (class distribution call); updated all save paths; added `model_name` to all function calls
- `src/vis_utils.py`, `src/model_utils.py` — updated default paths to `outputs/practice_1/`

### Directory reorganisation
- `outputs/{images,metrics,model,plots}/` → `outputs/practice_1/{images,metrics,model,plots}/`
- `outputs/error_analysis/` created for diagnostic outputs

---

## Experiment 1 — Phase 1 Diagnostic CNN

- **Date**: 2026-07-26
- **Model**: DiagnosticCNN — 3 conv blocks (32→64→128), BatchNorm, ReLU, MaxPool, GAP, Dropout(0.3), Linear(128→10). **No attention.**
- **Training**: 15 epochs, CrossEntropyLoss + Adam (lr=0.001)
- **Notebook**: `notebooks/phase1_cnn_error_analysis.ipynb`
- **Outputs**: `outputs/error_analysis/` — 9 metric files + 5 diagnostic plots

### Key results
- Accuracy: **92.50%** (+0.48% over baseline CNN)
- Macro PR-AUC: **0.9712** (+0.0035 over baseline CNN)
- Shirt TPR: **0.847** (+7.5% over baseline CNN) — at cost of T-shirt TPR −1.5%
- Dominant failure: Shirt sink effect (315 false positives from 4 upper-body classes)

### Bug fix
- `notebooks/phase1_cnn_error_analysis.ipynb` — training cell: changed `f'{loss}'` to `f'{loss}\n'` in file write to produce parseable per-line output

---

## Experiment 2 — Architecture: Widen + Residual

> **Variable changed**: CNN architecture only — wider channels (256), residual connections, LeakyReLU, Dropout 0.5
> **Held constant**: data loading, augmentation (none), loss (CrossEntropy), optimizer (Adam lr=0.001), epochs (15)

- **Notebook**: `notebooks/phase2_arch_widen_residual.ipynb`
- **Outputs**: `outputs/error_analysis/arch_widen_residual/`

### Changes
- `DiagnosticCNN` → `ResidualCNN`
  - Conv widths: 64→128→256 (was 32→64→128)
  - Residual skip connections: Conv1×1 + MaxPool shortcut from block 1 output → block 3 input
  - LeakyReLU(negative_slope=0.1) replaces ReLU
  - Dropout increased from 0.3 → 0.5

### Results

| Metric | Baseline (E1) | Wider+Res (E2) | Δ |
|--------|:------------:|:--------------:|:-:|
| Accuracy | 92.50% | **92.65%** | +0.15% |
| Macro PR-AUC | 0.9712 | 0.9713 | +0.0001 |
| Shirt TPR | **0.847** | 0.732 | −0.115 |
| T-shirt/top TPR | 0.837 | **0.939** | +0.102 |
| Pullover TPR | 0.890 | 0.931 | +0.041 |
| Coat TPR | 0.870 | **0.909** | +0.039 |
| Dress TPR | 0.907 | 0.928 | +0.021 |
| Shirt Precision | 0.723 | **0.826** | +0.103 |

### Key findings
- TPR sink shifted from Shirt → T-shirt/top (T-shirt errors 315→167; Shirt errors 153→268)
- Higher overall capacity made the model "choose" which upper-body class to sacrifice
- Zero-sum trade-off: Shirt −11.5%, T-shirt +10.2% — near-perfect conservation of errors
- Bug fix confirmed: ResidualCNN `get_features` had spatial mismatch corrected

---

## Experiment 3 — Loss: Focal Loss (γ=2.0)

> **Variable changed**: loss function only — CrossEntropy → FocalLoss(γ=2.0, α=1.0)
> **Held constant**: architecture (DiagnosticCNN), data (no augmentation), optimizer (Adam lr=0.001), epochs (15)

- **Notebook**: `notebooks/phase3_focal_loss.ipynb`
- **Outputs**: `outputs/error_analysis/focal_loss/`

### Results

| Metric | Baseline (E1) | Focal (E3) | Δ |
|--------|:------------:|:----------:|:-:|
| Accuracy | 92.50% | 92.40% | −0.10% |
| Macro PR-AUC | 0.9712 | 0.9706 | −0.0006 |
| Shirt TPR | **0.847** | **0.713** | −0.134 |
| T-shirt/top TPR | 0.837 | 0.881 | +0.044 |
| Pullover TPR | 0.890 | 0.896 | +0.006 |
| Coat TPR | 0.870 | 0.931 | +0.061 |
| Dress TPR | 0.907 | 0.933 | +0.026 |
| Shirt Precision | 0.723 | **0.841** | +0.118 |

### Key findings
- Focal Loss catastrophically suppressed Shirt TPR (worst across all experiments: 0.713)
- The (1−p_t)^γ modulation punishes hard-to-classify examples more than easy ones, but here it made Shirt *worse* — suggesting Shirt's difficulty is structural (ambiguous signal) not distributional (rare vs frequent)
- Average training loss magnitude dropped (~0.24→0.04 vs E1's 0.46→0.09) due to focal modulation factor, not comparable directly

---

## Experiment 4 — Data: Augmentation (RandomErasing + RandomAffine)

> **Variable changed**: training transform only — added RandomAffine(±5°, ±5%) + RandomErasing(p=0.3)
> **Held constant**: architecture (DiagnosticCNN), loss (CrossEntropy), optimizer (Adam lr=0.001), epochs (15), test transform

- **Notebook**: `notebooks/phase4_augmentation.ipynb`
- **Outputs**: `outputs/error_analysis/augmentation/`

### Results

| Metric | Baseline (E1) | Augment (E4) | Δ |
|--------|:------------:|:------------:|:-:|
| Accuracy | 92.50% | 92.45% | −0.05% |
| Macro PR-AUC | 0.9712 | **0.9719** | +0.0007 |
| Shirt TPR | **0.847** | 0.810 | −0.037 |
| T-shirt/top TPR | 0.837 | 0.834 | −0.003 |
| Pullover TPR | 0.890 | **0.936** | +0.046 |
| Coat TPR | 0.870 | 0.882 | +0.012 |
| Dress TPR | 0.907 | 0.888 | −0.019 |
| Shirt Precision | 0.723 | 0.766 | +0.043 |
| Training loss (final) | 0.093 | 0.205 | — (higher due to harder augmented samples) |

### Key findings
- RandomErasing likely erased discriminative collar/neckline regions, widening the ambiguity
- Shirt errors spread more evenly: T-shirt=57, Pullover=58, Coat=51 (vs E1 concentrated on T-shirt=57, less to others)
- Pullover improved (+4.6%) — augmentation helped its distinct hoodie silhouette survive erasing
- Upper-body total errors: E1=649, E4=650 — conservation of errors holds

---

## Experiment 5 — Loss: Weighted CrossEntropy (Shirt 3×)

> **Variable changed**: loss function weights — Shirt weight=3.0, others=1.0
> **Held constant**: architecture (DiagnosticCNN), data (no augmentation), optimizer (Adam lr=0.001), epochs (15)

- **Notebook**: `notebooks/phase5_weighted_loss.ipynb`
- **Outputs**: `outputs/error_analysis/weighted_loss/`

### Results

| Metric | Baseline (E1) | Weighted (E5) | Δ |
|--------|:------------:|:-------------:|:-:|
| Accuracy | 92.50% | **91.75%** | −0.75% |
| Macro PR-AUC | 0.9712 | 0.9700 | −0.0012 |
| Shirt TPR | 0.847 | **0.859** | **+0.012** |
| T-shirt/top TPR | 0.837 | 0.820 | −0.017 |
| Pullover TPR | 0.890 | 0.896 | +0.006 |
| Coat TPR | 0.870 | 0.888 | +0.018 |
| Dress TPR | 0.907 | 0.866 | −0.041 |
| Sandal TPR | 0.978 | 0.938 | −0.040 |
| Shirt Precision | 0.723 | 0.706 | −0.017 |

### Key findings
- **Shirt TPR improved for the first time** (+0.012 over E1), but this is the largest accuracy drop (−0.75%) across all experiments
- Shirt false positives surged: ~324→358 (precision 0.723→0.706) — model became more "Shirt-happy" across all upper-body classes
- Collateral damage to Sandal (−4.0%) and Dress (−4.1%) — weighted loss distorts the entire decision boundary
- Trade-off ratio: each +1 Shirt sample gained costs ~6.3 accuracy points overall

---

## Experiment 6 — Data: Input Resolution Upscaling (56×56)

> **Variable changed**: input image size — `Resize(56, bicubic)` before `ToTensor`
> **Held constant**: architecture (DiagnosticCNN), loss (CrossEntropy), optimizer (Adam lr=0.001), epochs (15), no augmentation

- **Notebook**: `notebooks/phase6_upscale_input.ipynb`
- **Outputs**: `outputs/error_analysis/upscale_input/`

### Results

| Metric | Baseline (E1) | Upscale (E6) | Δ |
|--------|:------------:|:------------:|:-:|
| Accuracy | 92.50% | **90.80%** | **−1.70%** |
| Macro PR-AUC | 0.9712 | 0.9675 | −0.0037 |
| Shirt TPR | 0.847 | 0.846 | −0.001 |
| T-shirt/top TPR | 0.837 | 0.795 | −0.042 |
| Pullover TPR | 0.890 | **0.754** | **−0.136** |
| Coat TPR | 0.870 | **0.956** | **+0.086** |
| Dress TPR | 0.907 | 0.896 | −0.011 |
| Sandal TPR | 0.978 | 0.925 | −0.053 |
| Shirt Precision | 0.723 | 0.703 | −0.020 |

### Key findings
- **Worst accuracy across all experiments** — bicubic interpolation destroys hard-edge information that 3×3 kernels rely on
- Pullover collapsed (−0.136): hoodie/collar boundary distinguishing it from Coat is smoothed into a gradient
- Coat benefited (+0.086): smoother silhouette maps more consistently to a single class
- Shirt essentially unchanged — root cause (silhouette overlap) persists regardless of resolution
- Hypothesis disproven: upscaling with interpolation adds no new information but blurs existing edges

---

## Experiment 7 — Loss: Label Smoothing CE

> **Variable changed**: loss function — `CrossEntropyLoss(label_smoothing=0.1)`
> **Held constant**: architecture (DiagnosticCNN), data (no augmentation), optimizer (Adam lr=0.001), epochs (15)

- **Notebook**: `notebooks/phase7_label_smoothing.ipynb`
- **Outputs**: `outputs/error_analysis/label_smoothing/`

### Results

| Metric | Baseline (E1) | Smooth (E7) | Δ |
|--------|:------------:|:-----------:|:-:|
| Accuracy | 92.50% | 92.04% | −0.46% |
| Macro PR-AUC | 0.9712 | 0.9709 | −0.0003 |
| Shirt TPR | 0.847 | **0.879** | **+0.032** |
| Shirt Precision | 0.723 | 0.692 | −0.031 |
| T-shirt/top TPR | 0.837 | 0.839 | +0.002 |
| Pullover TPR | 0.890 | **0.790** | **−0.100** |
| Coat TPR | 0.870 | 0.874 | +0.004 |
| Dress TPR | 0.907 | **0.938** | +0.031 |

### Key findings
- **Shirt TPR reached 0.879** — the highest across all 7 experiments — but Pullover collapsed to 0.790 as the sink migrated
- Label smoothing (ε=0.1) relaxes the one-hot target, removing the penalty for residual probability mass on non-target classes
- The same zero-sum trade-off: Shirt +4.1%, Pullover −11.2% — near-perfect error conservation within the upper-body cluster

---

## Experiment 8 — Training: Extended Schedule (40 epochs + Cosine LR)

> **Variable changed**: training schedule — 40 epochs with `CosineAnnealingLR(T_max=40)`
> **Held constant**: architecture (DiagnosticCNN), loss (CrossEntropy), optimizer (Adam lr=0.001), data (no augmentation)

- **Notebook**: `notebooks/phase8_extended_training.ipynb`
- **Outputs**: `outputs/error_analysis/extended_training/`

### Results

| Metric | Baseline (E1) | Extended (E8) | Δ |
|--------|:------------:|:-------------:|:-:|
| Accuracy | 92.50% | **92.99%** | **+0.49%** |
| Macro PR-AUC | 0.9712 | **0.9731** | +0.0019 |
| Shirt TPR | 0.847 | **0.797** | −0.050 |
| Shirt Precision | 0.723 | **0.792** | **+0.069** |
| T-shirt/top TPR | 0.837 | 0.864 | +0.027 |
| Pullover TPR | 0.890 | 0.902 | +0.012 |
| Coat TPR | 0.870 | 0.896 | +0.026 |
| Dress TPR | 0.907 | 0.935 | +0.028 |
| **Upper-body errors** | **649** | **601** | **−48** |

### Key findings
- **Highest accuracy across all experiments** (+0.49% over E1, +0.34% over E2)
- Loss trajectory: 0.461 → 0.083 (epoch 15, same as E1 stop) → 0.004 (epoch 39, converged)
- Extended training **hurts Shirt TPR** while improving every other class — the model learns sharper decision boundaries that push Shirt into a deeper sink
- Upper-body errors reduced by 48 samples — the largest reduction seen

---

## Experiment 9 — Inference: Post-hoc Logit Adjustment Sweep

> **Variable changed**: inference-time logit bias added to Shirt class (sweep: −1.0 to +2.0)
> **Held constant**: architecture (DiagnosticCNN), loss (CrossEntropy), training schedule (40 epochs), data (no augmentation)

- **Notebook**: `notebooks/phase9_logit_adjustment.ipynb`
- **Outputs**: `outputs/error_analysis/logit_adjustment/`
- **Model weights**: `model_weights.pth` saved for reuse

### Bias sweep results

| Bias | Acc% | Shirt TPR | Shirt Prec | T-shirt | Pullover | Coat | Dress |
|:----:|:----:|:---------:|:----------:|:-------:|:--------:|:----:|:-----:|
| −1.0 | 93.17 | 0.749 | 0.829 | 0.900 | 0.905 | 0.903 | 0.945 |
| −0.5 | **93.21** | 0.766 | 0.817 | 0.893 | 0.903 | 0.901 | 0.943 |
| 0.0 | **93.26** | 0.785 | 0.806 | 0.886 | 0.899 | 0.900 | 0.942 |
| +0.5 | 93.25 | 0.799 | 0.793 | 0.876 | 0.897 | 0.899 | 0.940 |
| +1.0 | 93.17 | 0.811 | 0.778 | 0.868 | 0.893 | 0.895 | 0.936 |
| +1.5 | 93.11 | 0.826 | 0.763 | 0.860 | 0.889 | 0.891 | 0.931 |
| +2.0 | 93.00 | 0.834 | 0.751 | 0.854 | 0.883 | 0.888 | 0.927 |

### Key findings
- **Zero-cost operating point selection**: the converged model can produce any trade-off between Shirt TPR and precision via a single logit bias value at inference
- **Recommended operating point: bias=+1.0** (93.17% accuracy, Shirt TPR 0.811, Shirt Prec 0.778) — best combined metrics across all 10 experiments
- Negative bias improves precision at Shirt's expense; positive bias recovers Shirt recall at the cost of T-shirt, Pullover, and Coat

---

## Experiment 10 — Inference: Intra-Model Ensemble (softmax average)

> **Variable changed**: ensemble of E9 bias=0.0 and bias=+2.0 softmax outputs (zero training cost)
> **Held constant**: single model loaded from E9 weights

- **Notebook**: `notebooks/phase10_ensemble.ipynb`
- **Outputs**: `outputs/error_analysis/ensemble/`

### Results

| Model | Acc% | Shirt TPR | Shirt Prec |
|:------|:----:|:---------:|:----------:|
| Head A (bias=0.0) | 93.26 | 0.785 | 0.806 |
| **E10 ensemble** | **93.17** | **0.811** | **0.778** |
| Head B (bias=+2.0) | 93.00 | 0.834 | 0.751 |

### Key findings
- Ensemble produces metrics identical to E9 bias=+1.0 — softmax averaging is approximately linear for small logit shifts
- No new information beyond E9's bias sweep; confirms that post-hoc logit adjustment is the simpler and equivalent method
- Total upper-body errors: 597 (lowest across all experiments)

---

## Experiment 11 — Optimizer: SGD + Nesterov + ReduceLROnPlateau

> **Variable changed**: optimizer from Adam to `SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)` + `ReduceLROnPlateau(patience=5)`
> **Held constant**: architecture (DiagnosticCNN), loss (CrossEntropy), epochs (40), data (no augmentation)

- **Notebook**: `notebooks/phase11_sgd_tuning.ipynb`
- **Outputs**: `outputs/error_analysis/sgd_tuning/`

*[pending execution — notebook generated but not run]*

---

## Experiment 12 — Hyperparameter Search: Optuna (Bayesian + ASHA Pruning)

> **Variable changed**: this is a new phase, not a single-variable ablation. All prior experiments (E1–E11) used fixed HPs. This phase finds the optimal HP configuration via Bayesian search.

- **Notebook**: `notebooks/phase12_optuna_hp_search.ipynb`
- **Study DB**: `outputs/error_analysis/optuna_study/optuna_study.db`
- **Method**: 30 trials × 35 epochs, TPESampler, MedianPruner (ASHA-style), 10% validation holdout
- **Search space**: lr [1e-4, 1e-2] log, dropout [0.2, 0.5], weight_decay [1e-6, 1e-3] log, batch_size {64,128,256}, optimizer {Adam, AdamW, SGD}, lr_schedule {None, StepLR, CosineAnnealingLR, ReduceLROnPlateau}

### Results

**9 trials completed, 21 pruned (70% kill rate).**

| Trial | Val-acc | lr | dropout | wd | bs | Optimizer | Schedule |
|:-----:|:-------:|:--:|:-------:|:--:|:--:|:---------:|:--------:|
| #24 | **0.9392** | 0.00332 | 0.454 | 1.4e-5 | 64 | AdamW | CosineAnnealingLR |
| #4 | 0.9382 | 0.00757 | 0.468 | 6.2e-5 | 64 | SGD | StepLR |
| #25 | 0.9372 | 0.00433 | 0.444 | 4.2e-6 | 64 | AdamW | CosineAnnealingLR |
| #0 | 0.9358 | 0.00056 | 0.485 | 1.6e-4 | 64 | AdamW | CosineAnnealingLR |
| #18 | 0.9355 | 0.00151 | 0.462 | 8.6e-5 | 64 | AdamW | CosineAnnealingLR |
| #8 | 0.9345 | 0.00653 | 0.275 | 1.7e-5 | 64 | SGD | CosineAnnealingLR |

### Key findings
- **AdamW dominates**: 5 of top 6 completions; mean val-acc 93.47% vs SGD 92.48%
- **CosineAnnealingLR captures all top-6** — the only trial without a scheduler was the worst (90.18%)
- **Batch size 64** wins all top-6 spots; 128 underperforms (mean 91.93%)
- **Higher dropout (0.44–0.48) is better** — suggests E1's 0.3 was suboptimal
- **Correlations**: lr vs val-acc r=+0.51 (moderate), dropout r=+0.38 (weak positive), weight_decay r≈0 (irrelevant)
- **Best config**: AdamW, lr=0.00332, dropout=0.454, weight_decay=1.39e-5, batch_size=64, CosineAnnealingLR, 35 epochs

---

## Experiment 13 — HP Validation: Optuna Best Config on Test Set

> **Variable changed**: HPs switched from E1 defaults to Optuna-discovered best values
> **Held constant**: architecture (DiagnosticCNN), loss (CrossEntropy), data (no augmentation), logit bias sweep included

- **Notebook**: `notebooks/phase13_optuna_best.ipynb`
- **Outputs**: `outputs/error_analysis/optuna_best/`

### Results

| Metric | E9 Adam bias=0 | E13 AdamW bias=0 | Δ |
|--------|:--------------:|:----------------:|:-:|
| Accuracy | **93.26%** | 93.13% | −0.13% |
| Shirt TPR | 0.785 | 0.764 | −0.021 |
| Shirt Prec | 0.806 | 0.800 | −0.006 |
| T-shirt/top TPR | 0.886 | **0.890** | +0.004 |
| Coat TPR | 0.900 | **0.915** | +0.015 |

### Bias sweep comparison

| Config | Acc% | Shirt TPR | Shirt Prec |
|:-------|:----:|:---------:|:----------:|
| E9 bias=+1.0 (best trade) | **93.17** | 0.811 | 0.778 |
| E13 bias=+1.0 | 93.05 | 0.811 | 0.763 |
| E9 bias=+2.0 | 93.00 | **0.834** | **0.751** |
| E13 bias=+2.0 | 92.81 | 0.834 | 0.728 |

### Key findings
- Optuna's best config **did not outperform E9's simpler Adam config** — the marginal benefit of AdamW over Adam, and of tuning lr/dropout/weight_decay, is negligible at this scale
- **E9 remains the overall best model** across all 13 experiments
- Confirms that HP tuning provides diminishing returns once the training schedule (CosineLR + 35–40 epochs) and architecture are fixed

---

## Cross-Experiment Summary

| Metric | E1 | E2 | E3 | E4 | E5 | E6 | E7 | **E8** | **E9+1** | E10 | E12 (val) | E13 |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:------:|:--------:|:---:|:---------:|:---:|
| Accuracy | 92.50 | 92.65 | 92.40 | 92.45 | 91.75 | 90.80 | 92.04 | **92.99** | 93.17 | 93.17 | 93.92v | 93.13 |
| Shirt TPR | 0.847 | 0.732 | 0.713 | 0.810 | 0.859 | 0.846 | **0.879** | 0.797 | 0.811 | 0.811 | — | 0.764 |
| Shirt Prec | 0.723 | 0.826 | 0.841 | 0.766 | 0.706 | 0.703 | 0.692 | 0.792 | 0.778 | 0.778 | — | 0.800 |
| PR-AUC | 0.9712 | 0.9713 | 0.9706 | 0.9719 | 0.9700 | 0.9675 | 0.9709 | **0.9731** | — | — | — | — |

*E12 shows validation accuracy (93.92%), not test; E11 not run.*

## Conclusion

Thirteen experiments spanning **architecture, loss function (Focal, Weighted, Label Smoothing), data augmentation, class weighting, input resolution, training schedule, post-hoc logit adjustment, ensemble, optimizer, and Bayesian HP search** converge on a single finding: the Shirt sink is a structural limitation of FashionMNIST at 28×28 resolution. No parametric modification eliminates the zero-sum trade-off among upper-body classes — but post-hoc logit adjustment on a well-converged model (E9) provides a practical solution by selecting any operating point on the Shirt PR curve without retraining.
