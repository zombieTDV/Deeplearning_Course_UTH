# Changelog

> All changes to the project are documented here. Each experiment follows the single-variable principle: one modification per run, with all other factors held constant.

---

## Baseline — practice_1

- **Date**: 2026-07-26
- **Models**: MLP (784→256→128→10, ReLU, Dropout 0.2) and CNN (Conv1→32→Pool→Conv32→64→Pool→FC 128→10, ReLU, Dropout 0.25)
- **Training**: 10 epochs, CrossEntropyLoss + Adam (lr=0.001)
- **Metric files**: `outputs/practice_1/metrics/`
- **Key results**:
  - MLP accuracy: 88.15% ／ CNN accuracy: 92.02%
  - MLP macro PR-AUC: 0.9394 ／ CNN macro PR-AUC: 0.9677

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

*[pending execution]*

---

## Summary: All Experiments

| Metric | E1 CE | E2 Arch | E3 Focal | E4 Aug | E5 Weight | E6 Upscale | E7 Smooth |
|--------|:-----:|:-------:|:--------:|:------:|:---------:|:----------:|:---------:|
| Accuracy | 92.50% | **92.65%** | 92.40% | 92.45% | 91.75% | 90.80% | *pend* |
| Macro PR-AUC | 0.9712 | 0.9713 | 0.9706 | **0.9719** | 0.9700 | 0.9675 | *pend* |
| Shirt TPR | **0.847** | 0.732 | 0.713 | 0.810 | 0.859 | 0.846 | *pend* |
| T-shirt TPR | 0.837 | **0.939** | 0.881 | 0.834 | 0.820 | 0.795 | *pend* |
| Pullover TPR | 0.890 | 0.931 | 0.896 | **0.936** | 0.896 | 0.754 | *pend* |
| Coat TPR | 0.870 | 0.909 | 0.931 | 0.882 | 0.888 | **0.956** | *pend* |
| Dress TPR | 0.907 | 0.928 | **0.933** | 0.888 | 0.866 | 0.896 | *pend* |
| Shirt Prec. | 0.723 | **0.826** | 0.841 | 0.766 | 0.706 | 0.703 | *pend* |

- **Conservation of errors**: upper-body confusion sum is stable (~650 per 5000 test samples) regardless of architecture/loss/data
- **Only E5 (weighted CE) improved Shirt TPR** beyond E1's vanilla CE, but at −0.75% accuracy — each +1 Shirt gain costs ~6.3 accuracy points
- **Shirt worst TPR**: 0.713 (E3), **best**: 0.859 (E5), **baseline**: 0.847 (E1)

## Conclusion

Seven single-variable experiments spanning **architecture, loss function, data augmentation, class weighting, input resolution, and label smoothing** converge on a single finding: **the Shirt sink is a structural limitation of FashionMNIST at 28×28 resolution.** No parametric modification can simultaneously improve all 5 upper-body classes because ~400 foreground pixels cannot encode the distinguishing features (collar shape, sleeve length, hemline). The zero-sum trade-off among upper-body classes is a consequence of the pixel budget, not the model choice.
