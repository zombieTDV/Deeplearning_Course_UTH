# LAB2 — CIFAR-10 Transfer Learning (ResNet18, DenseNet121, ConvNeXt)

- **Motivation/Background**: Second laboratory of the Deep Learning course. Focuses on transfer learning using pretrained ImageNet backbones on CIFAR-10.
- **Purpose**: Document the full experimental pipeline, SOTA achievements, error analysis, and deliverables for LAB2.
- **Overview Pipeline**:
  1. Transfer learning baselines: ResNet18 and DenseNet121 feature extractors vs fine-tuning.
  2. Training stability & regularization: CosineAnnealingLR, Layer-wise Learning Rate Decay (LLRD), RandAugment, and Label Smoothing.
  3. Architecture sweep: ResNet18, DenseNet121, EfficientNet-B0, and ConvNeXt-Tiny.
  4. SOTA combinations: ConvNeXt-Tiny reaching **97.66% validation accuracy** (all-time course record).
  5. Ensembling & error reduction: Soft-voting ensembles, Class-Logit Bias Sweeping (resolving cat-dog confusion), Mixture of Experts (MoE), and Stacking MLPs.
  6. Deep feature analysis: 2D UMAP projections of deep representations across backbones.
- **References**: `src/lab2/`, `notebooks/lab2/`, `experiments/lab2/`, `docs/lab2/experiments/`, `docs/lab2/phases/`.

- **Created**: 2026-09-06T13:14:10+07:00
- **Last Updated**: 2026-09-06T13:14:10+07:00

---

## 1. Key Results Summary

| Experiment ID | Strategy | Backbone | Best Val Acc (%) | Test Acc (%) | Key Finding |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`EXP-06`** | SOTA Combination | **ConvNeXt-Tiny** | **97.66%** | **97.55%** | **All-time project record** (RandAugment + Label Smoothing + CosineAnnealing). |
| **`Logit Bias Sweep`** | Soft-Voting Ensemble | **ResNet18 + DenseNet121** | **97.28%** | **96.73%** | Peak classic backbone ensemble with cat-dog logit calibration. |
| **`EXP-05`** | Architecture Sweep | ConvNeXt-Tiny | **96.42%** | **96.35%** | Modern 7x7 depthwise convolutions outperforming classic CNNs. |
| **`EXP-07`** | ResNet18 + DenseNet121 Ensemble | Ensemble | **96.00%** | **95.95%** | Peak classic backbone baseline ensemble. |
| **`EXP-07`** | Deep LLRD + RandAug | DenseNet121 | **95.00%** | **94.90%** | Single DenseNet121 SOTA (+4.30% over baseline). |
| **`EXP-07`** | Deep LLRD + RandAug | ResNet18 | **94.72%** | **94.65%** | Single ResNet18 SOTA (+2.36% over baseline). |

Detailed reports for every experiment are located in [`experiments/`](experiments/).

---

## 2. Directory Layout & Artifacts

| Component | Location | Description |
| :--- | :--- | :--- |
| **Source Modules** | `src/lab2/` | Modular package: `data`, `models`, `training`, `eval`, `experiments`, `eda`, `utils`. |
| **Notebooks** | `notebooks/lab2/` | Deliverable notebook `practice_2.ipynb` and 6 satellite analysis notebooks. |
| **Visualizations** | `experiments/lab2/plots/` | ~60 plots covering loss curves, Pareto frontiers, confusion matrices, and UMAP embeddings. |
| **Persisted Metrics** | `experiments/lab2/results/` | `benchmark_metrics.json`, `logit_bias_sweep_results.json`, UMAP coordinates. |
| **Phase Specs** | `docs/lab2/phases/` | Detailed documentation for all 13 pipeline phases. |
| **Progress Logs** | `docs/lab2/progress/` | Task status and execution logs. |
| **Bug Reports** | `docs/lab2/bugs/` | Post-mortems for BrokenPipe and KeyError issues. |
