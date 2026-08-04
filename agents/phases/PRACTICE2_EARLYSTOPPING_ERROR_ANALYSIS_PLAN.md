# 📋 PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md — Comprehensive Upgrade & Retraining Plan

- **Motivation & Background**: Having successfully established the SOTA Transfer Learning benchmarking suite for Practice 2 (achieving a **96.00% Validation Accuracy** Soft-Voting Ensemble record), the pipeline requires two critical enhancements:
  1. **Automated EarlyStopping**: To prevent unnecessary epoch iterations and overfitting, an `EarlyStopping` callback mechanism will monitor `val_loss` and halt training when validation performance plateaus (`patience=3`, `min_delta=1e-4`), automatically restoring the best model weights.
  2. **Retraining & Misclassified Class Error Analysis**: All 6 model variants will undergo fresh 10-epoch retraining with EarlyStopping active. Furthermore, a dedicated **Misclassified Class Error Distribution Bar Plot (Chart 10)** will be added at the end of Section 6 to analyze category-level false negative counts and top confused class pairings across CIFAR-10.
- **Primary Objectives**:
  - Implement `EarlyStopping` class in [`src/training/train_model.py`](../../src/training/train_model.py) and update `train_model()` signature.
  - Update [`scratch/build_notebook.py`](../../scratch/build_notebook.py) to incorporate EarlyStopping, set `NUM_EPOCHS = 10`, force retraining across all 6 model variants, and append `plot_error_distribution()` to Section 6.
  - Regenerate [`notebooks/practice_2.ipynb`](../../notebooks/practice_2.ipynb) (22 cells) following [`NOTEBOOK_HEADER_CONVENTION.md`](../rules/NOTEBOOK_HEADER_CONVENTION.md).
- **Target Files**:
  - [`agents/phases/PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md`](PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md) (This master plan document)
  - [`src/training/train_model.py`](../../src/training/train_model.py) (`EarlyStopping` class & `train_model()` callback)
  - [`scratch/build_notebook.py`](../../scratch/build_notebook.py) (22-cell generator script update)
  - [`notebooks/practice_2.ipynb`](../../notebooks/practice_2.ipynb) (Output target notebook)

---

## Table of Contents
- [1. Full Project Context & Architectural Background](#1-full-project-context--architectural-background)
  - [1.1 Single Split & Reproducibility Policy](#11-single-split--reproducibility-policy)
  - [1.2 SOTA Benchmarking Milestones (EXP-01 to EXP-07)](#12-sota-benchmarking-milestones-exp-01-to-exp-07)
- [2. EarlyStopping Architecture & Algorithmic Formulation](#2-earlystopping-architecture--algorithmic-formulation)
- [3. 10-Epoch Retraining Strategy Across 6 Model Variants](#3-10-epoch-retraining-strategy-across-6-model-variants)
- [4. Misclassified Class Distribution Bar Plot Specification (Chart 10)](#4-misclassified-class-distribution-bar-plot-specification-chart-10)
- [5. Cell-by-Cell Notebook Integration Plan (22 Cells)](#5-cell-by-cell-notebook-integration-plan-22-cells)
- [6. Step-by-Step Implementation Roadmap](#6-step-by-step-implementation-roadmap)
  - [6.1 Step 1: Update `src/training/train_model.py`](#61-step-1-update-srctrainingtrain_modelpy)
  - [6.2 Step 2: Update `scratch/build_notebook.py`](#62-step-2-update-scratchbuild_notebookpy)
  - [6.3 Step 3: Regenerate & Verify `notebooks/practice_2.ipynb`](#63-step-3-regenerate--verify-notebookspractice_2ipynb)
- [7. Verification & Quality Assurance Checklist](#7-verification--quality-assurance-checklist)

---

## 1. Full Project Context & Architectural Background

### 1.1 Single Split & Reproducibility Policy
Per project-wide standards in [`agents/phases/DATALOADER.md`](../phases/DATALOADER.md) and [`agents/phases/DATA_PREP.md`](../phases/DATA_PREP.md):
- **Split File**: Saved at `data/processed/cifar10_split_seed42.json`.
- **Split Breakdown**: Fixed seed `42` dividing 50,000 CIFAR-10 training images into **45,000 Train** and **5,000 Validation** samples, plus the official **10,000 Test** samples.
- **Read-Only Policy**: The split file is strictly read-only across all experiments and notebook runs to ensure fair, reproducible evaluations.

### 1.2 SOTA Benchmarking Milestones (EXP-01 to EXP-07)

| Experiment ID | Focus / Strategy | Best Val Acc (%) | Key Findings & Applied Principles |
| :--- | :--- | :---: | :--- |
| **`EXP-01`** | Optuna HPO | 92.06% | Optimal config: `AdamW`, `lr=8.96e-5`, `weight_decay=3.61e-6`, `batch_size=64`. |
| **`EXP-02`** | LR Schedulers & LLRD | 92.78% | `CosineAnnealingLR` + Layer-wise Discriminative LR Decay avoids catastrophic forgetting. |
| **`EXP-03`** | Advanced Augmentations | 92.72% | `RandAugment(2, 9)` + `RandomErasing(0.25)` + `CE(ls=0.1)` eliminates overfitting. |
| **`EXP-04`** | Native 32x32 Conv Stem | 76.98% | High throughput (45.5s/epoch, ~50% faster, low VRAM) for edge deployment. |
| **`EXP-05`** | Arch Sweep | 96.42% | Evaluated ResNet18 (92.36%), DenseNet121 (90.70%), ConvNeXt-Tiny (96.42%), EfficientNet-B0 (84.44%). |
| **`EXP-06`** | ConvNeXt SOTA | **97.66%** | Ultimate project record with ConvNeXt-Tiny backbone. |
| **`EXP-07`** | ResNet & DenseNet Peak SOTA | **96.00%** | Deep LLRD + Soft-Voting Ensemble ($0.5 P_{\text{ResNet18}} + 0.5 P_{\text{DenseNet121}}$) setting classic backbone record. |

---

## 2. EarlyStopping Architecture & Algorithmic Formulation

### Mathematical & Operational Specification
During training loop execution over $T = 10$ epochs, `EarlyStopping` monitors validation loss $L_{\text{val}}^{(t)}$. If $L_{\text{val}}^{(t)}$ fails to decrease by at least $\delta = 1\times 10^{-4}$ for $P = 3$ consecutive epochs (`patience=3`), training is terminated early and the model parameters $\mathbf{\theta}^{\text{best}}$ corresponding to the minimum validation loss epoch are restored:

$$\text{If } L_{\text{val}}^{(t)} < L_{\text{val}}^{\text{best}} - \delta \implies L_{\text{val}}^{\text{best}} = L_{\text{val}}^{(t)}, \quad \text{counter} = 0, \quad \mathbf{\theta}^{\text{best}} = \text{clone}(\mathbf{\theta}^{(t)})$$

$$\text{If } L_{\text{val}}^{(t)} \ge L_{\text{val}}^{\text{best}} - \delta \implies \text{counter} = \text{counter} + 1$$

$$\text{If counter} \ge P \implies \text{Halt Loop \& Load } \mathbf{\theta}^{\text{best}}$$

### Python Class Definition (`src/training/train_model.py`):

```python
class EarlyStopping:
    """Early stops training if validation loss does not improve after a specified patience."""
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Counter: {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return self.early_stop
```

---

## 3. 20-Epoch Retraining Strategy Across 6 Model Variants

All 6 model variants will be retrained from scratch for a full **20-epoch budget** using `EarlyStopping(patience=4)`:

| Model Variant | Strategy | Data Loader | Loss Function | LR Scheduler | EarlyStopping | Max Epochs |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`ResNet18-frozen`** | Linear Classifier | Standard (224px) | `CrossEntropyLoss()` | None | `patience=4` | 20 |
| **`DenseNet121-frozen`** | Linear Classifier | Standard (224px) | `CrossEntropyLoss()` | None | `patience=4` | 20 |
| **`ResNet18-finetune`** | Unfreeze `layer4` | Standard (224px) | `CrossEntropyLoss()` | None | `patience=4` | 20 |
| **`DenseNet121-finetune`** | Unfreeze `block4` | Standard (224px) | `CrossEntropyLoss()` | None | `patience=4` | 20 |
| **`ResNet18-sota`** | Deep LLRD | SOTA (RandAug) | `CE(label_smoothing=0.1)` | `CosineAnnealingLR` | `patience=4` | 20 |
| **`DenseNet121-sota`** | Deep LLRD | SOTA (RandAug) | `CE(label_smoothing=0.1)` | `CosineAnnealingLR` | `patience=4` | 20 |

---

## 4. Misclassified Class Distribution Bar Plot Specification (Chart 10)

Added at the end of Section 6 in [`scratch/build_notebook.py`](../../scratch/build_notebook.py), `plot_error_distribution()` renders a 2-subplot visual error analysis (`figsize=(16, 6)`):

### Subplot A: Per-Class Misclassification Error Count (False Negatives)
- **X-axis**: 10 CIFAR-10 category names (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
- **Y-axis**: Number of misclassified test instances ($\text{Errors}_c = N_c - \text{CM}_{c,c}$).
- **Visual Styling**: Red gradient color map (`plt.cm.Reds`) with bold exact error count callouts on top of each bar.

### Subplot B: Top 10 Most Confused Class Pairs
- **Y-axis**: Pairwise directional confusion label (`True Class → Predicted Class`).
- **X-axis**: Frequency of directional confusion ($\text{CM}_{r,c}$ where $r \neq c$).
- **Visual Styling**: Horizontal bar chart styled with `plt.cm.magma` color palette.

```python
def plot_error_distribution(eval_results, test_loader, device, class_names):
    """Plot Misclassified Class Error Distribution & Top 10 Confused Class Pairs."""
    best_name = max(eval_results, key=lambda k: eval_results[k]["test_acc"])
    cm = np.array(eval_results[best_name]["confusion_matrix"])
    n_classes = len(class_names)

    # 1. Per-class false negatives
    total_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    errors_per_class = total_per_class - correct_per_class

    # 2. Extract top confused pairs
    confused_pairs = []
    for r in range(n_classes):
        for c in range(n_classes):
            if r != c and cm[r, c] > 0:
                pair_name = f"{class_names[r]} → {class_names[c]}"
                confused_pairs.append((pair_name, cm[r, c]))
    
    confused_pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = confused_pairs[:10]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Subplot A: Per-Class Error Distribution ---
    ax1 = axes[0]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_classes))
    bars = ax1.bar(class_names, errors_per_class, color=colors, edgecolor="black", linewidth=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight="bold")

    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.set_ylabel("Number of Misclassified Samples (Errors)")
    ax1.set_title(f"🏆 {best_name}: Per-Class Misclassification Error Count", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # --- Subplot B: Top 10 Confused Class Pairs ---
    ax2 = axes[1]
    pair_names = [p[0] for p in top_pairs]
    pair_counts = [p[1] for p in top_pairs]
    colors_b = plt.cm.magma(np.linspace(0.4, 0.85, len(top_pairs)))

    bars_b = ax2.barh(pair_names[::-1], pair_counts[::-1], color=colors_b[::-1], edgecolor="black", linewidth=0.8)
    for bar in bars_b:
        width = bar.get_width()
        ax2.annotate(f'{int(width)}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=8, fontweight="bold")

    ax2.set_xlabel("Number of Confusion Instances")
    ax2.set_title(f"🏆 {best_name}: Top 10 Most Confused Class Pairs (True → Pred)", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="x")

    plt.suptitle("Error Analysis: Misclassified Class Distribution & Confusion Pairs", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
```

---

## 5. Cell-by-Cell Notebook Integration Plan (22 Cells)

| Cell ID | Type | Section | Purpose |
| :---: | :---: | :---: | :--- |
| `cell_0` | Markdown | Header | Title, Subtitle, 4-column Roadmap Table ([`NOTEBOOK_HEADER_CONVENTION.md`](../rules/NOTEBOOK_HEADER_CONVENTION.md)) |
| `cell_1` | Markdown | Workflow | 5-Phase Mermaid System Architecture Diagram |
| `cell_2` | Markdown | §1 | Section 1 Heading: *Import Libraries & Detect Environment* |
| `cell_3` | Code | §1 | Package imports (`EarlyStopping` imported from `src.training.train_model`), device detection |
| `cell_4` | Markdown | §2 | Section 2 Heading: *Data Preparation — CIFAR-10 Pipeline* |
| `cell_5` | Code | §2 | DataLoaders, split check, 3 data charts (split bar, class dist, 2x4 augmented grid) |
| `cell_6` | Markdown | §3 | Section 3 Heading: *Model Architecture & LLRD Setup* with mathematical formulas |
| `cell_7` | Code | §3 | SOTA builders, LLRD param groups, 6-variant instantiation, parameter summary table |
| `cell_8` | Markdown | §3.1 | Subsection: *Forward Pass Sanity Check* |
| `cell_9` | Code | §3.1 | Dummy tensor `(4, 3, 224, 224)` verification $\to$ output `(4, 10)` |
| `cell_10` | Markdown | §4 | Section 4 Heading: *Train Models* with Label Smoothing, Cosine Annealing & EarlyStopping math |
| `cell_11` | Code | §4 | `NUM_EPOCHS = 10`, `RUN_CONFIGS`, fresh retraining loop invoking `EarlyStopping(patience=3)` |
| `cell_12` | Markdown | §4.1 | Subsection: *Results Summary* |
| `cell_13` | Code | §4.1 | Formatted ASCII summary table of Best Val Acc / Loss / Epoch per model |
| `cell_14` | Markdown | §5 | Section 5 Heading: *Evaluate & Soft-Voting Ensemble* |
| `cell_15` | Code | §5 | `evaluate_ensemble_with_probs()`, per-variant test eval, soft-voting probability fusion |
| `cell_16` | Markdown | §6 | Section 6 Heading: *Compare & Report — SOTA Visualization Suite* |
| `cell_17` | Code | §6 | **5 Diagnostic Charts**: Per-Model Loss Grid, Per-Feature OvR ROC Grid, Grouped Bar + Table, Confusion Matrices, **Chart 10: Error Distribution Bar Plot** |
| `cell_18` | Markdown | §6.1 | Subsection: *Visualize Predictions* with denormalization math |
| `cell_19` | Code | §6.1 | `denormalize()`, correct & misclassified prediction grids ($2 \times 4$ each) with Softmax confidence % |
| `cell_20` | Markdown | §7 | Section 7 Heading: *Save Models & Results* |
| `cell_21` | Code | §7 | Persist `comparison_table.txt`, `test_metrics.json`, and `training_history.json` |

---

## 6. Step-by-Step Implementation Roadmap

### 6.1 Step 1: Update `src/training/train_model.py`
1. Define `EarlyStopping` class in `src/training/train_model.py`.
2. Update `train_model()` signature to support `early_stopping: bool = True`, `patience: int = 3`, `min_delta: float = 1e-4`.
3. Inside the epoch loop of `train_model()`, check:
   ```python
   if early_stopping:
       if early_stopper(val_loss, model):
           print(f"\n[EarlyStopping] Triggered at epoch {epoch}. Restoring best model weights...")
           model.load_state_dict(early_stopper.best_state_dict)
           break
   ```

### 6.2 Step 2: Update `scratch/build_notebook.py`
1. **Cell 3 (Imports)**: Add `EarlyStopping` import from `src.training.train_model`.
2. **Cell 10 & 11 (Training)**:
   - Ensure `NUM_EPOCHS = 10`.
   - Pass `early_stopping=True, patience=3` to `train_model()`.
3. **Cell 17 (Visualization Suite)**:
   - Add `plot_error_distribution()` function definition.
   - Invoke `plot_error_distribution(eval_results, test_loader, device, CIFAR10_CLASSES)`.

### 6.3 Step 3: Regenerate & Verify `notebooks/practice_2.ipynb`
1. Execute `.venv/bin/python scratch/build_notebook.py`.
2. Verify exit code 0.
3. Validate cell structure and json format of `notebooks/practice_2.ipynb`.

---

## 7. Verification & Quality Assurance Checklist

- [ ] `EarlyStopping` class implemented cleanly in `src/training/train_model.py`.
- [ ] `train_model()` halts early when `val_loss` fails to improve for 3 consecutive epochs and restores best model weights.
- [ ] `scratch/build_notebook.py` runs a fresh 10-epoch retraining loop across all 6 model variants.
- [ ] **Misclassified Class Distribution Bar Plot (Chart 10)** (Subplot A: Per-class false negatives, Subplot B: Top 10 confused pairs) renders at the end of Section 6 in `notebooks/practice_2.ipynb`.
- [ ] Running `.venv/bin/python scratch/build_notebook.py` exits with code 0.
- [ ] Saved benchmark artifacts in `experiments/results/` (`comparison_table.txt`, `test_metrics.json`, `training_history.json`) are updated.
