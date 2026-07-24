# Lab 1 — PyTorch FashionMNIST Classification
**Course:** Deep Learning  
**Assignment:** Practice 1 — PyTorch FashionMNIST Classification  
**For:** AI coding agent (Claude Code, Cursor, etc.) implementing this lab step by step.  
**Companion documents (read before starting):**
- `agents/ML_PIPELINE_REFERENCE_v3.md` — pipeline concepts still apply (split boundary, baseline first, log everything, 5W on every number)
- `agents/ML_SESSION_GUIDE.md` — output format conventions (results/, plots/, logs/, metrics/)

---

## Context for the AI Agent

This is a **supervised image classification** task using a well-known benchmark dataset.

- **Dataset:** FashionMNIST — 70,000 grayscale 28×28 images across 10 clothing categories
- **Framework:** PyTorch (CUDA if available, CPU fallback)
- **Library constraint:** PyTorch IS allowed in this course (this is a DL course, not the ML from-scratch course)
- **Deliverables required by the assignment:**
  1. Python code (clean, modular, well-commented)
  2. Brief written report
  3. Loss graphs (train loss curve, validation loss curve)
  4. Predicted vs. actual image display grid

**The pipeline concepts from ML_PIPELINE_REFERENCE_v3.md still fully apply:**
- Split boundary: FashionMNIST provides a canonical 60k/10k split — respect it, never shuffle train and test together
- Baseline first (§11): establish a simple model before experimenting
- Single-variable experiments (§18.3): change one hyperparameter per run
- Log every number with a 5W explanation (§19.3)
- Report results as μ ± σ where multiple runs are compared

---

## Output Folder Structure

Enforce this before writing any code:

```
labs/
└── lab1/
    ├── scripts/
    │   └── 2026-07-24_fashionmnist_classification.py   ← main script
    ├── src/
    │   ├── data_loader.py      ← dataset + dataloader logic
    │   ├── model.py            ← network architecture
    │   ├── trainer.py          ← training loop
    │   └── evaluator.py        ← evaluation + metrics
    └── results/
        ├── plots/
        │   ├── 01_sample_grid.png            ← dataset preview
        │   ├── 02_class_distribution.png     ← class balance check
        │   ├── 03_loss_curve_baseline.png    ← baseline training
        │   ├── 04_loss_curve_experiment_NNN.png  ← per experiment
        │   └── 05_predicted_vs_actual.png    ← final deliverable
        ├── logs/
        │   ├── 01_environment.txt            ← device, versions
        │   ├── 02_dataset_summary.txt        ← dataset stats
        │   ├── 03_baseline_training.txt      ← epoch-by-epoch log
        │   └── experiment_NNN_{param}.txt    ← one per tuning run
        ├── metrics/
        │   ├── baseline_results.txt          ← baseline final numbers
        │   ├── experiment_summary.txt        ← all runs compared
        │   └── final_results.txt             ← best model numbers
        └── report/
            └── LAB1_REPORT.md               ← written report (auto-drafted)
```

---

## Operating Rules for the AI Agent

1. **One stage at a time.** Implement one stage completely before moving to the next. Do not write the entire script in one shot.
2. **Ask before assuming.** Every stage has questions — ask them and wait for answers before coding.
3. **Never skip the baseline.** The simplest possible network runs first. Experiments come after.
4. **Single-variable principle (ML_PIPELINE_REFERENCE_v3.md §18.3).** Each experiment changes exactly one thing. Log what changed, what stayed the same, and the result delta.
5. **Every number gets a 5W explanation.** Do not print `Accuracy: 87%` alone — always state what, where, when, why, which.
6. **No plt.show().** All figures save to `results/plots/`. Interactive display breaks in headless/server environments.
7. **Every print() of a metric also writes to a log file** in `results/logs/`.
8. **Sign-off required.** After each stage, show the output and ask: *"Does this look correct? Shall I proceed to Stage [N+1]?"*

---

## Stage 0 — Environment Setup

### Ask the human

```
1. Are you running on CPU or GPU?
   (If GPU: run torch.cuda.is_available() and report the result)

2. What is your PyTorch version? (torch.__version__)

3. Where should the FashionMNIST data be downloaded?
   Default: Deeplearning_Course/data/fashionmnist/

4. Is there a deadline for this lab? (affects how many experiments to run)

5. Do you want the report drafted in English or Vietnamese?
```

### Produce

```python
# Stage 0 — Environment check
# Save output to: results/logs/01_environment.txt

import torch
import torchvision
import sys

info = {
    "Python": sys.version,
    "PyTorch": torch.__version__,
    "Torchvision": torchvision.__version__,
    "CUDA available": torch.cuda.is_available(),
    "CUDA version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    "Device": "cuda" if torch.cuda.is_available() else "cpu",
    "GPU name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
}

for k, v in info.items():
    print(f"{k:20s}: {v}")
```

**Produce this record:**

```
## Stage 0 Output — Environment

| Property | Value |
|----------|-------|
| Python | [version] |
| PyTorch | [version] |
| CUDA available | [yes/no] |
| Device | [cuda / cpu] |
| GPU | [name or N/A] |

Training device confirmed: [DEVICE]
All subsequent training will run on this device.
```

**⚠️ Warning:** If CUDA is not available but the human expects GPU training, stop here and help them diagnose before proceeding. Training FashionMNIST on CPU is feasible but slow for large experiments.

### Sign-off prompt
*"Environment confirmed. Training will run on [device]. Proceed to Stage 1 — Data?"*

---

## Stage 1 — Data Loading and Inspection

**Pipeline reference:** ML_PIPELINE_REFERENCE_v3.md §3 (EDA), §10 (split boundary)

### What to implement

```python
# src/data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# FashionMNIST class labels
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def get_transforms(mean=0.2860, std=0.3530):
    """
    FashionMNIST dataset statistics (computed from training set only):
      mean = 0.2860, std = 0.3530
    These values must be computed from TRAIN SET ONLY (ML_PIPELINE_REFERENCE §10).
    The canonical values above are pre-computed and safe to use directly.
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))   # same stats, not recomputed
    ])
    return train_transform, test_transform


def get_dataloaders(data_dir, batch_size=64, num_workers=2):
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_dataset, test_dataset
```

### Ask the human

```
1. What batch size do you want to start with?
   (Recommended default: 64. GPU with ≥4GB VRAM can use 128 or 256.)

2. Do you want to use a validation split from the training set?
   - YES (recommended): split 60k train → 54k train + 6k validation
     This gives you a validation curve during training, not just test accuracy.
   - NO: use the full 60k for training, evaluate only on 10k test set.
   
   ⚠️ If YES: the validation split must be done BEFORE any transform
   statistics are computed — but since we're using canonical FashionMNIST
   stats (not fitting them), a random split is safe here.
```

### Produce — Dataset Summary

```
## Stage 1 Output — Dataset Summary
Save to: results/logs/02_dataset_summary.txt

Dataset: FashionMNIST
Source: torchvision.datasets (auto-downloaded)

Split sizes:
| Partition | Samples | % |
|-----------|---------|---|
| Train     | 60,000  | 85.7% |
| Test      | 10,000  | 14.3% |
[If validation split used:]
| Train (after split) | 54,000 | 77.1% |
| Validation          |  6,000 |  8.6% |
| Test                | 10,000 | 14.3% |

Image shape: (1, 28, 28) — grayscale, 28×28 pixels
Pixel value range (raw): [0, 255] → normalized to approx. [-1, 1] after transforms
Classes: 10 (balanced — each class has exactly 6,000 train samples)

Normalization (applied to both train and test using TRAIN statistics):
  mean = 0.2860
  std  = 0.3530
```

### Produce — Plot 1: Sample Grid

```python
# Visualize 5×10 grid: one column per class, 5 random samples per class
# Save → results/plots/01_sample_grid.png

import matplotlib.pyplot as plt
import numpy as np

def plot_sample_grid(dataset, class_names, n_per_class=5, save_path=None):
    fig, axes = plt.subplots(n_per_class, len(class_names),
                             figsize=(len(class_names) * 1.5, n_per_class * 1.5))
    fig.suptitle('FashionMNIST — Sample Images per Class', fontsize=13)

    for col, cls_idx in enumerate(range(len(class_names))):
        cls_samples = [dataset[i][0] for i in range(len(dataset))
                       if dataset[i][1] == cls_idx][:n_per_class]
        for row, img in enumerate(cls_samples):
            ax = axes[row][col]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(class_names[cls_idx], fontsize=8, rotation=45, ha='left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Produce — Plot 2: Class Distribution

```python
# Bar chart confirming class balance
# Save → results/plots/02_class_distribution.png
# Expected: perfectly balanced (6,000 per class in train)
```

**⚠️ Leakage note to record:** FashionMNIST provides a canonical pre-split dataset. The test set is already separated — do NOT combine train and test, shuffle, then re-split. The canonical split must be respected. Document this explicitly in the report.

### Sign-off prompt
*"Data loaded. Train: [n] samples, Test: [n] samples. Sample grid saved. Class distribution confirmed [balanced/imbalanced]. Proceed to Stage 2 — Baseline Model?"*

---

## Stage 2 — Baseline Model

**Pipeline reference:** ML_PIPELINE_REFERENCE_v3.md §11 (Baseline Thinking)

> The baseline is the floor. Every architectural decision in later stages is only justified if it beats this number.

### What the baseline is

A **single fully-connected layer** (linear classifier — no hidden layers, no activation):

```
Input: 784 (28×28 flattened) → Output: 10 (classes)
```

This is the simplest possible model that can solve this task. It has no representation power — it can only learn linear decision boundaries in pixel space. Any reasonable neural network should beat it. If it doesn't, the problem is not the network.

### What to implement

```python
# src/model.py

import torch.nn as nn

class BaselineLinear(nn.Module):
    """
    Baseline: single linear layer (no hidden layers, no activation).
    Purpose: establish a lower bound for accuracy before adding complexity.
    Expected accuracy: ~84% on FashionMNIST (from literature).
    """
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)
```

### Ask the human

```
1. How many epochs for the baseline run? (Recommended: 5–10)
   Reason: baseline is fast; we just need a stable convergence number.

2. Learning rate? (Recommended default: 0.001)

3. Optimizer? (Recommended: Adam — standard default for most DL tasks)
```

### What to implement — Trainer

```python
# src/trainer.py

import torch
import torch.nn as nn
import time

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def train(model, train_loader, val_loader, optimizer, criterion,
          device, epochs, log_path):
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}

    with open(log_path, 'w') as f:
        header = f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}"
        print(header); f.write(header + '\n')
        print('-' * len(header)); f.write('-' * len(header) + '\n')

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            vl_loss, vl_acc = evaluate(
                model, val_loader, criterion, device)
            elapsed = time.time() - t0

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(vl_acc)

            row = (f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>9.4f} | "
                   f"{vl_loss:>8.4f} | {vl_acc:>7.4f} | {elapsed:>5.1f}s")
            print(row); f.write(row + '\n')

    return history
```

### Produce — Baseline Training Log

```
## Stage 2 Output — Baseline Training Log
Save to: results/logs/03_baseline_training.txt

Model: BaselineLinear (1 linear layer, no activation)
Hyperparameters:
  optimizer   = Adam
  lr          = 0.001
  batch_size  = 64
  epochs      = [N]
  loss        = CrossEntropyLoss

 Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time
------------------------------------------------------------
     1 |     0.XXXX |    0.XXXX |   0.XXXX |  0.XXXX | XX.Xs
   ...
```

### Produce — Plot 3: Baseline Loss Curve

```python
# src/evaluator.py — plot_loss_curve()

def plot_loss_curve(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train loss', color='steelblue')
    ax1.plot(history['val_loss'],   label='Val loss',   color='coral')
    ax1.set_title(f'{title} — Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train acc', color='steelblue')
    ax2.plot(history['val_acc'],   label='Val acc',   color='coral')
    ax2.set_title(f'{title} — Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Produce — Baseline Final Metrics

```
## Stage 2 Output — Baseline Results
Save to: results/metrics/baseline_results.txt

5W for every number:

Test Accuracy: [value]
  What:  Fraction of 10,000 test images correctly classified
  Where: Test set (canonical FashionMNIST 10k split, never seen during training)
  When:  After epoch [N], final model checkpoint
  Why:   Linear model cannot learn non-linear texture/shape features in pixel space
  Which: All 10 classes combined

Published reference accuracy for single linear layer on FashionMNIST: ~84%
Our result: [value] → [within / above / below] expected range

⚠️ This number is now the FLOOR. Every subsequent model must beat it.
```

### Sign-off prompt
*"Baseline complete. Test accuracy: [X]%. This is our floor. Proceed to Stage 3 — MLP Architecture?"*

---

## Stage 3 — MLP Architecture (Main Model)

**Pipeline reference:** ML_PIPELINE_REFERENCE_v3.md §11 (model selection), §13 (bias-variance)

### Ask the human

```
1. How many hidden layers do you want to start with?
   (Recommended starting point: 2 hidden layers)

2. What hidden layer size? (Recommended: 512 → 256)

3. Which activation function?
   - ReLU (recommended default — fast, works well for image classification)
   - Tanh (smoother gradients, slower)
   - Sigmoid (avoid for hidden layers — vanishing gradient)

4. Do you want Dropout regularization?
   - YES (recommended if you see train_acc >> val_acc — overfitting signal)
   - NO (start without it; add only if needed based on bias-variance diagnosis)

5. Do you want Batch Normalization?
   - YES: stabilizes training, often improves convergence speed
   - NO: simpler, start without it if you want to understand the baseline MLP first
```

### What to implement

```python
# src/model.py — add MLP class

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for FashionMNIST classification.

    Architecture decision log:
      Input:   784 (28×28 flattened grayscale)
      Hidden:  [sizes] with [activation]
      Dropout: [rate or None]
      BatchNorm: [yes/no]
      Output:  10 (classes) — raw logits (no softmax; CrossEntropyLoss handles it)

    Why no softmax at output: PyTorch's CrossEntropyLoss combines
    LogSoftmax + NLLLoss internally. Adding softmax before it would
    cause double-normalization and slow convergence.
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256],
                 num_classes=10, dropout_rate=0.0, batch_norm=False):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        in_size = input_size

        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_size = h_size

        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
```

### Produce — Architecture Record

```
## Stage 3 Output — Architecture Decision Record

Model: MLP
  Input:          784 (28×28 × 1 channel, flattened)
  Hidden layer 1: Linear(784 → 512) → [BatchNorm?] → ReLU → [Dropout(p)?]
  Hidden layer 2: Linear(512 → 256) → [BatchNorm?] → ReLU → [Dropout(p)?]
  Output:         Linear(256 → 10)   (raw logits)

Total parameters: [compute: sum(p.numel() for p in model.parameters())]

Loss function: CrossEntropyLoss
  Why: multi-class classification (10 classes), logits as input.
  Note: includes LogSoftmax internally — do NOT add softmax to model output.

Optimizer: Adam (lr=[value])
  Why: adaptive learning rate, robust default for most DL tasks.

Inductive bias (ML_PIPELINE_REFERENCE §12):
  MLP assumes: features interact through learned weighted combinations.
  Limitation: no spatial awareness — treats each pixel independently.
  FashionMNIST implication: ignores spatial structure of clothing.
  Expected ceiling: ~89-92% (CNN would push higher by exploiting locality).
```

### Sign-off prompt
*"MLP architecture defined. [X] parameters. Proceed to Stage 4 — Training and Experiments?"*

---

## Stage 4 — Training and Hyperparameter Experiments

**Pipeline reference:** ML_PIPELINE_REFERENCE_v3.md §15, §18 (single-variable principle)

### Ask the human

```
1. How many epochs for the main MLP training run? (Recommended: 10–20)

2. Which hyperparameters do you want to experiment with?
   Pick from:
   a) Learning rate         (e.g. compare 0.01 vs 0.001 vs 0.0001)
   b) Hidden layer size     (e.g. compare [512,256] vs [256,128] vs [1024,512])
   c) Dropout rate          (e.g. compare 0.0 vs 0.3 vs 0.5)
   d) Batch size            (e.g. compare 32 vs 64 vs 128)
   e) Number of layers      (e.g. compare 1 vs 2 vs 3 hidden layers)

3. Confirm you understand the single-variable rule:
   Each experiment changes EXACTLY ONE item from the list above.
   Everything else stays at the base configuration.
```

### Per-experiment template

**Before each experiment, confirm with the human:**

```
Experiment [NNN] — [parameter name]

Base configuration (unchanged):
  hidden_sizes   = [512, 256]
  lr             = 0.001
  dropout        = 0.0
  batch_size     = 64
  epochs         = 15

What changes in this experiment:
  [parameter]    = [new value]   (was: [old value])

Why this value: [reason — e.g. "testing if lower lr reduces val loss spike at epoch 3"]
```

### Per-experiment output

```python
# All experiments use the same train() function from src/trainer.py
# Save log to: results/logs/experiment_NNN_{param}.txt
# Save plot to: results/plots/04_loss_curve_experiment_NNN.png
```

```
## Experiment [NNN] Summary
Save to: results/logs/experiment_NNN_{param}.txt

What changed:    [param] = [new value]  (base: [old value])
Held constant:   [everything else listed]

Results:
  Final train acc: [value]
  Final val acc:   [value]
  Final test acc:  [value — only check if this is the final chosen config]

Bias-variance diagnosis (ML_PIPELINE_REFERENCE §13):
  Train acc: [value]
  Val acc:   [value]
  Gap:       [value]
  Verdict:   [underfitting (gap < 0.02, both low) /
              good fit (gap < 0.05, both high) /
              overfitting (gap > 0.05)]

Compared to baseline:
  Baseline test acc:       [value]
  This experiment val acc: [value]
  Delta:                   [+/- value]
  Verdict:                 [improvement / regression / within noise]

Decision: [keep this value as new base / revert / explore further]
```

### Produce — Experiment Summary Table

After all experiments are done:

```
## Stage 4 Output — Experiment Summary
Save to: results/metrics/experiment_summary.txt

Base config: hidden=[512,256], lr=0.001, dropout=0.0, batch=64, epochs=15

| Exp | Changed param | Value | Train Acc | Val Acc | Gap | vs Baseline | Decision |
|-----|--------------|-------|-----------|---------|-----|-------------|----------|
| 000 | baseline     | —     | [v]       | [v]     | [v] | floor       | reference |
| 001 | lr           | 0.01  | [v]       | [v]     | [v] | [+/-]       | [keep/revert] |
| 002 | lr           | 0.0001| [v]       | [v]     | [v] | [+/-]       | [keep/revert] |
...

Best configuration: [param values]
Best val accuracy:  [value]
```

### Sign-off prompt
*"All experiments logged. Best config: [params], val acc = [X]%. Proceed to Stage 5 — Final Evaluation?"*

---

## Stage 5 — Final Evaluation

**Pipeline reference:** ML_PIPELINE_REFERENCE_v3.md §13, §16, §18.4

**⚠️ Test set is used here for the FIRST AND ONLY TIME for the chosen model.**

### What to implement

```python
# src/evaluator.py

import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
# Note: sklearn is allowed in this course for evaluation utilities only.
# The model itself is implemented in PyTorch.

def full_evaluation(model, test_loader, class_names, device, results_dir):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # Save to file
    with open(f'{results_dir}/metrics/final_results.txt', 'w') as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title('Confusion Matrix — FashionMNIST Test Set')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plots/05_confusion_matrix.png', dpi=150)
    plt.close()

    return all_preds, all_labels
```

### Produce — Final Metrics Table

```
## Stage 5 Output — Final Evaluation
Save to: results/metrics/final_results.txt

Model: MLP — [final architecture and hyperparameters]
Evaluated on: FashionMNIST test set (10,000 samples, seen for the first time)

Per-class results:

| Class          | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| T-shirt/top    |           |        |          | 1000    |
| Trouser        |           |        |          | 1000    |
| Pullover       |           |        |          | 1000    |
| Dress          |           |        |          | 1000    |
| Coat           |           |        |          | 1000    |
| Sandal         |           |        |          | 1000    |
| Shirt          |           |        |          | 1000    |
| Sneaker        |           |        |          | 1000    |
| Bag            |           |        |          | 1000    |
| Ankle boot     |           |        |          | 1000    |
| **Overall**    |           |        |          | 10000   |

5W for overall accuracy:
  What:  Fraction of 10,000 test samples correctly classified
  Where: FashionMNIST canonical test split (10k samples)
  When:  Final model, after [N] epochs, best checkpoint
  Why:   [explain relative to baseline and experiments]
  Which: All 10 classes combined; see per-class table for breakdown

Comparison to baseline:
  Baseline accuracy:   [value]
  Final model accuracy: [value]
  Absolute improvement: [+value]

Expected MLP range from literature: 88–92%
Our result: [value] → [within / above / below] expected range
```

### Produce — Plot: Predicted vs. Actual Grid

```python
# Assignment deliverable: display predicted vs actual images
# Save → results/plots/05_predicted_vs_actual.png

def plot_predictions(model, test_loader, class_names, device,
                     n_rows=4, n_cols=8, save_path=None):
    """
    Grid of n_rows × n_cols test images.
    Title of each image: Actual: [class] / Pred: [class]
    Correct predictions: green title. Wrong predictions: red title.
    """
    model.eval()
    images_shown, labels_shown, preds_shown = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            images_shown.extend(images.cpu())
            labels_shown.extend(labels)
            preds_shown.extend(preds.cpu())
            if len(images_shown) >= n_rows * n_cols:
                break

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.8, n_rows * 2.2))
    fig.suptitle('Predicted vs. Actual — FashionMNIST Test Set',
                 fontsize=13, y=1.01)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(images_shown):
            ax.axis('off')
            continue
        img = images_shown[idx].squeeze().numpy()
        label = labels_shown[idx].item()
        pred  = preds_shown[idx].item()
        correct = (label == pred)

        ax.imshow(img, cmap='gray')
        ax.axis('off')
        color = 'green' if correct else 'red'
        ax.set_title(
            f"A: {class_names[label]}\nP: {class_names[pred]}",
            fontsize=7, color=color
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Sign-off prompt
*"Final evaluation complete. Test accuracy: [X]%. Confusion matrix and prediction grid saved. Proceed to Stage 6 — Save Model and Report?"*

---

## Stage 6 — Model Saving and Written Report

**Assignment deliverable:** Python code + brief report + loss graphs + image displays

### Model saving

```python
# Save final model checkpoint
# Path: model/backup/2026-07-24_fashionmnist_mlp.pt

import os
from datetime import date

def save_checkpoint(model, optimizer, epoch, val_acc, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config,            # dict of all hyperparameters
        'date': str(date.today()),
    }
    filename = f"{date.today()}_fashionmnist_mlp.pt"
    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_acc={checkpoint['val_acc']:.4f}")
    return checkpoint
```

### Produce — Written Report Draft

Auto-draft this report from all stage outputs. Save to `results/report/LAB1_REPORT.md`.

```markdown
# Lab 1 Report — FashionMNIST Classification with PyTorch

**Student:** [name]
**Date:** [date]
**Dataset:** FashionMNIST (70,000 images, 10 classes)
**Framework:** PyTorch [version], Device: [cpu/cuda]

---

## 1. Problem Statement

Image classification task: assign each 28×28 grayscale image to one of
10 clothing categories. This is a supervised multi-class classification
problem (ML_PIPELINE_REFERENCE §1).

**Success criterion:** Exceed baseline linear classifier accuracy ([value]%)
using a Multi-Layer Perceptron with PyTorch.

---

## 2. Dataset

FashionMNIST provides 60,000 training and 10,000 test images.
The dataset is class-balanced: 6,000 samples per class in training.
The canonical train/test split was respected without modification
(ML_PIPELINE_REFERENCE §10 — split boundary maintained).

Normalization applied to both splits using training-set statistics:
  mean = 0.2860, std = 0.3530

*Figure 1: Sample images per class (results/plots/01_sample_grid.png)*
*Figure 2: Class distribution (results/plots/02_class_distribution.png)*

---

## 3. Baseline Model

A single linear layer (784 → 10, no hidden units, no activation) was
trained as the performance floor (ML_PIPELINE_REFERENCE §11).

| Metric | Value |
|--------|-------|
| Test Accuracy | [value] |
| Expected (literature) | ~84% |

*Figure 3: Baseline loss curve (results/plots/03_loss_curve_baseline.png)*

---

## 4. MLP Architecture

[Architecture from Stage 3 decision record — insert here]

Total parameters: [value]

---

## 5. Experiments

The single-variable principle (ML_PIPELINE_REFERENCE §18.3) was applied:
each experiment changed exactly one hyperparameter while holding all
others constant.

[Insert experiment summary table from Stage 4]

*Figure 4–N: Loss curves per experiment (results/plots/04_loss_curve_experiment_*.png)*

**Bias-variance diagnosis (ML_PIPELINE_REFERENCE §13):**
[Insert diagnosis from best experiment — train/val gap analysis]

---

## 6. Final Results

Best configuration: [hyperparameters]

| Metric | Baseline | Final MLP | Improvement |
|--------|----------|-----------|-------------|
| Test Accuracy | [v] | [v] | [+v] |
| Precision (macro) | — | [v] | — |
| Recall (macro) | — | [v] | — |
| F1-score (macro) | — | [v] | — |

*Figure N: Confusion matrix (results/plots/05_confusion_matrix.png)*
*Figure N+1: Predicted vs. actual (results/plots/05_predicted_vs_actual.png)*

**Hardest classes (lowest F1):** [class names — read from confusion matrix]
**Likely reason:** [e.g. Shirt vs. T-shirt are visually similar in grayscale]

---

## 7. Conclusions

[3–5 sentences summarizing: what was achieved, how it compares to baseline
and to expected MLP range (88–92%), and one concrete next step — e.g.
CNN would exploit spatial structure that MLP ignores]
```

### Sign-off prompt
*"All deliverables complete. Summary of outputs:"*

```
## Stage 6 Output — Deliverables Index

CODE:
  labs/lab1/scripts/2026-07-24_fashionmnist_classification.py
  labs/lab1/src/data_loader.py
  labs/lab1/src/model.py
  labs/lab1/src/trainer.py
  labs/lab1/src/evaluator.py

PLOTS (required by assignment):
  results/plots/03_loss_curve_baseline.png      ← loss graph ✓
  results/plots/04_loss_curve_experiment_*.png  ← loss graphs ✓
  results/plots/05_predicted_vs_actual.png      ← predicted vs actual ✓
  results/plots/05_confusion_matrix.png         ← bonus

LOGS AND METRICS:
  results/logs/01_environment.txt
  results/logs/02_dataset_summary.txt
  results/logs/03_baseline_training.txt
  results/logs/experiment_NNN_*.txt             ← one per experiment
  results/metrics/baseline_results.txt
  results/metrics/experiment_summary.txt
  results/metrics/final_results.txt

REPORT:
  results/report/LAB1_REPORT.md                ← brief report ✓

MODEL:
  model/backup/2026-07-24_fashionmnist_mlp.pt  ← saved checkpoint ✓
```

---

## Quick Reference — Stage Order

| Stage | Name | Key output | Gate condition |
|-------|------|-----------|----------------|
| 0 | Environment setup | `01_environment.txt` | CUDA status confirmed |
| 1 | Data loading | Sample grid, class dist plots | Split boundary documented |
| 2 | Baseline model | `baseline_results.txt`, loss curve | Floor number recorded |
| 3 | MLP architecture | Architecture decision record | Inductive bias documented |
| 4 | Training + experiments | Experiment summary table | Single-variable rule followed |
| 5 | Final evaluation | `final_results.txt`, prediction grid | Test set used exactly once |
| 6 | Save + report | All deliverables indexed | All 4 assignment items present |

---

## Known Pitfalls — FashionMNIST + PyTorch

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Softmax before CrossEntropyLoss | NaN loss or very slow convergence | Remove softmax from model output — CE loss applies it internally |
| Forgetting `model.eval()` before evaluation | Dropout/BN behave differently at test time | Always call `model.eval()` and `torch.no_grad()` during evaluation |
| Normalizing test set with test statistics | Data leakage (§10) | Use train-set mean/std (0.2860, 0.3530) for both splits |
| Calling `plt.show()` in a script | Blocks execution on servers | Always use `plt.savefig()` + `plt.close()` |
| Not zeroing gradients | Gradients accumulate across batches | `optimizer.zero_grad()` at the start of every training step |
| Changing multiple hyperparameters per experiment | Results unattributable | Single-variable principle (§18.3) — one change per run |
| Looking at test accuracy during tuning | Implicit data leakage | Use val_acc for all decisions; test is evaluated once at the end |
