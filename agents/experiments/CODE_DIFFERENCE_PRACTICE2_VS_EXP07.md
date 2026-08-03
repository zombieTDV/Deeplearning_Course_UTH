# CODE_DIFFERENCE_PRACTICE2_VS_EXP07.md — Code Architecture & Technical Differences Analysis

## 📌 Executive Summary

This document provides a comprehensive, line-by-line and architectural comparison between the exploratory baseline code in [`../../notebooks/practice_2.ipynb`](../../notebooks/practice_2.ipynb) and the production SOTA optimization module in [`../../src/experiments/exp_07_resnet_densenet_sota.py`](../../src/experiments/exp_07_resnet_densenet_sota.py).

While `practice_2.ipynb` establishes the foundational baseline pipeline for transfer learning, `exp_07_resnet_densenet_sota.py` introduces advanced deep learning techniques (Deep LLRD, RandAugment, Label Smoothing, Cosine Annealing, and Soft-Voting Ensembling) to push `ResNet18` and `DenseNet121` to their peak classification performance on CIFAR-10 (**96.00% Ensemble Accuracy**).

---

## 📊 Summary Comparison Matrix

| Technical Category | `notebooks/practice_2.ipynb` (Baseline) | `src/experiments/exp_07_resnet_densenet_sota.py` (EXP-07 Peak SOTA) | Impact / Performance Gain |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Educational baseline transfer learning pipeline | Peak SOTA accuracy optimization & model ensembling | Achieved **96.00% Val Acc** vs 92.36% baseline |
| **Data Augmentation** | Basic Resize (224x224) + Normalization | `RandAugment(num_ops=2, mag=9)` + `RandomErasing(p=0.25)` | Eliminates training set overfitting |
| **Model Unfreezing** | Single-block (`layer4` / `denseblock4`) or Frozen | Deep Discriminative LLRD (`layer3+layer4` / `denseblock3+4`) | Unlocks deep feature adaptation (+2.36% to +4.30%) |
| **Learning Rate Policy** | Fixed Constant LR (`1e-3` frozen / `1e-4` finetune) | Discriminative Per-Layer LR + `CosineAnnealingLR` | Smooth loss decay down to `eta_min=1e-6` |
| **Loss Function** | Standard Hard One-Hot `CrossEntropyLoss()` | Regularized `CrossEntropyLoss(label_smoothing=0.1)` | Prevents model overconfidence on hard labels |
| **Model Combination** | Evaluates models independently | **Soft-Voting Ensemble** ($0.5 P_{\text{ResNet}} + 0.5 P_{\text{DenseNet}}$) | **🏆 Peak Classic Record (96.00%)** |
| **Output Artifacts** | Per-class accuracy, confusion matrix, sample grids | PyTorch SOTA checkpoints, benchmark metrics, plot suites | Production-ready checkpoints & reporting |

---

## 🔍 Detailed Component-by-Component Comparison

### 1. Data Transforms & Augmentation Pipeline

- **`practice_2.ipynb` (Cell 6)**:
  Uses the standard data loading pipeline without heavy data augmentations:
  ```python
  train_loader, val_loader, test_loader = get_cifar10_loaders(
      batch_size=64, num_workers=2
  )
  ```
- **`exp_07_resnet_densenet_sota.py` (Lines 77-87)**:
  Applies advanced AutoAugment and Random Erasing regularization:
  ```python
  train_transform = get_advanced_train_transform(
      resize_size=224,
      use_randaugment=True,     # RandAugment(num_ops=2, magnitude=9)
      use_random_erasing=True,  # RandomErasing(p=0.25)
  )
  eval_transform = get_eval_transform(resize_size=224)
  ```

---

### 2. Model Unfreezing & Layer-wise Learning Rate Decay (LLRD)

- **`practice_2.ipynb` (Cell 8)**:
  Uses predefined single-block unfreezing (`mode="finetune"` unfreezes only `layer4` for ResNet18 and `denseblock4` for DenseNet121):
  ```python
  rn_finetune = build_resnet18(num_classes=10, mode="finetune", device=device)
  dn_finetune = build_densenet121(num_classes=10, mode="finetune", device=device)
  ```
- **`exp_07_resnet_densenet_sota.py` (Lines 22-38)**:
  Defines custom deep unfreezing functions unfreezing both mid-level and deep feature blocks:
  ```python
  def build_resnet18_full_sota(num_classes: int = 10, device=None):
      model = build_resnet18(num_classes=num_classes, mode="frozen", device=device)
      set_parameter_requires_grad(model.layer3, True)  # Mid-stage unfreezing
      set_parameter_requires_grad(model.layer4, True)  # Deep-stage unfreezing
      set_parameter_requires_grad(model.fc, True)
      return model
  ```

---

### 3. Optimizer Parameter Groups & Discriminative Learning Rates

- **`practice_2.ipynb` (Cell 12)**:
  Assigns a uniform single learning rate to all trainable parameters:
  ```python
  optimizer = optim.AdamW(
      [p for p in model.parameters() if p.requires_grad],
      lr=1e-4, weight_decay=1e-4
  )
  ```
- **`exp_07_resnet_densenet_sota.py` (Lines 98-103, 119-125)**:
  Applies discriminative per-layer learning rate decay so deeper feature blocks update slower than the new classification head:
  ```python
  param_groups_resnet = [
      {"params": model_resnet.fc.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
      {"params": model_resnet.layer4.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
      {"params": model_resnet.layer3.parameters(), "lr": 3e-5, "weight_decay": 1e-4},
  ]
  optimizer_resnet = torch.optim.AdamW(param_groups_resnet)
  ```

---

### 4. Learning Rate Scheduling & Loss Regularization

- **`practice_2.ipynb` (Cell 12)**:
  Trains models with a constant learning rate over 10 epochs using unregularized Cross-Entropy Loss:
  ```python
  criterion = nn.CrossEntropyLoss()
  # No learning rate scheduler used
  ```
- **`exp_07_resnet_densenet_sota.py` (Lines 89, 104, 126)**:
  Uses Label Smoothing (0.1) and Cosine Annealing learning rate decay down to `1e-6`:
  ```python
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  scheduler_resnet = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer_resnet, T_max=epochs, eta_min=1e-6
  )
  ```

---

### 5. Model Ensembling (Soft-Voting Probability Averaging)

- **`practice_2.ipynb` (Cells 14-22)**:
  Evaluates models individually and compares their test metrics without combining predictions.
- **`exp_07_resnet_densenet_sota.py` (Lines 40-67)**:
  Implements a **Soft-Voting Probability Ensemble** that averages predicted probabilities from both `ResNet18` and `DenseNet121`:
  ```python
  prob_resnet = torch.softmax(model_resnet(images), dim=1)
  prob_densenet = torch.softmax(model_densenet(images), dim=1)
  prob_ensemble = 0.5 * (prob_resnet + prob_densenet)
  ```

---

## 📈 Empirical Result Comparison

| Model Configuration | `practice_2.ipynb` (Val Acc) | `exp_07_resnet_densenet_sota.py` (Val Acc) | Improvement |
| :--- | :---: | :---: | :---: |
| **ResNet18 Fine-tune** | 92.36% | **94.72%** | **+2.36%** |
| **DenseNet121 Fine-tune** | 90.70% | **95.00%** | **+4.30%** |
| **ResNet18 + DenseNet121 Ensemble** | N/A | **96.00%** | **+3.64% (vs Baseline)** |

---

## 💡 Key Architectural Takeaways

1. **Evolution from Baseline to SOTA**: `practice_2.ipynb` serves as an interactive exploratory notebook for standard fine-tuning. `EXP-07` converts these models into production SOTA performers by incorporating deep discriminative LLRD, RandAugment, Label Smoothing, and Cosine Annealing.
2. **Complementary Feature Ensemble**: Combining residual skip connections (`ResNet18`) with dense feature reuse (`DenseNet121`) via soft-voting probability averaging achieved **96.00% Validation Accuracy**, establishing the project record for classic CNN backbones.
