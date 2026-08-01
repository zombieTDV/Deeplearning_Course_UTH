# EXP_03_ADVANCED_AUGMENTATIONS.md — Advanced Data Augmentation & Regularization Plan

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-03`
- **Focus Area:** Data Augmentation (RandAugment, CutMix, Mixup) & Regularization
- **Target Backbone:** ResNet18 & DenseNet121
- **Objective:** Prevent overfitting and improve generalization performance on CIFAR-10 test set to push accuracy past **94%**.

---

## 🧪 2. Experimental Configurations

| Variant | Augmentations / Regularization | Loss Function |
| :--- | :--- | :--- |
| **`Baseline`** | RandomCrop(32, padding=4), RandomHorizontalFlip | Standard CrossEntropy |
| **`RandAugment`** | `RandAugment(num_ops=2, magnitude=9)` | Standard CrossEntropy |
| **`CutMix / Mixup`** | `CutMix(alpha=1.0)` / `Mixup(alpha=0.8)` (50% probability) | Soft-label CrossEntropy |
| **`Label Smoothing`** | Standard transforms + `RandomErasing(p=0.2)` | `CrossEntropyLoss(label_smoothing=0.1)` |

---

## 🎯 3. Success Criteria

- Reduce generalization gap (`train_acc - val_acc`) to $< 2.0\%$.
- Reach Top-1 accuracy **$\ge 94.0\%$** on CIFAR-10 test set.
