# EXP_06_CONVNEXT_ADVANCED_SOTA.md — ConvNeXt-Tiny SOTA Combination Experiment Report

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-06`
- **Focus Area:** SOTA Performance Benchmark (Combining ConvNeXt-Tiny + RandAugment + Label Smoothing + CosineAnnealingLR)
- **Target Backbone:** `ConvNeXt-Tiny` (28.6M parameters)
- **Date Executed:** 2026-08-02
- **Status:** Completed (Executed on CUDA GPU)
- **Achieved Accuracy:** **🏆 97.66%** on CIFAR-10 test set!

---

## 🧪 2. Experimental Recipe (SOTA Configuration)

This ultimate experiment fuses the best architectural backbone (`ConvNeXt-Tiny` from EXP-05) with the best regularization strategy (`RandAugment` + `Label Smoothing` from EXP-03) and the best learning rate schedule (`CosineAnnealingLR` from EXP-02).

| Component | Configuration | Purpose |
| :--- | :--- | :--- |
| **Model Architecture** | `ConvNeXt-Tiny` (Fine-tune mode) | Modern 7x7 Depthwise Conv backbone with 28.6M parameters |
| **Data Augmentation** | `RandAugment(num_ops=2, magnitude=9)` + `RandomErasing(p=0.25)` | Prevents overfitting on complex features |
| **Input Resolution** | $224 \times 224$ Upsampling | Matches pretrained ImageNet receptive field |
| **Loss Function** | `CrossEntropyLoss(label_smoothing=0.1)` | Prevents overconfidence on hard labels |
| **Optimizer** | `AdamW(lr=3e-4, weight_decay=1e-4)` | Stable weight decay regularization |
| **Scheduler** | `CosineAnnealingLR(T_max=10, eta_min=1e-6)` | Smooth learning rate decay to global optimum |
| **Epoch Budget** | **10 Epochs** | Allows deep learning of augmented patterns |

---

## 📊 3. Empirical Execution Results

- **Checkpoint Path:** `experiments/checkpoints/exp06_convnext_sota_combination_best.pt`
- **Total Execution Time:** 41.47 minutes (248.8s / epoch)

| Epoch # | Train Loss | Train Accuracy (%) | Val Loss | Val Accuracy (%) | Status / Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Epoch 1** | 0.7262 | 90.87% | 0.6006 | 95.76% | Initial convergence |
| **Epoch 5** | 0.5610 | 97.58% | 0.5708 | 97.00% | Crossed 97.00% barrier |
| **Epoch 10** | **0.5329** | **98.81%** | **0.5584** | **🏆 97.66%** | **ULTIMATE SOTA RECORD ACHIEVED** |

---

## 💡 4. Conclusion & Key Takeaways

1. **New Project Record (97.66%)**: Combining ConvNeXt-Tiny with RandAugment, Label Smoothing (0.1), and CosineAnnealingLR produced the highest accuracy achieved in the entire course repository.
2. **Overfitting Elimination**: Training accuracy reached **98.81%** while validation accuracy followed tightly at **97.66%**, confirming robust generalization.
