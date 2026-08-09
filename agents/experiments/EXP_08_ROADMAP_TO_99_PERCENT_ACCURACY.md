# EXP_08_ROADMAP_TO_99_PERCENT_ACCURACY.md — ConvNeXt-Small SOTA Peak Benchmark Report

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-08`
- **Focus Area:** Ultra-Peak SOTA Optimization (`ConvNeXt-Small` + `Mixup/CutMix` + `EMA` + `AMP FP16` + `CosineAnnealingLR`)
- **Target Backbone:** `ConvNeXt-Small` (50.0M parameters)
- **Date Executed:** 2026-08-05
- **Status:** Completed (Executed on NVIDIA GeForce RTX 3050 Laptop GPU)
- **Achieved Accuracy:** **🏆 98.82% Validation Accuracy** / **🏆 98.62% Test Accuracy** (**NEW ALL-TIME PROJECT RECORD!**)

---

## 🧪 2. Experimental Recipe (SOTA Peak Configuration)

This benchmark fuses modern Vision Backbone architecture with mixed-sample data augmentations, weight smoothing, and low-precision GPU acceleration:

| Component | Configuration | Purpose |
| :--- | :--- | :--- |
| **Model Architecture** | `ConvNeXt-Small` (Fine-tune mode) | Modern 7x7 Depthwise Conv backbone with 50.0M parameters |
| **Input Resolution** | $224 \times 224$ Upsampling (BICUBIC) | Matches ImageNet receptive field; optimized for 4GB VRAM |
| **Mixed Augmentations** | `Mixup(α=0.8)` + `CutMix(α=1.0)` + `RandAugment(M=12)` | Eliminates overconfidence & creates soft decision boundaries |
| **Weight Averaging** | **EMA (Exponential Moving Average, decay=0.9999)** | Smooths loss landscape to converge into flat minima |
| **Precision Mode** | **Automatic Mixed Precision (AMP FP16)** | Reduces VRAM consumption by 50% & accelerates GPU throughput |
| **Optimizer** | `AdamW(lr=1e-4, weight_decay=0.05)` | Stable weight decay regularization for deep vision backbones |
| **Scheduler** | `CosineAnnealingLR(T_max=10, eta_min=1e-6)` | Smooth learning rate decay to global optimum |
| **Epoch Budget** | **10 Epochs** | Fast convergence under AMP acceleration (~650s / epoch) |

---

## 📊 3. Empirical Execution Results

- **Jupyter Notebook:** [`notebooks/exp_8.ipynb`](file:///home/bush/Desktop/Deeplearning_Course_UTH/notebooks/exp_8.ipynb)
- **Checkpoint Path:** [`experiments/checkpoints/exp08_convnext_small_sota_best.pt`](file:///home/bush/Desktop/Deeplearning_Course_UTH/experiments/checkpoints/exp08_convnext_small_sota_best.pt)
- **Total Execution Time:** ~109 minutes (655s / epoch average)

| Epoch # | Train Loss | EMA Val Accuracy (%) | Status / Notes |
| :---: | :---: | :---: | :--- |
| **Epoch 1** | 0.8490 | 54.60% | Initial convergence |
| **Epoch 2** | 0.7126 | 90.82% | Crossed 90% threshold |
| **Epoch 3** | 0.6922 | 96.22% | Rapid SOTA convergence |
| **Epoch 5** | 0.6315 | 97.88% | Surpassed EXP-06 baseline (97.66%) |
| **Epoch 7** | 0.5790 | 98.44% | Crossed 98.40% barrier |
| **Epoch 9** | 0.5414 | 98.68% | Approaching peak minima |
| **Epoch 10** | **0.5310** | **🏆 98.82%** | **NEW ALL-TIME VALIDATION RECORD** |

### 🏆 Final Official Test Set Benchmark (10,000 samples):
- **EXP-08 Standard Test Accuracy:** **🏆 98.62%**
- **Validation-to-Test Generalization Gap:** **0.20%** (98.82% Val vs 98.62% Test) — Confirming near-perfect generalization!

---

## 📈 4. Comparison with Previous Project Records

| Experiment ID | Backbone | Strategy | Val Accuracy | Test Accuracy | Status / Highlight |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`EXP-08 (Current)`** | **ConvNeXt-Small** | **Mixup + CutMix + EMA + AMP** | **🏆 98.82%** | **🏆 98.62%** | **👑 ALL-TIME PROJECT RECORD (98.62% Test)** |
| `EXP-06` | ConvNeXt-Tiny | RandAugment + Label Smoothing | 97.66% | 97.12% | Previous single model record |
| `Logit Bias Sweep` | ResNet18 + DenseNet121 | Soft-Voting Ensemble + Bias Sweep | 97.28% | 96.73% | Classic backbone peak ensemble |
| `EXP-07` | ResNet18 + DenseNet121 | Deep LLRD Ensemble | 96.00% | 95.95% | Baseline classic ensemble |

---

## 💡 5. Conclusion & Key Takeaways

1. **New All-Time Project Record (98.62% Test / 98.82% Val)**: `EXP_08` shattered the previous project record set by `EXP_06` (+1.50% Test accuracy boost).
2. **Impact of Mixup/CutMix & EMA**: Combining mixed-sample data augmentations (`Mixup` + `CutMix`) with Exponential Moving Average (`EMA 0.9999`) eliminated overfitting and produced a smooth, robust loss landscape.
3. **4GB GPU Efficiency**: By leveraging Automatic Mixed Precision (`AMP FP16`) and Batch Size 16 with Gradient Accumulation, `ConvNeXt-Small` (50M params) ran smoothly on a 3.68 GB VRAM GPU without CUDA Out-Of-Memory errors.
