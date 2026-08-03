# EXP_05_MODEL_ARCH_SWEEP.md — Modern Vision Architecture (ConvNeXt & EfficientNet) Benchmark Report

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-05`
- **Focus Area:** Modern Backbone Architecture Benchmarking
- **Architectures Evaluated:** `ResNet18` vs `DenseNet121` vs `ConvNeXt-Tiny` vs `EfficientNet-B0`
- **Date Executed:** 2026-08-02
- **Status:** Completed (Executed on CUDA GPU)
- **Objective:** Compare accuracy, parameter efficiency, inference latency, and memory footprint across classical vs modern CNN architectures on CIFAR-10 fine-tuning.

---

## 📊 2. Empirical Comparison Matrix

| Architecture | Total Parameters | Trainable Parameters | Time / Epoch (s) | Epoch 1 Val Acc (%) | Epoch 2 Val Acc (%) | Best Val Loss | Best Val Acc (%) | Checkpoint File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`ResNet18`** | 11,181,642 | 8,398,858 | 81.74s | 90.12% | 92.36% | 0.2450 | **92.36%** | `exp05_ResNet18_best.pt` |
| **`DenseNet121`** | 6,964,106 | 2,170,378 | 199.17s | 89.16% | 90.70% | 0.2810 | **90.70%** | `exp05_DenseNet121_best.pt` |
| **`ConvNeXt_Tiny`** | 27,827,818 | 14,298,634 | 250.97s | 95.94% | 96.42% | **0.1085** | **🏆 96.42%** | `exp05_ConvNeXt_Tiny_best.pt` |
| **`EfficientNet_B0`** | 4,020,358 | 424,970 | 91.53s | 82.54% | 84.44% | 0.4602 | **84.44%** | `exp05_EfficientNet_B0_best.pt` |

---

## 📈 3. Architecture Performance & Loss Trajectory

- **Architecture Loss Curves:** [../../experiments/plots/exp_05_architecture_loss_curves.png](../../experiments/plots/exp_05_architecture_loss_curves.png)
- **Architecture Trajectories Plot:** [../../experiments/plots/exp_05_architecture_trajectories.png](../../experiments/plots/exp_05_architecture_trajectories.png)
- **Master Architecture Sweep Dashboard:** [../../experiments/plots/exp_05_architecture_sweep.png](../../experiments/plots/exp_05_architecture_sweep.png)

---

## 💡 4. Key Takeaways & Architectural Insights

1. **`ConvNeXt-Tiny` Outperformance**: `ConvNeXt-Tiny` achieved an extraordinary **96.42% Validation Accuracy** with a lowest Val Loss of **0.1085**, outperforming ResNet18 by **+4.06%**.
2. **Tradeoff Analysis**:
   - **Maximum Accuracy Winner**: `ConvNeXt-Tiny` (96.42% Acc, 250.97s/epoch).
   - **Balanced Winner**: `ResNet18` (92.36% Acc, 81.74s/epoch - 3x faster).
   - **Lightweight Winner**: `EfficientNet-B0` (only 0.42M trainable parameters).
