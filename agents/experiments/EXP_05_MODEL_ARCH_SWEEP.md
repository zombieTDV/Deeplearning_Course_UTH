# EXP_05_MODEL_ARCH_SWEEP.md — Modern Vision Architecture (ConvNeXt & EfficientNet) Benchmark Plan

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-05`
- **Focus Area:** Modern Backbone Architecture Benchmarking
- **Architectures:** `ResNet18` vs `DenseNet121` vs `ConvNeXt-Tiny` vs `EfficientNet-B0`
- **Objective:** Compare accuracy, parameter efficiency, inference latency, and memory footprint across classical vs modern CNN architectures on CIFAR-10 fine-tuning.

---

## 🧪 2. Comparison Matrix Structure

| Architecture | Total Params | Trainable Params | Time / Epoch (s) | Peak VRAM (MB) | Test Acc (%) | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ResNet18`** | 11.2M | 8.4M | TBD | TBD | TBD | TBD |
| **`DenseNet121`** | 7.0M | 7.0M | TBD | TBD | TBD | TBD |
| **`ConvNeXt_Tiny`** | 28.6M | 28.6M | TBD | TBD | TBD | TBD |
| **`EfficientNet_B0`** | 5.3M | 5.3M | TBD | TBD | TBD | TBD |

---

## 🎯 3. Success Criteria

- Select the best overall architecture considering Accuracy/Parameter trade-off.
