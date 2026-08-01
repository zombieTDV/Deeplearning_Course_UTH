# EXP_04_STEM_NATIVE_LONG_TRAIN.md — Native 32x32 Conv Stem Extended Training Plan

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-04`
- **Focus Area:** Native $32 \times 32$ Input Stem Long-Epoch Training
- **Target Backbone:** `ResNet18_32_NativeStem` (Modified $3 \times 3$ Conv1, Identity MaxPool)
- **Objective:** Evaluate if extended epoch training (25-30 epochs) with `CosineAnnealingLR` allows native 32x32 input resolution to match 224x224 upsampling accuracy (>90%) while retaining **40% faster training throughput** and **15% lower VRAM footprint**.

---

## 🧪 2. Experimental Setup

- **Input Dimension:** $32 \times 32$ (No 224x224 upsampling).
- **Epochs:** 25 epochs.
- **Optimizer:** `AdamW(lr=3e-4, weight_decay=1e-4)`.
- **Scheduler:** `CosineAnnealingLR(T_max=25, eta_min=1e-6)`.
- **Augmentation:** `RandAugment(num_ops=2, magnitude=7)`.

---

## 🎯 3. Success Criteria

- Compare throughput (sec/epoch) vs accuracy tradeoff with `ResNet18_224_Finetune`.
- Target native 32x32 fine-tuned test accuracy **$\ge 90.0\%$**.
