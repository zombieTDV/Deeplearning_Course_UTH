# EXP_02_LR_SCHEDULER_LLRD.md — Learning Rate Schedulers & Layer-wise LR Decay Plan

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-02`
- **Focus Area:** LR Annealing Schedules & Layer-wise Learning Rate Decay (LLRD)
- **Target Backbone:** ResNet18 (Fine-tuning mode)
- **Objective:** Evaluate dynamic learning rate schedules and discriminative per-layer learning rates to achieve smooth convergence and avoid catastrophic forgetting of ImageNet features.

---

## 🧪 2. Experimental Configurations

### Variant A: Scheduler Comparison (Uniform Backbone LR)
1. **Baseline:** Constant LR (`1e-4`).
2. **`ReduceLROnPlateau`:** `patience=2, factor=0.5, min_lr=1e-6`.
3. **`CosineAnnealingLR`:** `T_max=10, eta_min=1e-6`.
4. **`OneCycleLR`:** `max_lr=1e-3, pct_start=0.2, anneal_strategy='cos'`.

### Variant B: Layer-wise Learning Rate Decay (LLRD)
Apply multiplicative decay factor $\eta = 0.5$ per stage from head down to backbone:
- **`FC Head`**: $LR = 1.0 \times 10^{-3}$
- **`Layer4`**: $LR = 1.0 \times 10^{-4}$
- **`Layer3`**: $LR = 1.0 \times 10^{-5}$
- **`Layer1-2 & Conv1`**: $LR = 1.0 \times 10^{-6}$

---

## 🎯 3. Success Criteria

- Compare validation loss curves in TensorBoard.
- Target `val_acc > 93.0%` with stable non-oscillating loss curves.
