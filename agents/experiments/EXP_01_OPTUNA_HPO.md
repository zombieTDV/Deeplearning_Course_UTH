# EXP_01_OPTUNA_HPO.md — Automated Hyperparameter Optimization (Optuna) Plan

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-01`
- **Focus Area:** Automated Hyperparameter Tuning (HPO)
- **Target Backbone:** ResNet18 & DenseNet121
- **Objective:** Find the optimal combination of Learning Rate, Weight Decay, Optimizer, and Batch Size using Optuna SQLite study.

---

## ⚙️ 2. Hyperparameter Search Space

| Hyperparameter | Search Range / Options | Type | Notes |
| :--- | :--- | :--- | :--- |
| **`learning_rate`** | `[1e-5, 1e-2]` | Log-uniform | Base learning rate for classifier head |
| **`weight_decay`** | `[1e-6, 1e-2]` | Log-uniform | L2 regularization to prevent overfitting |
| **`optimizer`** | `["AdamW", "SGD_Momentum"]` | Categorical | SGD momentum set to 0.9 |
| **`batch_size`** | `[32, 64, 128]` | Categorical | Evaluated on CUDA VRAM footprint |
| **`dropout`** | `[0.0, 0.5]` | Uniform | Classifier head dropout rate |

---

## 🧪 3. Execution Plan

1. **Trial Count:** 20-30 trials with `optuna.create_study(direction="maximize")`.
2. **Pruning Strategy:** `optuna.pruners.MedianPruner(n_warmup_steps=2)` to terminate low-performing trials early.
3. **Storage:** `experiments/optuna_study.db`.
4. **Metrics Logged:** `val_accuracy` (primary objective), `val_loss`, `train_loss`, `execution_time`.

---

## 🎯 4. Success Criteria

- Identify top 3 parameter sets yielding `val_acc > 92%` on 5-epoch runs.
- Generate parameter importance ranking via `optuna_db_report.py`.
