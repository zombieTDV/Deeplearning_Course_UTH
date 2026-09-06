# EXP_01_OPTUNA_HPO.md — Automated Hyperparameter Optimization (Optuna) Report

## 📌 1. Target & Objective

- **Experiment ID:** `EXP-01`
- **Focus Area:** Automated Hyperparameter Tuning (HPO)
- **Target Backbone:** ResNet18
- **Date Executed:** 2026-08-02
- **Status:** Completed (Executed on CUDA GPU)
- **Objective:** Find the optimal combination of Learning Rate, Weight Decay, Optimizer, and Batch Size using Optuna SQLite study.

---

## ⚙️ 2. Hyperparameter Search Space

| Hyperparameter | Search Range / Options | Type | Selected Best Value |
| :--- | :--- | :--- | :--- |
| **`learning_rate`** | `[1e-5, 1e-2]` | Log-uniform | **`8.96e-5`** |
| **`weight_decay`** | `[1e-6, 1e-2]` | Log-uniform | **`3.61e-6`** |
| **`optimizer`** | `["AdamW", "SGD"]` | Categorical | **`AdamW`** |
| **`batch_size`** | `[32, 64, 128]` | Categorical | **`64`** |

---

## 📊 3. Empirical Trial Execution Results

- **SQLite Database Path:** `experiments/optuna_study.db`
- **Study Name:** `cifar10_resnet18_hpo`

| Trial # | Optimizer | Learning Rate | Weight Decay | Batch Size | Epoch 1 Val Acc (%) | Epoch 2 Val Acc (%) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trial 2** | `AdamW` | `2.24e-5` | `4.38e-6` | 128 | 86.42% | 89.38% | Completed |
| **Trial 3** | `SGD` | `6.25e-5` | `6.03e-4` | 32 | 74.30% | 79.74% | Completed |
| **Trial 4** | **`AdamW`** | **`8.96e-5`** | **`3.61e-6`** | **64** | **90.12%** | **92.06%** | **🏆 Best Trial** |

---

## 📈 Benchmark Visualizations (EXP-01)

- **Trial Progression Plot:** [../../experiments/plots/exp_01_trial_progression.png](../../experiments/plots/exp_01_trial_progression.png)
- **Parameter Sensitivity & Importance:** [../../experiments/plots/exp_01_parameter_importance.png](../../experiments/plots/exp_01_parameter_importance.png)
- **Master HPO Dashboard:** [../../experiments/plots/exp_01_optuna_hpo.png](../../experiments/plots/exp_01_optuna_hpo.png)

---

## 💡 4. Key Takeaways & Conclusion

1. **Optimal Learning Rate & Optimizer**: `AdamW` with base learning rate `8.96e-5` significantly outperformed standard SGD, reaching **92.06% validation accuracy** in just 2 epochs.
2. **Weight Decay Impact**: Low weight decay (`3.61e-6`) provided sufficient L2 regularization without slowing down backbone fine-tuning.
3. **Batch Size Efficiency**: Batch size 64 balanced GPU memory usage and gradient stability best.
