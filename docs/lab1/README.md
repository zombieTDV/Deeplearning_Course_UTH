# LAB1 — Fashion-MNIST Classification (MLP vs CNN)

- **Motivation/Background**: First laboratory of the Deep Learning course. Implements baseline classification on the Fashion-MNIST dataset to benchmark Multi-Layer Perceptrons (MLPs) against Convolutional Neural Networks (CNNs).
- **Purpose**: Document the implementation, deliverables, experiments, and survivor map for LAB1.
- **Overview Pipeline**: 
  1. Diagnostic baseline (MLP & CNN) on Fashion-MNIST.
  2. Architecture variations (depth & width scaling for MLP; residual connections & filter scaling for CNN).
  3. Regularization & anti-overfitting (data augmentation, label smoothing, focal loss).
  4. Hyperparameter tuning via Optuna (30 trials).
- **References**: `src/lab1/`, `notebooks/lab1/`, `experiments/lab1/`, `data/lab1/splits/`.

---

## 1. Deliverables & Artifacts

| Component | Location | Description |
| :--- | :--- | :--- |
| **Source Modules** | `src/lab1/` | Flat utility modules (`data_utils.py`, `model_utils.py`, `train_utils.py`, `eval_utils.py`, `vis_utils.py`). |
| **Main Notebook** | [`notebooks/lab1/practice_1.ipynb`](../../notebooks/lab1/practice_1.ipynb) | End-to-end comparison between MLP and CNN on Fashion-MNIST. |
| **Diagnostic Notebooks** | `notebooks/lab1/error_analysis/` | Detailed error analysis, Optuna studies, and loss variant experiments. |
| **Experiment Results** | `experiments/lab1/` | Checkpoints, Optuna databases, confusion matrices, metrics, and PR/ROC plots. |
| **Persistent Splits** | `data/lab1/splits/` | 3-way train, validation, and test split indices (`train_indices.json`, `val_indices.json`, `test_indices.json`). |

---

## 2. Survivor Map (What Survives from LAB1)

As established in the consolidation specification:
* Genuine implementation code and notebooks survive under `src/lab1/` and `notebooks/lab1/`.
* Persistent split indices and generated metrics survive under `data/lab1/` and `experiments/lab1/`.
* Original documentation files (`purposre.md`, `GIT-WORKING-GUIDE.md`) survive under `docs/lab1/`.
* Early informal rulebases (`Rulebase.md`, `step_by_step.md`, `optuna_db_guide.md`) are archived under `docs/archive/lab1/`.
* The pedagogical markdown documentation guide (`MD_creation_guide.md`) was promoted to `docs/shared/MD_CREATION_GUIDE.md`.
* **No artificial AI-agent documentation or phase reports were fabricated** to preserve historical integrity.
