# PLAN: Exercise 2 Anti-Overfitting & Re-Finetuning Plan

**Goal**: Fix the overfitting phenomenon observed during Exercise 2 training, optimize regularization and step-wise evaluation, achieve higher held-out accuracy, and embed all visual charts directly inside Notebook 02 (`notebooks/02_ex2_finetune.ipynb`).

---

## 1. Problem Diagnosis (Overfitting Evidence)

In the initial 3-epoch run (`20260812_152704_distilbert-finetune`):
- **Epoch 1**: `train_loss ≈ 0.342`, `val_loss = 0.2584`, `val_acc = 89.32%`
- **Epoch 2**: `train_loss ≈ 0.215`, `val_loss = 0.3146`, `val_acc = 90.08%`
- **Epoch 3**: `train_loss ≈ 0.138`, `val_loss = 0.4161`, `val_acc = 91.12%`

**Divergence**: Validation loss worsened significantly (+61% from 0.2584 to 0.4161) while training loss continued dropping. Evaluation was performed only once per epoch (every 1,407 steps), missing the optimal checkpoint between steps 1,000 and 2,000.

---

## 2. Remediation Strategy

| Component | Initial Value | Optimized Value | Rationale |
|---|---|---|---|
| **Learning Rate (`lr`)** | `2.0e-5` | **`1.5e-5`** | Slower, more stable gradient updates to prevent sharp loss spikes |
| **Weight Decay** | `0.01` | **`0.05`** | 5x stronger $L_2$ regularization to penalize over-confident weights |
| **Evaluation Strategy** | `"epoch"` (1,407 steps) | **`"steps"` (`eval_steps: 300`)** | High-frequency validation logging every 300 steps to pinpoint minimum loss |
| **Warmup Ratio** | `0.1` | **`0.1`** | Smooth learning rate warmup |
| **Epochs** | `3` | **`4`** | Extended fine-tuning with fine-grained checkpointing |
| **Classifier Dropout** | `0.1` | **`0.2`** | Higher regularization on sequence classification head |

---

## 3. Implementation Steps

1. **Update Config** (`configs/config_imdb_sentiment.yaml`):
   - Set `lr: 1.5e-5`, `weight_decay: 0.05`, `epochs: 4`, `eval_strategy: "steps"`, `eval_steps: 300`.
2. **Execute Re-Finetuning** (`src/training/imdb_sentiment_train.py`):
   - Run multi-epoch fine-tuning and log step-by-step training loss and validation loss into `history.jsonl`.
3. **Run Sealed Evaluation** (`src/eval/evaluate_model.py`):
   - Evaluate new best checkpoint on 25,000 IMDB test set. Save updated `imdb_sentiment_eval.json`, `imdb_finetuned_confusion_matrix.png`, and `imdb_finetuned_roc_curve.png`.
4. **Update Loss History Script** (`scratch/plot_training_history.py`):
   - Generate dual-panel step-by-step Loss & Accuracy curves (`experiments/plots/imdb_finetune_history.png`).
5. **Rebuild Notebook 02** (`scratch/build_notebook_ex2.py`):
   - Embed all 4 charts:
     - Training vs Validation Loss & Accuracy history chart
     - Normalized Confusion Matrix heatmap
     - ROC Curve with AUC score
     - Misclassification error characteristics distribution
6. **Verification**:
   - Run `pytest tests/ -v` and `ruff check src tests scratch` (0 errors).

---

## 4. Expected Deliverables

- Optimized checkpoint saved at `experiments/runs/<timestamp>_distilbert-finetune/checkpoints/distilbert-finetune_best.pt`.
- Validation loss curve showing flat/controlled validation loss curve.
- Fully rendered `notebooks/02_ex2_finetune.ipynb` with embedded inline charts.
