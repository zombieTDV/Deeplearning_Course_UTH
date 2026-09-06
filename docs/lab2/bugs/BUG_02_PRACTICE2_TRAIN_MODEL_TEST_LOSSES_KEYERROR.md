# BUG-02: `KeyError: 'test_losses'` in Practice 2 Notebook & Build Script

## 📋 Summary Information

| Metric | Details |
| :--- | :--- |
| **Bug ID** | `BUG-02` |
| **Title** | `KeyError: 'test_losses'` in Practice 2 Notebook & Build Script |
| **Component** | `scratch/build_notebook.py`, `notebooks/practice_2.ipynb` |
| **Severity** | High (Caused runtime crash during fresh training execution) |
| **Status** | Resolved ✅ |
| **Date** | 2026-08-04 |

---

## 🔍 Problem Description

When running `notebooks/practice_2.ipynb` with `FORCE_RETRAIN = True` (or running the training loop via `scratch/build_notebook.py`), the script crashed with a `KeyError: 'test_losses'` in Cell 21 during results and training history serialization.

### Stack Trace / Symptom
```text
KeyError: 'test_losses'
  File "scratch/build_notebook.py", line 1107, in <module>
    "test_losses": [round(v, 4) for v in res["test_losses"]],
```

---

## 🧬 Root Cause Analysis

In `src/training/train_model.py`, the core training wrapper function `train_model(...)` executes per-epoch training and validation loops, returning a dictionary with the following contract:

```python
return {
    "run_name": run_name,
    "num_epochs": num_epochs,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accs": train_accs,
    "val_accs": val_accs,
    "best_val_loss": best_val_loss,
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "best_state_path": best_state_path,
}
```

Notice that `test_losses` is **not** part of `train_model()`'s return dictionary. Test-set evaluation happens separately in Phase 5 (`src/eval/evaluate_model.py`) after all models finish training.

However, the notebook builder (`scratch/build_notebook.py`) made invalid assumptions about `train_model()`'s return keys across 4 locations:

1. **Cell 11 (Checkpoint Fallback)**: Fabricated `"test_losses": [test_loss + 0.003]` and `"train_losses": [test_loss + 0.02]`. When `FORCE_RETRAIN = True`, the fresh training dictionary returned by `train_model()` lacked `test_losses`.
2. **Cell 17 (Loss Curves Plot)**: Executed `test_l = res.get("test_losses", val_l)` and plotted a third line for Test Loss. On fresh runs, it silently fell back to `val_l`, incorrectly showing Test Loss identical to Validation Loss.
3. **Cell 21 (JSON Serialization)**: Executed `"test_losses": [round(v, 4) for v in res["test_losses"]]`, throwing `KeyError: 'test_losses'` when `res` originated from a fresh `train_model()` run.
4. **Cell 1 (Mermaid Flowchart)**: Used stale labels `"Triple-Loss Line Tracking"` and `"Triple-Loss Curves"`.

---

## 🛠️ Resolution & Changes

All 4 locations were updated in `scratch/build_notebook.py`, and `notebooks/practice_2.ipynb` was cleanly regenerated:

1. **Cell 11 Update**:
   Removed synthetic `test_losses` in the checkpoint fallback dictionary. Standardized `train_losses` and `val_losses` to `[val_loss]`.

2. **Cell 17 Update**:
   Removed the third `test_loss` line from `plot_individual_model_loss_curves()`. The chart now cleanly plots **Train Loss vs. Validation Loss** with a shaded **Generalization Gap** and a star marker for **Best Epoch**.

3. **Cell 21 Update**:
   Removed `test_losses` serialization. `training_history.json` now cleanly serializes: `train_losses`, `val_losses`, `val_accs`, `best_epoch`, `best_val_acc`.

4. **Cell 1 Mermaid Diagram Update**:
   Updated text from `Triple-Loss` to `Train/Val Loss`.

---

## ✅ Verification & Testing

- Executed `python scratch/build_notebook.py` — exited with code `0`.
- Verified `notebooks/practice_2.ipynb` regenerated successfully with 22 cells.
- Audit confirmed 100% contract alignment across all modules in `src/`.
