# 🐛 Bug Reports & Troubleshooting Directory

This directory contains documented bug reports, runtime error analysis, and resolution guides encountered during the project lifecycle.

## 📋 Bug Index

| Bug ID | Title & Summary | Component / Module | Severity | Status |
| :---: | :--- | :--- | :---: | :---: |
| **`BUG-01`** | [BUG_01_DATALOADER_BROKEN_PIPE_PYTHON314.md](BUG_01_DATALOADER_BROKEN_PIPE_PYTHON314.md) — PyTorch DataLoader `BrokenPipeError` in Multiprocessing Workers under Python 3.14 & Linux Jupyter Environments | `src/data/dataloader.py`, `notebooks/practice_2.ipynb` | High | Resolved ✅ |
| **`BUG-02`** | [BUG_02_PRACTICE2_TRAIN_MODEL_TEST_LOSSES_KEYERROR.md](BUG_02_PRACTICE2_TRAIN_MODEL_TEST_LOSSES_KEYERROR.md) — `KeyError: 'test_losses'` in Practice 2 Notebook & Build Script | `scratch/build_notebook.py`, `notebooks/practice_2.ipynb` | High | Resolved ✅ |

---

## 🛠️ General Troubleshooting Guidelines

1. **`BrokenPipeError` in PyTorch DataLoaders**:
   Set `num_workers = 0` when running inside Jupyter Notebooks on Linux to avoid multiprocessing `forkserver` IPC pipe closures.
2. **`ModuleNotFoundError` for Project Submodules**:
   Ensure `PROJECT_ROOT` is dynamically detected and added to `sys.path`:
   ```python
   _cwd = Path(os.getcwd()).resolve()
   PROJECT_ROOT = _cwd if (_cwd / "src").exists() else (_cwd.parent if (_cwd.parent / "src").exists() else _cwd)
   if str(PROJECT_ROOT) not in sys.path:
       sys.path.insert(0, str(PROJECT_ROOT))
   ```
