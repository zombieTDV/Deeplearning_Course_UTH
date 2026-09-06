# 🐛 Bug Reports & Troubleshooting Directory

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---


This directory contains documented bug reports, runtime error analysis, and
resolution guides encountered during the project lifecycle.

## 🏗️ Architecture Overview

Bugs are reported against specific pipeline layers and tracked to resolution:

```
Component (src/data, src/training, notebooks) → BUG-<n> report
    → Root cause → Fix (committed) → Status (Resolved)
```

Each report follows [BUG_TEMPLATE.md](BUG_TEMPLATE.md) and links the affected
modules. Bug IDs are referenced from the codebase audit report and progress
docs.

## 📋 Bug Index

| Bug ID | Title & Summary | Component / Module | Severity | Status |
| :---: | :--- | :--- | :---: | :---: |
| `BUG-01` | [PyTorch CUDA Version Pin Failure on GitHub Actions CI/CD](BUG_01_CICD_TORCH_CUDA_REQUIREMENTS_FAIL.md) | `requirements.txt` | High | Resolved ✅ |
| `BUG-02` | [Ruff Linter Import & Type Hinting Violations](BUG_02_RUFF_LINT_ERRORS_FIX.md) | `src/data/eda_imdb.py`, `src/experiments/...` | Medium | Resolved ✅ |
| `BUG-03` | [PyTest Import & Missing `data/raw` Directory on CI](BUG_03_PYTEST_IMPORT_AND_DATA_RAW_DIR_MISSING.md) | `pytest.ini`, `.gitignore` | High | Resolved ✅ |
| `BUG-04` | [Missing `experiments/runs` Directory on CI Runner](BUG_04_EXPERIMENTS_RUNS_DIR_MISSING_ON_CI.md) | `.gitignore` | High | Resolved ✅ |
| `BUG-05` | [Missing `accelerate` Dependency in Clean Environment Training Pipeline](BUG_05_ACCELERATE_MISSING_IN_REQUIREMENTS.md) | `requirements.txt`, `requirements.lock` | High | Resolved ✅ |


---

## 🛠️ General Troubleshooting Guidelines

1. **`BrokenPipeError` in PyTorch DataLoaders**:
   Set `num_workers = 0` when running inside Jupyter Notebooks on Linux to
   avoid multiprocessing `forkserver` IPC pipe closures.
2. **`ModuleNotFoundError` for Project Submodules**:
   Ensure `PROJECT_ROOT` is dynamically detected and added to `sys.path`:
   ```python
   _cwd = Path(os.getcwd()).resolve()
   PROJECT_ROOT = _cwd if (_cwd / "src").exists() else (_cwd.parent if (_cwd.parent / "src").exists() else _cwd)
   if str(PROJECT_ROOT) not in sys.path:
       sys.path.insert(0, str(PROJECT_ROOT))
   ```
