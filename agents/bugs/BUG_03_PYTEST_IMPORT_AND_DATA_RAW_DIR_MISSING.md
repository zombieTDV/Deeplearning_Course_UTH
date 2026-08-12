# BUG_03_PYTEST_IMPORT_AND_DATA_RAW_DIR_MISSING.md — Bug Report (agents/bugs)

---

## Header

- **Title:** PyTest `ModuleNotFoundError: No module named 'src'` and Missing `data/raw` Directory on CI
- **Bug ID:** `BUG-03`
- **Date identified:** 2026-08-12
- **Description:** Running plain `pytest` on a fresh git checkout / CI runner failed because `src` was not automatically added to Python path, and empty git-ignored data directories (`data/raw`) were missing.
- **Status:** Resolved
- **Severity:** High
- **Category:** Testing / Environment / CI/CD
- **Target component:** [`pytest.ini`](../../pytest.ini), [`.gitignore`](../../.gitignore), [`tests/test_smoke.py`](../../tests/test_smoke.py)

---

## 1. Symptom & Error Traceback

During the CI test runner execution (`pytest`), two test failures occurred:

```text
FAILED tests/test_smoke.py::test_src_packages_importable - ModuleNotFoundError: No module named 'src'
FAILED tests/test_smoke.py::test_core_directories_exist - AssertionError: missing directory: data/raw
assert False
 +  where False = is_dir()
 +    where is_dir = (PosixPath('/home/runner/work/Deeplearning_Course_UTH/Deeplearning_Course_UTH') / 'data/raw').is_dir
```

---

## 2. Root Cause Analysis

1. **Missing `pythonpath` in `pytest.ini`:** When running plain `pytest` (instead of `python -m pytest`), the current working directory (`.`) was not included in `sys.path`, causing imports of top-level package `src` (`import src.data`) to throw `ModuleNotFoundError`.
2. **Git Ignoring Empty Directories:** Git does not track empty directories. Because `.gitignore` ignored `/data/raw/` entirely, `data/raw/` was absent on a fresh git checkout on CI runners, causing `test_core_directories_exist` to fail on `(PROJECT_ROOT / "data/raw").is_dir()`.

---

## 3. Solution & Remediation

1. **Configured `pythonpath` in `pytest.ini`:** Added `pythonpath = .` to [`pytest.ini`](../../pytest.ini) so `pytest` automatically adds the project root to `sys.path`.
2. **Created `.gitkeep` Files & Updated `.gitignore`:**
   - Created `.gitkeep` placeholder files in `data/raw/`, `data/processed/`, and `data/external/`.
   - Updated [`.gitignore`](../../.gitignore) to exclude data contents while keeping `.gitkeep` tracked:
     ```gitignore
     /data/raw/*
     !/data/raw/.gitkeep
     /data/processed/*
     !/data/processed/.gitkeep
     /data/external/*
     !/data/external/.gitkeep
     ```

---

## 4. Verification Evidence

- Executed direct `pytest` command:
  ```bash
  pytest tests/ -v
  ```
  Result: **3/3 smoke tests passed cleanly** (both `src` import and `data/raw` directory checks succeeded).

---

## 5. Related Links

- Master bug index: [README.md](README.md)
- Pytest configuration: [`pytest.ini`](../../pytest.ini)
- Git ignore rules: [`.gitignore`](../../.gitignore)
- Smoke test file: [`tests/test_smoke.py`](../../tests/test_smoke.py)
