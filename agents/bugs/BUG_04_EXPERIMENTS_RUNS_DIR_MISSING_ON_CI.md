# BUG_04_EXPERIMENTS_RUNS_DIR_MISSING_ON_CI.md — Bug Report (agents/bugs)

---

## Header

- **Title:** Missing `experiments/runs` Directory Failure on Fresh Git Checkout / CI Runner
- **Bug ID:** `BUG-04`
- **Date identified:** 2026-08-12
- **Description:** `test_core_directories_exist` failed on GitHub Actions CI runner because `.gitignore` ignored `/experiments/runs/` entirely, leaving the directory untracked by Git.
- **Status:** Resolved
- **Severity:** High
- **Category:** Testing / CI/CD
- **Target component:** [`.gitignore`](../../.gitignore), [`tests/test_smoke.py`](../../tests/test_smoke.py)

---

## 1. Symptom & Error Traceback

During the CI test runner execution on GitHub Actions:

```text
FAILED tests/test_smoke.py::test_core_directories_exist - AssertionError: missing directory: experiments/runs
assert False
 +  where False = is_dir()
 +    where is_dir = (PosixPath('/home/runner/work/Deeplearning_Course_UTH/Deeplearning_Course_UTH') / 'experiments/runs').is_dir
```

---

## 2. Root Cause Analysis

In [`.gitignore`](../../.gitignore), the rule `/experiments/runs/` completely excluded the `experiments/runs/` directory from Git tracking. Because Git does not commit empty folders or ignored folders, a fresh clone on a CI runner lacked `experiments/runs/`, breaking `test_core_directories_exist`.

---

## 3. Solution & Remediation

1. Created `experiments/runs/.gitkeep`.
2. Updated [`.gitignore`](../../.gitignore) to ignore all sub-contents of `experiments/runs/` while keeping `.gitkeep` tracked:
   ```gitignore
   /experiments/runs/*
   !/experiments/runs/.gitkeep
   ```

---

## 4. Verification Evidence

- Executed pytest test suite:
  ```bash
  pytest tests/ -v
  ```
  Result: **3/3 smoke tests passed cleanly**.

---

## 5. Related Links

- Master bug index: [README.md](README.md)
- Git ignore rules: [`.gitignore`](../../.gitignore)
- Smoke test file: [`tests/test_smoke.py`](../../tests/test_smoke.py)
