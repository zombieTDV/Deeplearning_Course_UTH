# BUG_02_RUFF_LINT_ERRORS_FIX.md — Bug Report (agents/bugs)

---

## Header

- **Title:** Ruff Linter Import & Type Hinting Violations in Data and Experiment Modules
- **Bug ID:** `BUG-02`
- **Date identified:** 2026-08-12
- **Description:** `ruff check src tests` failed with 16 linting errors (unused imports, unsorted import blocks, deprecated `typing` generics, and unhandled `zip` strictness).
- **Status:** Resolved
- **Severity:** Medium
- **Category:** Code Quality / Linting
- **Target component:** [`src/data/eda_imdb.py`](../../src/data/eda_imdb.py), [`src/experiments/baseline_imdb_sentiment.py`](../../src/experiments/baseline_imdb_sentiment.py)

---

## 1. Symptom & Error Traceback

During the CI/CD linter step, `ruff check src tests` failed with exit code 1:

```text
Found 16 errors.
- I001: Import block is un-sorted or un-formatted in src/data/eda_imdb.py and baseline_imdb_sentiment.py
- UP035/UP006: typing.Dict / typing.List / typing.Tuple are deprecated, use built-in dict / list / tuple
- F401: Unused imports `pandas`, `os`, `typing.List`, `typing.Tuple`
- F841: Local variable `test_word_counts` is assigned to but never used
- B905: `zip()` without an explicit `strict=` parameter
- W293: Blank line contains whitespace
```

---

## 2. Root Cause Analysis

1. **Deprecated Type Annotations:** Modules used legacy `typing.Dict` and `typing.List` type hints instead of Python 3.11+ built-in `dict` and `list` syntax (`UP035`/`UP006`).
2. **Unused Imports & Variables:** Dead scaffolding imports (`pandas`, `os`) and unused intermediate variables (`test_word_counts`) remained in the codebase (`F401`/`F841`).
3. **Import Ordering & Formatting:** Imports inside functions (e.g. `set_seed()`) and top-level blocks were unsorted (`I001`).

---

## 3. Solution & Remediation

Refactored both target files to comply with `ruff` rules:
- **`src/data/eda_imdb.py`:** Removed unused `pandas` import and `test_word_counts` variable; updated return type annotation to `dict[str, Any]`; sorted import block.
- **`src/experiments/baseline_imdb_sentiment.py`:** Removed unused `os`, `typing.List`, `typing.Tuple` imports; moved `random` and `np` imports to top-level; added `strict=False` to `zip()`; removed trailing whitespace on line 183.

---

## 4. Verification Evidence

Executed `ruff check` in the virtual environment:
```bash
.venv/bin/ruff check src tests
```
Output:
```text
All checks passed!
```

Executed test suite:
```bash
python -m pytest tests/ -v
```
Result: **3/3 smoke tests passed cleanly**.

---

## 5. Related Links

- Master bug index: [README.md](README.md)
- Target modules: [`src/data/eda_imdb.py`](../../src/data/eda_imdb.py), [`src/experiments/baseline_imdb_sentiment.py`](../../src/experiments/baseline_imdb_sentiment.py)
- Rule definition: [`AGENT_AI.md`](../rules/AGENT_AI.md)
