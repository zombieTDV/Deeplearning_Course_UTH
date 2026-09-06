# BUG_05_ACCELERATE_MISSING_IN_REQUIREMENTS.md — Bug Report (agents/bugs)

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---

## Header

- **Title:** Missing `accelerate` Dependency in Clean Environment Training Pipeline
- **Bug ID:** `BUG-05`
- **Date identified:** 2026-08-12
- **Description:** HF `Trainer` / `TrainingArguments` fails in clean environment checkouts because `accelerate>=1.1.0` is missing from `requirements.txt` and `requirements.lock`.
- **Status:** Resolved
- **Severity:** High
- **Category:** Dependencies / Environment
- **Target component:** [`requirements.txt`](../../requirements.txt), [`requirements.lock`](../../requirements.lock), [`src/training/imdb_sentiment_train.py`](../../src/training/imdb_sentiment_train.py)

---

## 1. Symptom & Error Traceback

When attempting to launch finetuning via `imdb_sentiment_train.py` in a clean environment:

```text
ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=1.1.0`:
Please install accelerate: `pip install accelerate>=1.1.0`
```

---

## 2. Root Cause Analysis

`accelerate` was not explicitly listed in `requirements.txt` or `requirements.lock`. Previous runs succeeded locally because `accelerate` happened to be installed ad hoc in the developer environment, causing silent failure on fresh checkouts and CI runners.

---

## 3. Solution & Remediation

1. Added `accelerate>=1.1.0` to [`requirements.txt`](../../requirements.txt).
2. Pinned `accelerate==1.14.0` in [`requirements.lock`](../../requirements.lock) and normalized file encoding to UTF-8.
3. Installed `accelerate` in the virtual environment.

---

## 4. Verification Evidence

Executed smoke test training run:
```bash
.venv/bin/python -m src.training.imdb_sentiment_train --smoke
```
Result: **Execution succeeded cleanly in 4.0 epochs with peak VRAM 1694 MiB ≤3.5 GB target.**

---

## 5. Related Links

- Master bug index: [README.md](README.md)
- Dependency file: [`requirements.txt`](../../requirements.txt)
- Lockfile: [`requirements.lock`](../../requirements.lock)
