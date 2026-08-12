# BUG_01_CICD_TORCH_CUDA_REQUIREMENTS_FAIL.md — Bug Report (agents/bugs)

---

## Header

- **Title:** PyTorch CUDA Version Pin Failure on GitHub Actions CI/CD
- **Bug ID:** `BUG-01`
- **Date identified:** 2026-08-12
- **Description:** `pip install -r requirements.txt` failed on GitHub Actions CI runner because `torch==2.13.0+cu130` could not be resolved from standard PyPI index.
- **Status:** Resolved
- **Severity:** High
- **Category:** Environment / CI/CD
- **Target component:** [`requirements.txt`](../../requirements.txt), [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

---

## 1. Symptom & Error Traceback

During the execution of the GitHub Actions CI workflow (on push / pull request), the dependency installation step (`pip install -r requirements.txt`) failed with exit code 1:

```text
ERROR: Could not find a version that satisfies the requirement torch==2.13.0+cu130 (from versions: 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0)
ERROR: No matching distribution found for torch==2.13.0+cu130
Error: Process completed with exit code 1.
on test ci/cd
```

---

## 2. Root Cause Analysis

1. **CUDA Index Omission:** `requirements.txt` hardcoded `torch==2.13.0+cu130` and `torchvision==0.28.0+cu130`, which are hosted on PyTorch's custom CUDA wheel repository (`https://download.pytorch.org/whl/cu130`). Standard `pip install -r requirements.txt` on GitHub Actions only searches standard PyPI (`pypi.org`), where CUDA-suffixed wheels (`+cu130`) do not exist.
2. **Python 3.11 CI Runner Compatibility:** The CI workflow [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml#L15) runs Python 3.11, whereas certain strict pinned versions in `requirements.txt` required different Python versions.

---

## 3. Solution & Remediation

Kept the CUDA pins in [`requirements.txt`](../../requirements.txt) and fixed the **installation path** instead:

- `torch==2.13.0+cu130` / `torchvision==0.28.0+cu130` and the HF pins
  (`transformers==5.15.0`, `datasets==5.0.1`, `evaluate==0.4.6`) remain pinned
  for reproducibility.
- CI now installs via the PyTorch CUDA index:
  ```bash
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
  ```
  (`--extra-index-url` keeps everything else on PyPI; the `+cu130` local
  version only resolves from the PyTorch index.)
- Bumped the CI runner to Python 3.12 to match the local `.venv` environment.
- `requirements.lock` remains the exact-reproducibility source of truth.

> **Why not unpin?** Removing the pins makes `pip install torch` resolve to
> the **CPU wheel** from PyPI on GPU machines — a silent, hard-to-detect loss
> of CUDA. The pin failure was an install-path problem, not a version problem.

---

## 4. Verification Evidence

- [`requirements.txt`](../../requirements.txt) still pins
  `torch==2.13.0+cu130`; [`requirements.lock`](../../requirements.lock) is
  consistent with it.
- Local `.venv` verified: `torch 2.13.0+cu130`, `cuda: True`,
  `cuda_version: 13.0`.
- Test suite: `python -m pytest` → **3 passed**.
- CI workflow updated: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml#L18).

---

## 5. Related Links

- Master bug index: [README.md](README.md)
- Dependency file: [`requirements.txt`](../../requirements.txt)
- Workflow configuration: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
