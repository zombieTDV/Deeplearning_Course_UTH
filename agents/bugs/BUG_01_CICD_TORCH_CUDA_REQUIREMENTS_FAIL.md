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

Updated [`requirements.txt`](../../requirements.txt) to unpin hardcoded `+cu130` local CUDA wheel suffixes and specify clean framework dependencies:
- Changed `torch==2.13.0+cu130` $\rightarrow$ `torch`
- Changed `torchvision==0.28.0+cu130` $\rightarrow$ `torchvision`
- Unpinned `transformers`, `datasets`, `evaluate` to allow version resolution across both local GPU and CI CPU environments.

Local CUDA GPU machines can continue to use CUDA-enabled PyTorch wheels in `.venv`, while CI/CD runners automatically pull compatible CPU PyTorch wheels from PyPI.

---

## 4. Verification Evidence

- Modified [`requirements.txt`](../../requirements.txt).
- Verified local test suite execution:
  ```bash
  python -m pytest tests/ -v
  ```
  Result: **3/3 smoke tests passed cleanly**.

---

## 5. Related Links

- Master bug index: [README.md](README.md)
- Dependency file: [`requirements.txt`](../../requirements.txt)
- Workflow configuration: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
