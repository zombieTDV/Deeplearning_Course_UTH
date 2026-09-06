# Repository Consolidation Handoff Specification

> **Target Repository:** `Deeplearning_Course`  
> **Consolidation Target Branch:** `consolidate/unified-main`  
> **Current Interrupted Branch:** `backup/phase2-interrupted` (Commit: `67b1178`)  
> **Master Specification Reference:** [`consolidation_plan.md`](file:///C:/Users/TDV/.gemini/antigravity-cli/brain/5ffaf680-ab14-44ae-af82-3b32af9c286c/consolidation_plan.md)  
> **Date:** 2026-09-06  

---

## 1. Executive Summary & Checkpoint Status

The repository consolidation unites three sequential lab projects into a unified research repository under `consolidate/unified-main`:
- **LAB1:** Fashion-MNIST Baseline & Error Analysis (`LAB1-FashionMNIST-Classification/main`)
- **LAB2:** CIFAR-10 Transfer Learning & SOTA Ensembles (`LAB2`)
- **LAB3:** Hugging Face NLP Sentiment Analysis (`LAB3_HuggingFace`)

### Checkpoint Progress Matrix

| Stage | Phase | Status | Commit / Reference | Details |
| :--- | :--- | :---: | :--- | :--- |
| **Safety** | Phase 0 | **DONE** | Git tags | `backup/main-pre-consolidation`, `backup/lab1-pre-consolidation`, `backup/lab2-pre-consolidation`, `backup/lab3-pre-consolidation` |
| **Governance** | Phase 4, Step 2 | **DONE** | `c709c69` | `agents/` (immutable constitution: `rules/`, `templates/`) & `docs/shared/` (5 reference SOPs) established on `consolidate/unified-main`. |
| **LAB1** | Phase 4, Step 3 | **DONE** | `42329f7` | `src/lab1/`, `notebooks/lab1/`, `data/lab1/splits/`, `experiments/lab1/`, `docs/lab1/` integrated on `consolidate/unified-main`. |
| **LAB2** | Phase 4, Step 4 | **COMMITTED (Interrupted)** | `67b1178` on `backup/phase2-interrupted` | 216 files: `src/lab2/`, `notebooks/lab2/`, `tests/lab2/`, `docs/lab2/`, `configs/lab2_data.yaml`, `data/lab2/processed/`, `experiments/lab2/`. Working tree clean. |
| **LAB3** | Phase 4, Step 5 | **NOT STARTED** | Branch: `LAB3_HuggingFace` | Needs checkout and migration into `src/lab3/`, `notebooks/lab3/`, `tests/lab3/`, `docs/lab3/`, `configs/`, `data/lab3/`, `experiments/lab3/`. |
| **Packaging & CI** | Phase 4, Step 6 | **NOT STARTED** | Workspace root | `pyproject.toml`, `requirements/` multi-tier, `.github/workflows/ci.yml`, `agents/rules/FOLDER_STRUCTURE.md`, root `README.MD`. |
| **Verification** | Phase 5 | **NOT STARTED** | Test batteries | pytest suites (`lab1`, `lab2`, `lab3`), ruff lint, import sanity. |
| **Merge Gate** | Phase 6 | **NOT STARTED** | Target `main` | Final audit and `--no-ff` merge to `main`. |

---

## 2. Exact Git Branch & Commit Topology

```text
67b1178 (HEAD -> backup/phase2-interrupted) chore: checkpoint interrupted phase 2 consolidation
42329f7 (consolidate/unified-main) feat(lab1): integrate Fashion-MNIST research, code, and documentation
c709c69 feat(governance): establish unified constitutional agents and shared docs
c3d3616 (tag: backup/lab3-pre-consolidation, origin/LAB3_HuggingFace, LAB3_HuggingFace)
be2f0ac (tag: backup/lab2-pre-consolidation, origin/LAB2, LAB2)
```

Working tree is **100% clean**. No unstaged or untracked changes.

---

## 3. Immediate Task: Finalizing LAB2 Migration

All LAB2 source code in `src/lab2/` has already been converted to `from src.lab2...` imports. Verification revealed **two minor test import adjustments** remaining before merging into `consolidate/unified-main`:

1. **Fix conftest imports in tests/lab2**:
   - `tests/lab2/test_build_model.py`: Line 13 change `from tests.conftest import requires_pretrained` → `from tests.lab2.conftest import requires_pretrained`
   - `tests/lab2/test_loaders.py`: Line 5 change `from tests.conftest import requires_data` → `from tests.lab2.conftest import requires_data`
2. **Commit the fix**:
   ```bash
   git add tests/lab2/test_build_model.py tests/lab2/test_loaders.py
   git commit -m "fix(tests): update conftest import path for lab2 test suite"
   ```
3. **Merge into `consolidate/unified-main`**:
   ```bash
   git checkout consolidate/unified-main
   git merge --ff-only backup/phase2-interrupted
   # Amend or re-title commit message to canonical:
   # feat(lab2): integrate CIFAR-10 transfer learning package and experiments
   ```
4. **Delete or keep backup**:
   ```bash
   git branch -d backup/phase2-interrupted
   ```

---

## 4. Next Step: LAB3 Integration (Phase 4, Step 5)

Source branch: `LAB3_HuggingFace` (tagged `backup/lab3-pre-consolidation`).

Execute while on `consolidate/unified-main`:

### 4.1 Checkout LAB3 Assets
```bash
# Checkout source, notebooks, tests, experiments, configs, data, and docs from LAB3
git checkout LAB3_HuggingFace -- src/
# Move to src/lab3
mkdir src/lab3
# Move contents of checked out src/ into src/lab3/
# Note: Preserve src/lab1 and src/lab2!

git checkout LAB3_HuggingFace -- notebooks/
# Move checked out notebooks into notebooks/lab3/ (01_ex1_sentiment_baseline.ipynb, 02_ex2_finetune.ipynb)

git checkout LAB3_HuggingFace -- tests/
# Move checked out tests into tests/lab3/

git checkout LAB3_HuggingFace -- configs/
# The configs in LAB3 are named lab3_config_imdb_sentiment*.yaml or similar. Keep them in configs/

git checkout LAB3_HuggingFace -- data/processed/
# Move processed files to data/lab3/processed/

git checkout LAB3_HuggingFace -- experiments/
# Move plots and results into experiments/lab3/ (preserve experiments/lab1 and experiments/lab2)

git checkout LAB3_HuggingFace -- agents/
# Move stage-specific docs to docs/lab3/:
# agents/OVERVIEW.md -> docs/lab3/OVERVIEW.md
# agents/PURPOSE.md -> docs/lab3/PURPOSE.md
# agents/PROJECT_ROADMAP.md -> docs/lab3/PROJECT_ROADMAP.md
# agents/CODEBASE_AUDIT_REPORT.md -> docs/lab3/CODEBASE_AUDIT_REPORT.md
# agents/experiments/*.md -> docs/lab3/experiments/
# agents/phases/*.md -> docs/lab3/phases/
# agents/plans/*.md -> docs/lab3/plans/
# agents/progress/*.md -> docs/lab3/progress/
# agents/bugs/*.md -> docs/lab3/bugs/
```

### 4.2 Namespacing Imports in LAB3
Convert all internal `src.` imports to `src.lab3.` across:
- `src/lab3/**/*.py`
- `tests/lab3/**/*.py`
- `notebooks/lab3/**/*.ipynb`

Example conversions:
- `from src.data.prepare_imdb` → `from src.lab3.data.prepare_imdb`
- `from src.models.model_builder` → `from src.lab3.models.model_builder`
- `from src.eval.evaluator` → `from src.lab3.eval.evaluator`
- `from src.utils.run_logger` → `from src.lab3.utils.run_logger`

### 4.3 Commit LAB3
```bash
git add src/lab3 tests/lab3 notebooks/lab3 configs/ data/lab3 experiments/lab3 docs/lab3
git commit -m "feat(lab3): integrate Hugging Face NLP sentiment analysis package"
```

---

## 5. Next Step: Environment, Tooling & Packaging (Phase 4, Step 6)

### 5.1 Multi-Tier Requirements (`requirements/`)
Create the following files:
- `requirements/base.txt`: Core scientific libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pyyaml`, `optuna`, `torch`, `torchvision`).
- `requirements/lab1.txt`: `-r base.txt` + `tabulate`, `jupyter`, `nbformat`.
- `requirements/lab2.txt`: `-r base.txt` + `tensorboard`, `pillow`.
- `requirements/lab3.txt`: `-r base.txt` + `seaborn`, `tensorboard`, `pillow`, `transformers`, `datasets`, `evaluate`, `accelerate`, `cleanlab`, `peft`, `ipython`.
- `requirements/dev.txt`: `-r lab3.txt` + `pytest`, `ruff`, `mypy`.
- `requirements.txt`: Root proxy referencing `-r requirements/dev.txt` with user documentation.

### 5.2 Build System (`pyproject.toml`)
Configure standard editable installation discovery:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "deeplearning-course-uth"
version = "1.0.0"
description = "Consolidated Deep Learning Coursework (Fashion-MNIST, CIFAR-10 Transfer Learning, Hugging Face NLP)"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
pythonpath = ["."]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = [
    "src/lab2/experiments/exp_*.py",
    "src/lab2/experiments/plot_*.py",
    "src/lab2/experiments/cifar_stem_experiment.ipynb",
    "src/lab3/experiments/baseline_imdb_sentiment.py",
]
```

### 5.3 CI Workflow (`.github/workflows/ci.yml`)
Configure a GitHub Actions workflow that runs:
- Linting (`ruff check src tests`)
- Matrix testing for `tests/lab1`, `tests/lab2`, `tests/lab3`

### 5.4 Unified Documentation
- Rewrite canonical `agents/rules/FOLDER_STRUCTURE.md` representing the unified layout (`agents/`, `docs/`, `src/lab*`, `notebooks/lab*`, `experiments/lab*`, `data/lab*`, `configs/`, `tests/lab*`).
- Rewrite root `README.MD` as the portfolio entry point.

### 5.5 Commit Build & Tooling
```bash
git add requirements/ requirements.txt pyproject.toml .github/ agents/rules/FOLDER_STRUCTURE.md README.MD
git commit -m "chore(build): configure unified packaging, requirements, and CI matrix"
```

---

## 6. Verification & Final Merge Protocol (Phases 5 & 6)

### Smoke Test Battery
```bash
# 1. Editable Install
pip install -e .

# 2. Package Import Sanity
python -c "import src.lab1.data_utils; print('LAB1 OK')"
python -c "import src.lab2.data.dataloader; print('LAB2 OK')"
python -c "import src.lab3.data.prepare_imdb; print('LAB3 OK')"

# 3. Test Suites
pytest tests/lab1 -v
pytest tests/lab2 -v
pytest tests/lab3 -v

# 4. Lint Check
ruff check src tests
```

### Final Merge Gate
```bash
git checkout main
git merge --no-ff -m "feat: consolidate LAB1, LAB2, and LAB3 into unified architecture" consolidate/unified-main
git tag v1.0.0-consolidated
```
