# Runs-Path Audit Report — LAB2 & LAB3

> **Scope:** All training, evaluation, inference, and RunLogger entry points.  
> **Mode:** READ-ONLY. Nothing modified.  
> **Target state:** `experiments/lab2/runs` for LAB2, `experiments/lab3/runs` for LAB3.

---

## Legend

| Symbol | Meaning |
|:---|:---|
| ✅ | Already correct — points to the right namespaced path |
| ❌ | Points to wrong path — needs changing |
| ➡️ | The fix required (file, line, old value → new value) |

---

## LAB2 — Full Entry Point Audit

### A. Primary Training CLI

**File:** [`src/lab2/training/train_lab2_models.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/training/train_lab2_models.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 66: `RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"` | `experiments/runs` | ❌ |
| Line 68: `REGISTRY_PATH = RUNS_ROOT / "registry.json"` | inherits from `RUNS_ROOT` | ❌ |
| Line 197: `RUNS_ROOT = RUNS_ROOT / "_smoke"` | inherits from `RUNS_ROOT` | ❌ |
| Line 233: `logger = RunLogger(run_name, runs_root=RUNS_ROOT)` | caller of `RUNS_ROOT` | ❌ |
| Line 274: `find_latest_run_dir(run_name, runs_root=RUNS_ROOT)` | caller of `RUNS_ROOT` | ❌ |
| Line 292: docstring `experiments/runs/registry.json` | documentation only | ❌ |
| Lines 25–29: module docstring artifact paths | documentation only | ❌ |

**Effective runtime output:** `experiments/runs/` (and `experiments/runs/_smoke/` for `--smoke`)

> ➡️ **Fix:** Line 66 — change `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"`. All dependent references (L68, L197, L233, L274) automatically resolve correctly. Update docstring L25–28 and L257, L293.

---

### B. RunLogger Default

**File:** [`src/lab2/utils/run_logger.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/utils/run_logger.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 68: `def __init__(self, run_name, runs_root="experiments/runs", ...)` | `experiments/runs` | ❌ |
| Line 31: docstring example | `experiments/runs` | ❌ |
| Line 64: docstring description | `experiments/runs` | ❌ |

**Note:** The default is used by any caller that doesn't pass `runs_root` explicitly. Callers that pass explicit paths override this, so fixing the default is a safety-net and documentation fix. Only callers that rely on the default without passing `runs_root` are directly affected.

> ➡️ **Fix:** Line 68 — change default `"experiments/runs"` → `"experiments/lab2/runs"`. Update docstring lines 31 and 64.

---

### C. Checkpoint Utils

**File:** [`src/lab2/utils/checkpoint_utils.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/utils/checkpoint_utils.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 35: `find_best_checkpoint(run_name, runs_root="experiments/runs", ...)` | `experiments/runs` | ❌ |
| Line 94: `find_latest_run_dir(run_name, runs_root="experiments/runs")` | `experiments/runs` | ❌ |
| Lines 4–6: module docstring | `experiments/runs/...` | ❌ |

**Note:** `train_lab2_models.py` passes `RUNS_ROOT` explicitly to `find_latest_run_dir`, so the default is not currently exercised by the training CLI. However, it matters for any caller that imports these functions without an explicit path (e.g. notebooks, evaluation scripts).

> ➡️ **Fix:** Line 35 — change default `"experiments/runs"` → `"experiments/lab2/runs"`. Line 94 — same. Update module docstring lines 4–6.

---

### D. Experiment Script — DiffusionBlocks (exp_08)

**File:** [`src/lab2/experiments/exp_08_diffusionblocks.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/experiments/exp_08_diffusionblocks.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 76: `RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"` | `experiments/runs` | ❌ |
| Line 517: `run_dir = RUNS_ROOT / f"{ts}_{run_name}"` | inherits from `RUNS_ROOT` | ❌ |
| Line 38: docstring `experiments/runs/<ts>_DB-...` | documentation | ❌ |
| Line 609: inline string `"runs under experiments/runs"` | documentation | ❌ |

**Effective runtime output:** `experiments/runs/<ts>_DB-<track>-<mode>/`

> ➡️ **Fix:** Line 76 — change `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"`. Update docstring L38 and inline string L609.

---

### E. Experiment Script — MoE Router (moe_router_train.py)

**File:** [`src/lab2/experiments/moe_router_train.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/experiments/moe_router_train.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 154: `RunLogger("moe_router", runs_root=PROJECT_ROOT / "experiments" / "runs")` | `experiments/runs` | ❌ |

**Effective runtime output:** `experiments/runs/<ts>_moe_router/`

> ➡️ **Fix:** Line 154 — change `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"`.

---

### F. Experiment Script — Stacking MLP (stacking_mlp_train.py)

**File:** [`src/lab2/experiments/stacking_mlp_train.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/experiments/stacking_mlp_train.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 137: `RunLogger("stacking_mlp", runs_root=PROJECT_ROOT / "experiments" / "runs")` | `experiments/runs` | ❌ |

**Effective runtime output:** `experiments/runs/<ts>_stacking_mlp/`

> ➡️ **Fix:** Line 137 — change `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"`.

---

### G. Experiment Scripts — exp_01 through exp_07

**Files:** [`exp_01_optuna_hpo.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/experiments/exp_01_optuna_hpo.py) through [`exp_07_resnet_densenet_sota.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/experiments/exp_07_resnet_densenet_sota.py)

These scripts **do not** instantiate `RunLogger` or write to a `runs/` directory directly. They write results to `experiments/results/` (via `RESULTS_DIR`) or `experiments/plots/` — not to `runs/`. The only run logging for these experiments is done by `train_lab2_models.py` which calls `train_model()`.

`exp_01` writes an SQLite DB to `experiments/optuna_study.db` — this is a separate, correctly targeted path (not a `runs/` issue).

**Status:** ✅ No `runs/` path changes needed for exp_01–07.

---

### H. run_all_experiments.py

**File:** [`src/lab2/experiments/run_all_experiments.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/experiments/run_all_experiments.py)

References `experiments/plots` (line 26) for output, delegates to individual `exp_0x` entry points. Contains **no** `runs/` path references.

**Status:** ✅ No `runs/` path changes needed.

---

### I. LAB2 Evaluation (evaluate_model.py)

**File:** [`src/lab2/eval/evaluate_model.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab2/eval/evaluate_model.py)

This module contains `evaluate()`, `per_class_accuracy()`, `load_checkpoint()`, `format_comparison_table()`, and `plot_metrics_heatmap_table()`. None of these functions write to a `runs/` directory. Checkpoint paths are passed in by the caller; this file does not resolve run directories itself.

**Status:** ✅ No `runs/` path changes needed.

---

### LAB2 Summary Table

| File | Line | Current | Required | Impact |
|:---|:---:|:---|:---|:---|
| `training/train_lab2_models.py` | 66 | `experiments/runs` | `experiments/lab2/runs` | **All 6 model variants** |
| `utils/run_logger.py` | 68 | `experiments/runs` (default) | `experiments/lab2/runs` | Fallback default |
| `utils/checkpoint_utils.py` | 35, 94 | `experiments/runs` (defaults) | `experiments/lab2/runs` | Checkpoint lookup defaults |
| `experiments/exp_08_diffusionblocks.py` | 76 | `experiments/runs` | `experiments/lab2/runs` | DiffusionBlocks runs |
| `experiments/moe_router_train.py` | 154 | `experiments/runs` | `experiments/lab2/runs` | MoE router runs |
| `experiments/stacking_mlp_train.py` | 137 | `experiments/runs` | `experiments/lab2/runs` | Stacking MLP runs |

**Total: 6 files, 8 executable lines need changing** (plus associated docstrings).

---

---

## LAB3 — Full Entry Point Audit

### A. YAML Config Files — `run_root` key

All 8 LAB3 config YAML files already have `run_root: "experiments/lab3/runs"`:

| Config file | `run_root` value | Status |
|:---|:---|:---|
| `config_imdb_sentiment.yaml` | `experiments/lab3/runs` | ✅ |
| `config_imdb_sentiment_512.yaml` | `experiments/lab3/runs` | ✅ |
| `config_imdb_sentiment_512_hyper.yaml` | `experiments/lab3/runs` | ✅ |
| `config_imdb_sentiment_baseline.yaml` | `experiments/lab3/runs` | ✅ |
| `config_imdb_sentiment_denoised.yaml` | `experiments/lab3/runs` | ✅ |
| `config_imdb_sentiment_denoised_fullft.yaml` | `experiments/lab3/runs` | ✅ |
| `config_imdb_sentiment_lora.yaml` | `experiments/lab3/runs` | ✅ |
| `config_imdb_sentiment_tuned.yaml` | `experiments/lab3/runs` | ✅ |

---

### B. Primary Training CLI

**File:** [`src/lab3/training/imdb_sentiment_train.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/training/imdb_sentiment_train.py)

| Location | Current behavior | Status |
|:---|:---|:---|
| Line 248: `run_dir = Path(resume_from) if resume_from else next_run_dir(t["run_root"], run_name)` | `t["run_root"]` comes from config YAML → `experiments/lab3/runs` | ✅ |
| Line 433: `update_registry(t["run_root"], run_name, run_dir)` | same config value | ✅ |
| Line 517: `latest_run_dir(t["run_root"], t["run_name"])` | same config value | ✅ |
| Line 6: module docstring `experiments/runs/<ts>_<run>/` | wrong path in docstring | ❌ (doc only) |
| Line 10–12: usage examples in docstring | `src.training.imdb_sentiment_train` (wrong module path) | ❌ (doc only) |

**Effective runtime output:** `experiments/lab3/runs/<ts>_<run_name>/` ✅

> ➡️ **Fix (doc only):** Line 6 docstring — update `experiments/runs/` → `experiments/lab3/runs/`. Lines 10–12 — update module path `src.training.imdb_sentiment_train` → `src.lab3.training.imdb_sentiment_train`.

---

### C. Baseline Experiment (Exercise 1)

**File:** [`src/lab3/experiments/baseline_imdb_sentiment.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/experiments/baseline_imdb_sentiment.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 205: `tb_dir = Path("experiments/lab3/runs/baseline_zero_shot/tensorboard")` | `experiments/lab3/runs` | ✅ |
| Line 289: `results_dir = Path("experiments/lab3/results")` | `experiments/lab3/results` | ✅ |
| Line 510: `output_json=Path("experiments/lab3/results/baseline_imdb_sentiment.json")` | `experiments/lab3/results` | ✅ |

**Status:** ✅ Fully correct.

---

### D. Evaluation CLI

**File:** [`src/lab3/eval/evaluate_model.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/eval/evaluate_model.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 283: `parser.add_argument("--run-root", default="experiments/lab3/runs")` | `experiments/lab3/runs` | ✅ |
| Line 10: usage docstring `experiments/runs/<ts>_<run>/...` | wrong path in docstring | ❌ (doc only) |

**Effective runtime behavior:** resolves checkpoints from `experiments/lab3/runs` ✅

> ➡️ **Fix (doc only):** Line 10 — update docstring `experiments/runs/` → `experiments/lab3/runs/`.

---

### E. Error Auditor (Inference / Analysis)

**File:** [`src/lab3/eval/error_auditor.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/eval/error_auditor.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 24: `_load_latest_model(run_root="experiments/lab3/runs", ...)` | `experiments/lab3/runs` | ✅ |
| Line 75: `audit_top_misclassifications(run_root="experiments/lab3/runs", ...)` | `experiments/lab3/runs` | ✅ |

**Status:** ✅ Fully correct.

---

### F. Inference — SentimentPredictor

**File:** [`src/lab3/models/predictor.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/models/predictor.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 58: `from_latest_run(run_root="experiments/lab3/runs", ...)` | `experiments/lab3/runs` | ✅ |

**Status:** ✅ Fully correct.

---

### G. Trainer Utility (Plot Helper)

**File:** [`src/lab3/training/trainer.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/training/trainer.py)

| Location | Current value | Status |
|:---|:---|:---|
| Line 74: `plot_latest_training_curves(run_root="experiments/lab3/runs")` | `experiments/lab3/runs` | ✅ |

**Status:** ✅ Fully correct.

---

### H. RunLogger (LAB3)

**File:** [`src/lab3/utils/run_logger.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/utils/run_logger.py)

LAB3's `RunLogger.__init__` takes `run_dir` (a fully resolved path) rather than `runs_root` — the caller always supplies the complete run directory path via `next_run_dir()`. No default `runs_root` exists in this version. There is no path to fix.

**Status:** ✅ Fully correct by design.

---

### I. Checkpoint Utils (LAB3)

**File:** [`src/lab3/utils/checkpoint_utils.py`](file:///C:/document/Study%20documents/Deeplearning_Course/src/lab3/utils/checkpoint_utils.py)

`latest_run_dir()`, `next_run_dir()`, `update_registry()` — all take `run_root` as a caller-supplied argument with no embedded default. The docstring on line 34 says `experiments/runs/` (stale reference), but the function itself has no wrong default.

| Location | Current value | Status |
|:---|:---|:---|
| Line 34: docstring `experiments/runs/<ts>_<run_name>/` | wrong path in docstring | ❌ (doc only) |
| Line 112: docstring `experiments/runs/registry.json` | wrong path in docstring | ❌ (doc only) |

> ➡️ **Fix (doc only):** Lines 34, 112 — update docstring references from `experiments/runs/` → `experiments/lab3/runs/`.

---

### LAB3 Summary Table

| File | Line | Current | Required | Impact |
|:---|:---:|:---|:---|:---|
| `training/imdb_sentiment_train.py` | 6 | docstring `experiments/runs/` | `experiments/lab3/runs/` | Doc only |
| `training/imdb_sentiment_train.py` | 10–12 | wrong module path in usage | `src.lab3.training.imdb_sentiment_train` | Doc only |
| `eval/evaluate_model.py` | 10 | docstring `experiments/runs/` | `experiments/lab3/runs/` | Doc only |
| `utils/checkpoint_utils.py` | 34, 112 | docstring `experiments/runs/` | `experiments/lab3/runs/` | Doc only |

**Total LAB3 executable fixes: ZERO.** All runtime paths are correct. Only stale docstrings remain.

---

---

## Cross-Cutting: `.gitignore` Scope

The current ignore rule is:
```
/experiments/runs/*
!/experiments/runs/.gitkeep
```

This pattern only ignores the **top-level** `experiments/runs/` directory.  
It does **not** cover `experiments/lab2/runs/` or `experiments/lab3/runs/`.

Once the LAB2 scripts are redirected to `experiments/lab2/runs/`, those runtime outputs will **not be gitignored** — model checkpoints (`.pt`), logs, and TensorBoard events would be committed.

> ➡️ **Additional fix required in `.gitignore`** (not a source file, but essential before any LAB2 training run executes after the path fix):
> ```gitignore
> # Add these two lines:
> /experiments/lab2/runs/*
> !/experiments/lab2/runs/.gitkeep
> ```
> LAB3 already has `experiments/lab3/runs/.gitkeep` tracked and the `runs/` contents are not yet ignored at the new path either. The same pattern should be added for LAB3:
> ```gitignore
> /experiments/lab3/runs/*
> !/experiments/lab3/runs/.gitkeep
> ```

---

## Consolidated Fix List (Minimal, Ordered)

### Executable changes (will change runtime behavior)

| # | File | Line(s) | Change |
|:---|:---|:---|:---|
| 1 | `src/lab2/training/train_lab2_models.py` | 66 | `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"` |
| 2 | `src/lab2/utils/run_logger.py` | 68 | default `"experiments/runs"` → `"experiments/lab2/runs"` |
| 3 | `src/lab2/utils/checkpoint_utils.py` | 35 | default `"experiments/runs"` → `"experiments/lab2/runs"` |
| 4 | `src/lab2/utils/checkpoint_utils.py` | 94 | default `"experiments/runs"` → `"experiments/lab2/runs"` |
| 5 | `src/lab2/experiments/exp_08_diffusionblocks.py` | 76 | `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"` |
| 6 | `src/lab2/experiments/moe_router_train.py` | 154 | `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"` |
| 7 | `src/lab2/experiments/stacking_mlp_train.py` | 137 | `"experiments" / "runs"` → `"experiments" / "lab2" / "runs"` |
| 8 | `.gitignore` | new lines | Add `experiments/lab2/runs/*` and `experiments/lab3/runs/*` ignore rules |

### Documentation-only changes (no runtime effect)

| # | File | Line(s) | Change |
|:---|:---|:---|:---|
| 9 | `src/lab2/training/train_lab2_models.py` | 25–28, 257, 293 | Update docstring path references |
| 10 | `src/lab2/utils/run_logger.py` | 31, 64 | Update docstring example path |
| 11 | `src/lab2/utils/checkpoint_utils.py` | 4–6 | Update module docstring path |
| 12 | `src/lab3/training/imdb_sentiment_train.py` | 6, 10–12 | Update docstring path + module name |
| 13 | `src/lab3/eval/evaluate_model.py` | 10 | Update docstring example path |
| 14 | `src/lab3/utils/checkpoint_utils.py` | 34, 112 | Update docstring path references |

---

> [!IMPORTANT]
> **Historical `experiments/runs/` contents must not be moved or deleted.** The 21 existing run directories under `experiments/runs/` are gitignored (local only) and represent completed historical training sessions. They remain readable by any code that still points to the old path for reference purposes. The fix only redirects **new** runs.

> [!NOTE]
> **LAB3 is already correct at the executable level.** All LAB3 scripts resolve run paths from YAML config (`run_root: "experiments/lab3/runs"`), which was correctly set during the LAB3 consolidation. Only stale docstrings need updating.
