**Status:** [DONE] Completed & Archived

# BUGFIX, AUDIT & PHASE COMPLETION REMEDIATION PLAN

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---


**Goal**: Resolve blocking bugs B1 (missing `.gitkeep` on `experiments/runs`) and B2 (missing `accelerate` dependency), fix non-blocking pipeline/config issues, perform a full Step-10 codebase audit, and legitimately validate all phase completions.

---

## 1. Summary of Issues & Directives

### Blocking Bugs
- **B1 — Missing `experiments/runs/.gitkeep`**:
  - *Symptom*: CI fails `test_smoke.py::test_core_directories_exist` on clean checkouts (`AssertionError: missing directory: experiments/runs`).
  - *Fix*: Re-create `experiments/runs/.gitkeep` and ensure `.gitignore` maintains `!/experiments/runs/.gitkeep`.
- **B2 — `accelerate` Missing from Dependencies**:
  - *Symptom*: HF `Trainer`/`TrainingArguments` raises `ImportError: Using the Trainer with PyTorch requires accelerate>=1.1.0` in clean environments.
  - *Fix*: Declare and pin `accelerate>=1.1.0` in `requirements.txt` and `requirements.lock`. Verify with `--smoke` execution.

### Non-Blocking Issues
- **1. Results Indexing Gap**: `experiments/results/imdb_sentiment_eval.json` was committed without updating `experiments/results/README.md` under the 5W1H format.
- **2. Un-gated Phase Statuses**: Phase status docs were flipped to `Done` without running the required Step-10 codebase audit gate.
- **3. Dead Config Parameter**: `training.save_total_limit: 2` in `configs/config_imdb_sentiment.yaml` is unused because `TrainingArguments(save_strategy="no")` delegates saving to the custom callback.
- **4. Resume Gaps in Trainer**:
  - Checkpoint stores `rng` state but resume mode never restores it.
  - Scheduler was created for full epochs before `num_train_epochs` was updated to `remaining_epochs`.
- **5. Dataset Tokenization Cache Staleness**: `prepare_imdb` reuses disk cache without checking parameter metadata (`max_length`, `model_name`, etc.), risking silent contamination if config changes.
- **6. Scratch Folder Gitignore Inconsistency**: `scratch/` is ignored in `.gitignore` while `scratch/build_notebook.py` is tracked.

---

## 2. Step-by-Step Action Plan

### Step 1: Fix B1 (CI Directory Tracking)
1. Touch `experiments/runs/.gitkeep`.
2. Confirm `.gitignore` has `/experiments/runs/*` and `!/experiments/runs/.gitkeep`.
3. Test with `pytest tests/test_smoke.py`.

### Step 2: Fix B2 (Dependency Declaration & Smoke Test Verification)
1. Add `accelerate>=1.1.0` to `requirements.txt`.
2. Update `requirements.lock` with `accelerate==1.14.0` and ensure UTF-8 encoding.
3. Install/verify `accelerate` in `.venv`.
4. Run `.venv/bin/python -m src.training.imdb_sentiment_train --smoke` to verify end-to-end training execution.

### Step 3: Fix Results Indexing (5W1H)
1. Update `experiments/results/README.md` table to index `imdb_sentiment_eval.json` with 5W1H metrics:
   - *What*: Finetuned DistilBERT test evaluation (Acc 91.19%, ROC-AUC 0.9699, F1 0.9119, VRAM peak 264.62 MB).
   - *Why*: Generalization benchmark vs Exercise 1 baseline (89.07%).
   - *When*: 2026-08-12, via `src/eval/evaluate_model.py`.
   - *Where*: `experiments/results/imdb_sentiment_eval.json` & `experiments/plots/`.
   - *Who*: bush-le + AI agent.
   - *How*: Single held-out pass on `distilbert-finetune_best.pt`.

### Step 4: Fix Dead Config, Resume Gaps, and Cache Staleness
1. **Config**: Remove `save_total_limit: 2` from `configs/config_imdb_sentiment.yaml`.
2. **Trainer Resume**:
   - In `src/training/imdb_sentiment_train.py`, compute `remaining_epochs` and set `train_args.num_train_epochs = remaining_epochs` *before* `trainer.create_optimizer_and_scheduler(...)`.
   - Restore RNG seed when resuming if `ckpt.get("rng")` is present.
3. **Cache Validation**:
   - In `src/data/prepare_imdb.py`, check `meta.json` metadata (`max_length`, `model_name`, `dataset_id`, `val_size`, `seed`) before returning cached dataset. If parameters mismatch, invalidate cache and re-tokenize.

### Step 5: Fix `.gitignore` Scratch Scaffolding
1. Update `.gitignore` to explicitly un-ignore `scratch/build_notebook.py` and `scratch/.gitkeep` while ignoring temporary scratch files (`/scratch/*`).

### Step 6: Document Bug Reports & Perform Step-10 Codebase Audit
1. Create `agents/bugs/BUG_05_ACCELERATE_MISSING_IN_REQUIREMENTS.md` documenting B2 and update `agents/bugs/README.md`.
2. Execute Step-10 codebase audit per `agents/rules/CODEBASE_AUDIT.md`.
3. Append Step-10 audit report section to `agents/CODEBASE_AUDIT_REPORT.md` validating Phase 3, 5, 6, 7, 8 status gates.
4. Verify all tests pass with `pytest`.

---

## 3. Verification & Acceptance Criteria
- `pytest` passes 100% (3/3 smoke tests).
- `--smoke` training run completes with 0 errors.
- `git status` shows clean tracking configuration for `experiments/runs/` and `scratch/`.
- All doc indexes and audit reports reflect verified project state.
