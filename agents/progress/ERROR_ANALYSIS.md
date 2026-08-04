# ERROR_ANALYSIS.md

**Purpose:** Systematically track and document every error, issue, and
limitation encountered during the cat/dog confusion-reduction work
(`src/experiments/catdog_confusion_reduction.py`) and related analysis.
**Last updated:** 2026-08-04

Status legend: `RESOLVED` (fixed + verified) · `OPEN` (mitigated, further work
needed) · `INFO` (environment/design constraint, not a code bug).

---

## Error Log

| # | Severity | Status | Summary | Component |
|---|----------|--------|---------|-----------|
| ERR-01 | Medium | INFO | SOTA ensemble checkpoints missing from disk | checkpoints |
| ERR-02 | Medium | RESOLVED | CIFAR-10 data-location mismatch (`data/external` vs `data/raw`) | data loading |
| ERR-03 | High | RESOLVED | `hard_idx or []` → NumPy bool ambiguity `ValueError` | `train_arbiter` |
| ERR-04 | Critical | RESOLVED | ~46 GB memory blow-up in hard-negative oversampling | `train_arbiter` |
| ERR-05 | Critical | RESOLVED | S2 focal/class-weight collapse to all-cat (dog acc 0%) | `FocalLoss` |
| ERR-06 | Medium | OPEN | None of S1/S2/S3 improve isolated accuracy vs baseline | arbitration design |

---

## ERR-01 — SOTA ensemble checkpoints missing
- **Status:** INFO (environment)
- **Symptom:** The 96% Soft-Voting Ensemble referenced in
  `practice_2_verB.ipynb` / `practice_2_logit_bias_sweep.ipynb` uses
  checkpoints `ResNet18-sota_best.pt`, `DenseNet121-sota_best.pt`,
  `exp07_*_sota_peak_best.pt`. None of these exist in
  `experiments/checkpoints/` — only `ResNet18-frozen/finetune_best.pt` and
  `DenseNet121-frozen/finetune_best.pt`.
- **Impact:** The live baseline had to use the finetune ensemble
  (isolated cat/dog acc **87.40%**, cross_conf **143**) instead of the
  persisted SOTA ensemble reference (isolated **93.30%**, cross_conf **91**).
- **Mitigation:** Benchmark against the live finetune ensemble and report the
  persisted SOTA numbers only as an external reference.

## ERR-02 — CIFAR-10 data-location mismatch
- **Status:** RESOLVED
- **Symptom:** `src/data/dataloader.py` defaults to
  `data/external/CIFAR-10`, which does not exist; the actual dataset lives at
  `data/raw/cifar-10-batches-py`.
- **Root cause:** Two data roots in the repo (`data/raw` in `load_cifar10.py`,
  `data/external/CIFAR-10` in `dataloader.py`).
- **Fix:** `catdog_confusion_reduction.py` loads directly from
  `data/raw` using the persisted split, avoiding a redundant download.

## ERR-03 — `hard_idx or []` NumPy bool ambiguity
- **Status:** RESOLVED
- **Symptom:** `ValueError: The truth value of an array with more than one
  element is ambiguous`.
- **Root cause:** `hard_idx = np.where(...)[0]` is a NumPy array; `hard_idx or
  []` calls `bool(array)`, which is undefined for multi-element arrays.
- **Fix:** use `len(hard_idx) > 0` (works for array/list/tuple/empty/None):
  ```python
  if hard_idx is not None and len(hard_idx) > 0:
      hard_set = {int(i) for i in hard_idx}
  else:
      hard_set = set()
  ```

## ERR-04 — ~46 GB memory blow-up in hard-negative oversampling
- **Status:** RESOLVED
- **Symptom:** `RuntimeError: DefaultCPUAllocator: not enough memory`
  (~46.9 GB) inside `train_arbiter` at `X[sel]`.
- **Root cause:** label-remap bug. `CAT_IDX=3, DOG_IDX=5`, so
  `DOG_IDX - CAT_IDX == 2`, producing binary labels `{0, 2}` instead of
  `{0, 1}`. Then `n_dog = (y == 1).sum() == 0`, making `w_dog ≈ 4500` and
  exploding `sel` to ~20M entries (`X[sel]` ≈ 41+ GB). Not a batch-size or
  DataLoader issue.
- **Fix:** remap cat→0 / dog→1 with `y_tr_b = (y_tr == DOG_IDX).long()`
  (also in the specialist CNN), and add an oversampling guard:
  ```python
  assert len(sel) <= 10 * len(y), "oversampled selection too large"
  ```

## ERR-05 — S2 focal/class-weight collapse to all-cat
- **Status:** RESOLVED
- **Symptom:** S2 isolated acc **46.50%**, dog acc **0.0%**, cross_conf **961**
  — the arbiter predicted cat for every dog.
- **Root cause (two compounding bugs in `FocalLoss`):**
  1. Wrong exponent — used `pt**gamma * ce`; focal loss is `(1-pt)**gamma * ce`.
     `pt**gamma` **up-weights easy** examples (opposite of focal).
  2. `pt = exp(-ce)` computed from the **class-weighted** CE, so the 3× dog
     weight was folded in and killed the gradient on misclassified dogs.
- **Training-dynamics evidence** (`diag_focal.py`): the buggy loss plateaus at
  a cat-biased solution (dog acc ~34%, unable to learn dogs); the corrected
  loss with the old strong weight `[1, 3.0]` over-corrects and collapses cats.
- **Fix:** correct `FocalLoss` (apply `(1-pt)**gamma`, then class weight last)
  and reduce the dog weight to a mild `[1, 1.5]` (configurable via
  `--focal-dog-weight`).
- **Result after fix (40 epochs):** S2 isolated **86.90%**, dog **88.9%**,
  cross_conf **153** — collapse eliminated, healthy balanced classifier.

## ERR-06 — Strategies do not beat baseline (OPEN)
- **Status:** OPEN (limitation of the arbitration design)
- **Symptom:** after fixes, extended run (40 arbiter epochs):
  - Baseline ensemble: 87.40% (cross 143)
  - S1 hard-negative: 86.10% (cross 169)
  - S2 focal (fixed): 86.90% (cross 153)
  - S3 specialist CNN: 87.15% (cross 148)
- **Interpretation:** the frozen-feature linear arbiter + arbitration rule does
  not out-perform the ensemble's own cat/dog judgment on the isolated test set.
  Fixing the loss removed a catastrophic failure but did not create an
  improvement. Remaining errors are high-confidence hard negatives, so
  feature-level work is required (see `phases/ERROR_ANALYSIS.md`).

---

## References
- Implementation: `src/experiments/catdog_confusion_reduction.py`
- Results: `experiments/results/catdog_confusion_reduction.json`
- Error-analysis notebook: `notebooks/practice_2_error_analysis.ipynb`
- Related bug index: `agents/bugs/README.md`
