# ERROR_ANALYSIS.md

**Phase / area:** Cat/Dog Confusion Reduction — error taxonomy, diagnostic
methodology, and known failure modes for the targeted cat↔dog strategy work.
**Last updated:** 2026-08-04

## Purpose

Document the systematic approach to error analysis and the known failure modes
so future work can (a) reproduce the diagnostics, (b) avoid the resolved bugs,
and (c) target the still-open limitation. This complements
`progress/ERROR_ANALYSIS.md` (the running error log).

---

## 1. Error Taxonomy for CIFAR-10 Cat/Dog

The classifier's residual errors concentrate in a few semantic clusters:

| Cluster | Example (true → pred) | Count (SOTA ensemble) |
|---------|------------------------|----------------------|
| Cat↔Dog | dog → cat (61), cat → dog (30) | **91 cross-confusions** |
| Vehicle | truck → automobile (30), ship → airplane (28) | ~58 |
| Animal shape | horse → deer (22), deer → cat (10) | ~32 |

Targeting the Cat↔Dog cluster is the highest-leverage isolated improvement.

## 2. Isolated Benchmark Methodology

- **Scope:** CIFAR-10 test samples whose true class is cat(3) or dog(5) (2000).
- **Metrics:** isolated accuracy, per-class cat/dog accuracy,
  cat↔dog **cross-confusion count** (true-cat→pred-dog + true-dog→pred-cat).
- **Rule:** no full-test-set accuracy is used as an acceptance criterion; the
  benchmark is intentionally isolated to measure the targeted change.

## 3. Baseline

| Model | Isolated acc | cat | dog | cross_conf |
|-------|-------------|-----|-----|-----------|
| Finetune ensemble (live) | 87.40% | 85.6% | 89.2% | 143 |
| SOTA ensemble (persisted reference) | 93.30% | 94.5% | 92.1% | 91 |

> The live baseline is the finetune ensemble because the SOTA checkpoints are
> not on disk (see `progress/ERROR_ANALYSIS.md` ERR-01).

## 4. Strategy Failure Modes (learned)

### 4.1 Focal-loss pitfalls (S2)
- `pt**gamma` up-weights **easy** examples — the opposite of focal loss.
- Computing `pt = exp(-ce)` from a **class-weighted** CE kills gradients on
  weighted/misclassified samples → collapse.
- A strong dog weight (e.g. 3.0) over-corrects after the fix and collapses
  the cat class. **Use a mild weight (1.0–1.5).**
- Reference correct form: `(1 - pt)**gamma * ce`, class weight applied last.

### 4.2 Hard-negative mining pitfalls (S1)
- Oversampling weight must not depend on a corrupted label map
  (see ERR-04: `DOG_IDX - CAT_IDX == 2`).
- Hard negatives must be defined **leakage-free** on the training set (from
  the base model's train-set errors), not on the validation/test set.

### 4.3 Arbitration/ensemble pitfalls (S3)
- A frozen-feature linear arbiter can under-perform the ensemble's own
  cat/dog belief; arbitration must be validated on the isolated test set and
  ablated against the "keep ensemble argmax" policy before being trusted.

## 5. Diagnostic Tooling

- `diag_focal.py` (temp): trains buggy vs fixed focal heads on synthetic
  cat/dog features and tracks per-epoch cat/dog accuracy to expose collapse
  dynamics and tune the class weight.

## 6. Open Limitation & Recommended Next Steps

**Open:** after fixing all code bugs, none of S1/S2/S3 improves isolated
accuracy over the baseline (all within ±1.3%, S2 −0.5%). The frozen-feature
arbiter approach does not out-perform the ensemble's own cat/dog judgment.

**Recommended next steps to actually reduce cross-confusion:**
1. Feature-level: fine-tune (not freeze) the backbone's top block on a
   class-balanced cat/dog set, so cat/dog features are pushed apart.
2. Specialist that takes raw images (S3 CNN) but arbitrate by **marginal
   cat/dog probability** with a tuned blend, rather than a fixed 0.5/0.5 mix.
3. Per-class decision threshold on the cat/dog logit margin (extend the
   logit-bias sweep to cat/dog only), tuned on validation.
4. Verify against the **SOTA ensemble** once its checkpoints are restored —
   the target is reducing the reference 91 cross-confusions, not the weaker
   finetune baseline.

---

## References
- Implementation: `src/experiments/catdog_confusion_reduction.py`
- Results: `experiments/results/catdog_confusion_reduction.json`
- Error log: `progress/ERROR_ANALYSIS.md`
- Error-analysis notebook: `notebooks/practice_2_error_analysis.ipynb`
