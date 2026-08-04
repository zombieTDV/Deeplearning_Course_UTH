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

## 6. Outcome on the Real SOTA Model (post-hoc approaches exhausted)

Once the SOTA checkpoints were restored, the three strategies were re-verified
against the **real** baseline (soft-voting ensemble, 96.96% full / 93.45%
isolated cat/dog / 86 cross-confusions):

| Approach | Isolated acc | cross_conf | Δ vs baseline |
|----------|-------------|-----------|---------------|
| Baseline ensemble (sota ×2) | 93.45% | 86 | — |
| S1 Hard-negative mining | 92.40% | 107 | −1.05% (worse) |
| S2 Focal loss / class-weight | 92.35% | 108 | −1.10% (worse) |
| S3 Specialist CNN member | 93.50% | 85 | +0.05% (noise) |

**Key finding:** the SOTA model fits the training set almost perfectly
(**only 1 cat/dog hard negative**), so hard-negative mining has nothing to
mine; and arbitration overrides strong SOTA features, degrading S1/S2. S3 is
within noise. **Post-hoc arbitration has reached its ceiling.**

## Remaining levers (feature-level only)

1. **Feature-level fine-tuning**: unfreeze the top block (`layer4` /
   `denseblock4`) with LLRD on a class-balanced cat/dog set — pushes cat/dog
   features apart rather than reading them post-hoc. Expected small (+0.2–0.5%).
2. **TTA (multi-crop voting)** — cheap, typically +0.2–0.5%.
3. **Larger native resolution / stem upgrade** — attacks the 32×32→224
   upsampling ceiling noted in `data_prep.md`.

Expected realistic ceiling: ~97–98% full test; 99% is unlikely at this
resolution. If the goal is a practice deliverable, the current 96.96% already
fully demonstrates transfer learning + ensembling, and further effort has low ROI.

---

## References
- Implementation: `src/experiments/catdog_confusion_reduction.py`
- Results: `experiments/results/catdog_confusion_reduction.json`
- Error log: `progress/ERROR_ANALYSIS.md`
- Error-analysis notebook: `notebooks/practice_2_error_analysis.ipynb`
