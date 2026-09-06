# TRAINING_STATUS.md

**Phase:** [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md)
**Last updated:** 2026-08-04

## Status: Done

## Log

- 2026-07-29: File created, not started
- 2026-07-30: Wrote `src/training/train_model.py` — `train_one_epoch()`, `validate()`, `train_model()` full-loop wrapper.
- 2026-08-02: Added `src/experiments/` suite with 5 executable experiment modules (`EXP-01` to `EXP-05`).
- 2026-08-02: Executed full experiment suite on CUDA GPU:
  - `EXP-01` (Optuna HPO): Best trial #4 reached **92.06%** validation accuracy.
  - `EXP-02` (LR Schedulers & LLRD): `CosineAnnealingLR` reached **92.78%** accuracy.
  - `EXP-03` (Advanced Augmentations): `RandAugment` + `Label Smoothing` reached **92.72%** accuracy.
  - `EXP-04` (Native 32x32 Stem): Achieved **45.5s/epoch** throughput (2x faster than 224x224).
  - `EXP-05` (Model Architecture Sweep): **`ConvNeXt-Tiny` achieved 96.42% validation accuracy**.
- 2026-08-03: Completed EXP-06 (`ConvNeXt-Tiny SOTA` combination reaching **97.66%** all-time record).
- 2026-08-03: Completed EXP-07 (`ResNet18 + DenseNet121 Peak SOTA` reaching **96.00%** classic ensemble record).
- 2026-08-04: Performed Class-Logit Bias Sweeping & Decision Threshold Optimization (`notebooks/practice_2_logit_bias_sweep.ipynb`), reaching **97.28% Val Acc**.

## Blockers (if any)

- None

## Next step

- All training and post-processing optimization phases complete and verified.

