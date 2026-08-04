# EVAL_STATUS.md

**Phase:** [../phases/EVAL.md](../phases/EVAL.md)
**Last updated:** 2026-08-04

## Status: Done

## Log

- 2026-07-29: File created, not started
- 2026-07-30: Wrote `src/eval/evaluate_model.py` — `evaluate()`, `per_class_accuracy()`, `load_checkpoint()`, `format_comparison_table()` utilities
- 2026-07-30: Updated `notebooks/practice_2.ipynb` — replaced Sections 5, 6, and 7 stubs with full eval, compare-and-report, and save-results code.
- 2026-08-02: Evaluated EXP-01 to EXP-05 benchmarks across ResNet18, DenseNet121, ConvNeXt-Tiny, and EfficientNet-B0.
- 2026-08-04: Completed EXP-06 (`97.66%` all-time project record) and EXP-07 (`96.00%` peak classic ensemble).
- 2026-08-04: Implemented Class-Logit Bias Sweep Tuning & Threshold Optimization in `notebooks/practice_2_logit_bias_sweep.ipynb`. Reached **97.28% Val Acc** / **96.73% Test Acc** on Soft-Voting Ensemble.

## Blockers (if any)

- None

## Next step

- All pipeline evaluation phases complete. Benchmark summary updated in `agents/experiments/SUMMARY_RESULTS.md` and `agents/experiments/LOGIT_BIAS_SWEEP_STATUS.md`.

