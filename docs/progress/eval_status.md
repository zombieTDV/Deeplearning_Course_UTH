# eval_status.md

**Phase:** [../eval.md](../eval.md)
**Last updated:** 2026-07-30

## Status: Done

## Log

- 2026-07-29: File created, not started
- 2026-07-30: Wrote `src/eval/evaluate_model.py` — `evaluate()`, `per_class_accuracy()`,
  `load_checkpoint()`, `format_comparison_table()` utilities
- 2026-07-30: Updated `notebooks/practice_2.ipynb` — replaced Sections 5, 6, and 7 stubs
  with full eval, compare-and-report, and save-results code.  Loads trained checkpoints,
  evaluates on test set, computes per-class accuracy and confusion matrices, generates
  comparison table and normalised heatmaps, saves metrics JSON.

## Blockers (if any)

- None

## Next step

- All pipeline phases complete.  Next: run `notebooks/practice_2.ipynb` top to bottom
  to produce final outputs and the comparison report.  Launch TensorBoard via
  ``tensorboard --logdir experiments/tb_logs`` to review training curves.
