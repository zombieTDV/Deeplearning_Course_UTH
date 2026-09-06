# src/eval — Evaluation Layer

- **Created**: 2026-09-06T13:14:10+07:00
- **Last Updated**: 2026-09-06T13:14:10+07:00

---


## Architecture Overview (LAB2)

The evaluation layer produces the reported test metrics consumed by analysis
notebooks and reports (with 5W1H context — see
[agents/rules/RESULTS_REPORTING.md](../../agents/rules/RESULTS_REPORTING.md)):

```
checkpoints (experiments/runs/...) → evaluate_model.py → results JSON / plots
        (load via src.utils.checkpoint_utils.find_best_checkpoint)
```

- `evaluate()` — average loss + top-1 accuracy on a loader (test/val).
- `per_class_accuracy()` — per-class accuracy + full confusion matrix.
- `format_comparison_table()` — human-readable comparison table across variants.
- `load_checkpoint()` — safe `weights_only=True` loading of legacy
  state-dict-only checkpoints.
- `CIFAR10_CLASSES` — canonical class-name list used across the lab.

Test metrics are persisted by analysis notebooks into `experiments/results/`
(`test_metrics.json`, `comparison_table.txt`, `training_history.json`); every
file is described in `experiments/results/README.md`.
