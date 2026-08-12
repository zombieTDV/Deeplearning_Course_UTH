# src/experiments — Experiment Scripts

Long-running experiment pipelines (benchmarks, meta-model training, ablations)
live here as CLI scripts: `python -m src.experiments.<experiment>`.

- `baseline_imdb_sentiment.py` — Exercise 1 zero-shot baseline CLI: pipeline
  demo, IMDB test evaluation, majority-class floor, ROC-AUC, TensorBoard
  scalars, 5W1H JSON persistence (**implemented**, Phase 4 — Done).

## Rules

- Every experiment script persists its own outputs (JSON/NPZ/state dicts)
  under `experiments/results/<experiment>/` and indexes them in
  [experiments/results/README.md](../../experiments/results/README.md) with
  5W1H context.
- Document each experiment in
  [agents/experiments/README.md](../../agents/experiments/README.md) from
  [EXPERIMENT_TEMPLATE.md](../../agents/experiments/EXPERIMENT_TEMPLATE.md).
- Smoke-test before real runs
  ([SMOKE_TEST_CHECKLIST.md](../../agents/templates/SMOKE_TEST_CHECKLIST.md)).

> **Status:** `baseline_imdb_sentiment.py` is **implemented** (Phase 4 — Done,
> see [BASELINE.md](../../agents/phases/BASELINE.md)). Further experiment
> scripts are still **planned** (later phases; audit finding `ARC-1`, see
> [agents/CODEBASE_AUDIT_REPORT.md](../../agents/CODEBASE_AUDIT_REPORT.md)).
