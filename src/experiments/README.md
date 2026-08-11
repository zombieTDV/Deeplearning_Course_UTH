# src/experiments — Experiment Scripts

Long-running experiment pipelines (benchmarks, meta-model training, ablations)
live here as CLI scripts: `python -m src.experiments.<experiment>`.

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

> **Status:** experiment scripts are **planned** (roadmap Phase 4 —
> [BASELINE.md](../../agents/phases/BASELINE.md), and later phases); none
> implemented yet — audit finding `ARC-1`
> ([agents/CODEBASE_AUDIT_REPORT.md](../../agents/CODEBASE_AUDIT_REPORT.md)).
