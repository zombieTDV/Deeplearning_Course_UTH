# REPORT.md — Phase 8: Error Analysis, Interpretability & Reporting (agents/phases)

---

## Header

- **Title:** Error Analysis, Interpretability & Reporting
- **Execution order:** 8 of 8
- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-08-12T00:00:00+07:00
- **Description:** Group misclassified reviews by root cause (§19), show local interpretability examples (§20), and produce the final 5W1H report, demo notebook, and updated indexes.
- **Status:** Done

## Background

Pipeline stages §18–§20 of [ML_PIPELINE_REFERENCE_v3.md](../ML_PIPELINE_REFERENCE_v3.md): error analysis studies where and why the model fails (§19), interpretability explains individual predictions (§20), and the coursework deliverables (scripts, notebook, report) must be submission-ready with every metric under 5W1H. This phase covers roadmap tasks T15–T18 and milestone M6.

## Goals / Purpose

- What "done" looks like, concretely:
  - Root-cause groups for misclassified reviews documented (frequency × severity prioritization).
  - Local interpretability examples (sample predictions with confidence).
  - Demo notebook `notebooks/01_imdb_sentiment_analysis.ipynb` runs standalone (loads artifacts only).
  - Final report written; [OVERVIEW.md](../OVERVIEW.md), `experiments/results/README.md`, and `experiments/runs/registry.json` updated.
- What this phase explicitly does NOT try to solve:
  - No new training, no re-running experiments.

## Input / Output

- **Input:** metrics and misclassifications (Phase 7); run artifacts (`experiments/runs/<ts>_<run>/`).
- **Output:** this doc; `notebooks/01_imdb_sentiment_analysis.ipynb`; final report; updated [OVERVIEW.md](../OVERVIEW.md), `experiments/results/README.md`, `experiments/runs/registry.json`.

## How to do it (general plan)

1. Extract misclassified test reviews; group by root cause per the [§19.1 table](../ML_PIPELINE_REFERENCE_v3.md#191-process-for-labeled-data-supervised-learning) (label noise, rare patterns, long reviews, etc.); prioritize by frequency × severity.
2. Local interpretability: show sample predictions with confidence scores (§20).
3. Build the demo notebook with the header per [NOTEBOOK_HEADER_CONVENTION.md](../rules/NOTEBOOK_HEADER_CONVENTION.md) — analysis only, no training loop.
4. Write the final report with 5W1H for every metric (results vs baseline delta from Phase 4).
5. Update [OVERVIEW.md](../OVERVIEW.md), `experiments/results/README.md` (5W1H index), and `experiments/runs/registry.json`.

## Pipeline

```
metrics + misclassifications (Phase 7) → root-cause groups (§19)
  → notebook notebooks/01_imdb_sentiment_analysis.ipynb (loads artifacts)
  → final report → OVERVIEW.md + experiments/results/README.md + registry.json
```

## Detailed plan / gotchas

- Notebooks never train and never re-persist run state ([§1 of the logging rules](../rules/LOGGING_CHECKPOINT_RULES.md#1-script-only-runs--automatic-persistence)).
- Every reported metric carries a 5W1H block ([§2 of RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md#2-required-5w1h-block)).
- Label noise sets an accuracy ceiling (~10–20% in public datasets, §19) — attribute residual errors correctly.
- Keep the notebook header in ONE markdown cell (NOTEBOOK_HEADER_CONVENTION.md).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 8, §5 T15–T18)
- Progress tracking: [../progress/REPORT_STATUS.md](../progress/REPORT_STATUS.md)
- Related phases: [EVAL.md](EVAL.md), [OVERVIEW.md](../OVERVIEW.md)
- Reference: [ML_PIPELINE_REFERENCE_v3.md §19–§20](../ML_PIPELINE_REFERENCE_v3.md)
