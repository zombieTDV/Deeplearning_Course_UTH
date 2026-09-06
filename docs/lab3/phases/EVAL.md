# EVAL.md — Phase 7: Evaluation & Validation (agents/phases)

---

## Header

- **Title:** Evaluation & Validation
- **Execution order:** 7 of 8
- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-08-12T00:00:00+07:00
- **Description:** Evaluate the finetuned model exactly once on the held-out IMDB test set (accuracy, confusion matrix, per-class precision/recall/F1) and run optional single-variable experiments with 5W1H reporting.
- **Status:** Done

## Background

Pipeline stages §16–§18 of [ML_PIPELINE_REFERENCE_v3.md](../ML_PIPELINE_REFERENCE_v3.md): metrics must be reported with 5W1H (§16.3), the test set is evaluated once (golden rule 4, §10), and experiments follow the single-variable principle (§18). K-Fold CV is explicitly N/A for this coursework scope (§17) and is documented as such. This phase covers roadmap tasks T13–T14 and milestone M5.

## Goals / Purpose

- What "done" looks like, concretely:
  - `src/eval/evaluate_model.py` outputs accuracy, confusion matrix, and per-class precision/recall/F1 with full 5W1H.
  - Optional: single-variable experiment (one lr or epoch delta) and μ±σ across 2–3 seeds.
  - K-Fold N/A decision documented.
- What this phase explicitly does NOT try to solve:
  - No error analysis (Phase 8), no hyperparameter sweeps (out of scope).

## Input / Output

- **Input:** best checkpoint (Phase 6), tokenized test set (Phase 3).
- **Output:** `experiments/results/imdb_sentiment_eval.json`; confusion-matrix plot in `experiments/plots/`.

## How to do it (general plan)

1. Load `<run>_best.pt` with `weights_only=True`; rebuild model + tokenizer.
2. Run inference on the test set once; compute accuracy, confusion matrix, per-class precision/recall/F1 (§16.1).
3. Report with full 5W1H per [RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md); classes are balanced so accuracy is a meaningful headline.
4. Optional: single-variable experiment (one lr or epoch delta) and μ±σ across 2–3 seeds (§17.4, §18.3).
5. Document the K-Fold N/A decision in this doc.

## Pipeline

```
python -m src.eval.evaluate_model --checkpoint experiments/runs/<ts>_<run>/checkpoints/<run>_best.pt
  → test-set metrics → experiments/results/imdb_sentiment_eval.json
  → confusion-matrix plot → experiments/plots/
```

## Detailed plan / gotchas

- Evaluate the test set exactly once; never tune against it (golden rule 4, [§10](../ML_PIPELINE_REFERENCE_v3.md#10-train--test-split-and-data-leakage)).
- Verify `id2label`/`label2id` before inference (roadmap risk R6).
- Every number in the report needs What/Why/When/Where/Who/How ([RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md)).
- Report `μ ± σ` if the seed experiment runs (σ < 0.02 = stable, [§17.4](../ML_PIPELINE_REFERENCE_v3.md#174-reporting-results--scientific-standard)).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 7, §5 T13–T14)
- Progress tracking: [../progress/EVAL_STATUS.md](../progress/EVAL_STATUS.md)
- Related phases: [TRAINING_INFO.md](TRAINING_INFO.md), [REPORT.md](REPORT.md)
- Reference: [ML_PIPELINE_REFERENCE_v3.md §16–§18](../ML_PIPELINE_REFERENCE_v3.md)
