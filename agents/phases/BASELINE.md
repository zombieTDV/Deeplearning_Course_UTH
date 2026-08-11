# BASELINE.md — Phase 4: Baseline Model (agents/phases)

---

## Header

- **Title:** Baseline Model (Exercise 1 — Zero-Shot Sentiment Analysis)
- **Execution order:** 4 of 8
- **Date created:** 2026-08-11
- **Last updated:** 2026-08-11
- **Description:** Deliver Exercise 1 — run a pretrained Hugging Face sentiment pipeline on sample sentences and record baseline metrics (including the majority-class floor) as the reference for Exercise 2.
- **Status:** To Do

## Background

Pipeline stage §11 of [ML_PIPELINE_REFERENCE_v3.md](../ML_PIPELINE_REFERENCE_v3.md): baseline-first. Exercise 1 is itself the baseline: a zero-shot, pretrained sentiment model from the Hub with no finetuning. Its measured accuracy on the IMDB test set, plus the majority-class accuracy, establishes the floor that the finetuned model in Exercise 2 must beat. This phase covers roadmap task T7 and milestone M2.

## Goals / Purpose

- What "done" looks like, concretely:
  - Baseline script runs on sample English sentences (Exercise 1 steps 1–4: install, use a pretrained Hub model, tokenize a sentence, run sentiment analysis).
  - Zero-shot accuracy on the IMDB test set and majority-class accuracy recorded with full 5W1H.
- What this phase explicitly does NOT try to solve:
  - No finetuning (Phase 6), no final evaluation protocol (Phase 7).

## Input / Output

- **Input:** verified environment (Phase 1); sample sentences; IMDB test labels.
- **Output:** `src/experiments/baseline_imdb_sentiment.py`; baseline metrics JSON in `experiments/results/`.

## How to do it (general plan)

1. Write the baseline script using `pipeline("sentiment-analysis")` with a pretrained Hub model (Exercise 1 step 2).
2. Tokenize a sample sentence (Exercise 1 step 3) and run sentiment analysis (step 4) — this also demonstrates the tokenizer.
3. Compute the zero-shot model's accuracy on the IMDB test set.
4. Compute the majority-class baseline accuracy (always predict the majority label).
5. Record all metrics with full 5W1H context per [RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md).

## Pipeline

```
python -m src.experiments.baseline_imdb_sentiment --samples "I loved this film."
  → pipeline predictions (zero-shot)
  → test-set zero-shot accuracy + majority-class accuracy
  → experiments/results/baseline_imdb_sentiment.json
```

## Detailed plan / gotchas

- The default sentiment pipeline returns POSITIVE/NEGATIVE — map to IMDB labels explicitly (roadmap risk R6).
- Record model id, checkpoint/date, seed, and split for every reported number (5W1H; §11.4).
- The zero-shot vs finetuned accuracy delta is the headline of the final report (T18).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 4, §5 T7)
- Progress tracking: [../progress/BASELINE_STATUS.md](../progress/BASELINE_STATUS.md)
- Related phases: [SETUP.md](SETUP.md), [EVAL.md](EVAL.md)
- Reference: [ML_PIPELINE_REFERENCE_v3.md §11](../ML_PIPELINE_REFERENCE_v3.md#11-baseline-thinking)
