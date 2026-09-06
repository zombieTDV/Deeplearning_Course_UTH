# MODEL.md — Phase 5: Model Selection & Training Config (agents/phases)

---

## Header

- **Title:** Model Selection & Training Config
- **Execution order:** 5 of 8
- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-08-12T00:00:00+07:00
- **Description:** Justify `distilbert-base-uncased` for Exercise 2 (No Free Lunch, §12) and define the training hyperparameters with bias-variance reasoning (§13–§15).
- **Status:** In Progress

## Background

Pipeline stages §12–§15 of [ML_PIPELINE_REFERENCE_v3.md](../ML_PIPELINE_REFERENCE_v3.md): model selection must match the inductive bias to the data structure and constraints (VRAM ≤3.5 GB on 4 GB team machines, coursework compute), and every hyperparameter must be justified and logged. The brief in [PURPOSE.md](../PURPOSE.md) locked `distilbert-base-uncased`; this phase documents why and pins the config. This phase covers roadmap tasks T8–T9.

## Goals / Purpose

- What "done" looks like, concretely:
  - Model selection rationale written in this doc (No Free Lunch reasoning, §12).
  - `configs/config_imdb_sentiment.yaml` created with lr, epochs, batch size, weight decay, `max_length`.
  - Bias-variance reasoning documented for each choice (§13–§15).
- What this phase explicitly does NOT try to solve:
  - No implementation of the training loop (Phase 6), no hyperparameter sweeps (out of scope).

## Input / Output

- **Input:** EDA stats (Phase 2), split design (Phase 3).
- **Output:** this doc; `configs/config_imdb_sentiment.yaml`.

## How to do it (general plan)

1. Compare candidate families (e.g. `bert-base-uncased`, `distilbert-base-uncased`, `roberta-base`) on VRAM footprint, throughput, and expected accuracy (§12).
2. Document the choice: DistilBERT ≈ 66M params, ~40% smaller than BERT-base with near-parity accuracy — fits the ≤3.5 GB VRAM target.
3. Define hyperparameters (§13–§15): lr ≈ 2e-5, epochs 2–3, batch size tuned to VRAM (e.g. 16 with gradient accumulation), weight decay (AdamW default 0.01), `max_length` from Phase 3.
4. Write `configs/config_imdb_sentiment.yaml`; every value will be logged with each run (single-variable principle, §18).
5. Document bias-variance reasoning: more epochs ↔ overfitting risk; weight decay ↔ regularization.

## Pipeline

```
EDA + split design → MODEL.md rationale → configs/config_imdb_sentiment.yaml
  → consumed by src/training/imdb_sentiment_train.py (Phase 6)
```

## Detailed plan / gotchas

- Change exactly ONE hyperparameter between experiment runs (roadmap task T14; [§18.3](../ML_PIPELINE_REFERENCE_v3.md#183-the-single-variable-principle)).
- Keep batch size + gradient accumulation so peak VRAM stays ≤3.5 GB (roadmap risk R1).
- Log every config value with every run — required by §18.3 and [LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 5, §5 T8–T9)
- Progress tracking: [../progress/MODEL_STATUS.md](../progress/MODEL_STATUS.md)
- Related phases: [FEATURE_SPLIT.md](FEATURE_SPLIT.md), [TRAINING_INFO.md](TRAINING_INFO.md)
- Reference: [ML_PIPELINE_REFERENCE_v3.md §12–§15](../ML_PIPELINE_REFERENCE_v3.md)
