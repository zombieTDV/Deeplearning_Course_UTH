# Model Experiments Log

This directory contains experiment plans, hyperparameter trial logs, and model
comparison documentation in `.md` format.

## 🏗️ Architecture Overview

Experiments are the **research layer** between training and reporting:

```
Training scripts (src/training, src/experiments/*_train.py)
        │  full-state checkpoints + JSONL history (experiments/runs/)
        ▼
This folder: plans, status, results — every number carries 5W1H context
        │
        ▼
Summary reports → teacher/team presentations
```

Rules: results must follow the 5W1H principle
([agents/rules/RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md)); training
never happens in notebooks
([LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md)).

## 📋 Experiments Index

Add one row per experiment document, created from
[EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md):

| ID | Title & Summary | Status | Link |
|:--:|:---|:---:|:---|
| `EXP-EX1-BASELINE` | Exercise 1 zero-shot sentiment pipeline & IMDB baseline floor | Done | [EX1_SENTIMENT_BASELINE.md](EX1_SENTIMENT_BASELINE.md) |
| `EXP-EX2-FINETUNE` | Exercise 2 — HF `Trainer` finetune of `distilbert-base-uncased` on IMDB | To Do | [EX2_IMDB_FINETUNE.md](EX2_IMDB_FINETUNE.md) |
