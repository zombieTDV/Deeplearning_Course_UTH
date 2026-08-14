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
| `EXP-EX2-FINETUNE` | Exercise 2 — HF `Trainer` finetune of `distilbert-base-uncased` on IMDB | Done | [EX2_IMDB_FINETUNE.md](EX2_IMDB_FINETUNE.md) |
| `EXP-EX2-HYPERPARAMETER-SEARCH` | Systematic hyperparameter search & advanced anti-overfitting protocol | Done | [EX3_HYPERPARAMETER_SEARCH_ANTI_OVERFITTING.md](EX3_HYPERPARAMETER_SEARCH_ANTI_OVERFITTING.md) |
| `EXP-EX2-ANTI-OVERFITTING-REPORT` | 5W1H Benchmark Report for EXP-01 through EXP-05 hyperparameter trials | Done | [EX3_ANTI_OVERFITTING_REPORT.md](EX3_ANTI_OVERFITTING_REPORT.md) |
| `EXP-EX2-ADVANCED-SPEC` | Specification for 4 advanced anti-overfitting techniques (Label Smoothing, LLRD, SWA) | Done | [EX4_ADVANCED_ANTI_OVERFITTING_PLAN.md](EX4_ADVANCED_ANTI_OVERFITTING_PLAN.md) |
| `EXP-EX2-ADVANCED-RESULTS` | Results & diagnostics report for the advanced anti-overfitting techniques (EXP-01…EXP-05) | Done | [EX4_RESULTS_AND_DIAGNOSTICS_REPORT.md](EX4_RESULTS_AND_DIAGNOSTICS_REPORT.md) |
| `EXP-EX2-512-BREAKTHROUGH` | **93.23% test accuracy** — EXP-06/EXP-07 512-token sequence-length expansion breakthrough | Done | [EX6_512_TOKENS_BREAKTHROUGH_REPORT.md](EX6_512_TOKENS_BREAKTHROUGH_REPORT.md) |



