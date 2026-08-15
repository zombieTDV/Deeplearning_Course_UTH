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
| `EXP-EX2-HTML-CLEANING` | **Token Economy & Context Reclaim** — EX-07 HTML `<br />` Artifact Stripping Report | Done | [EX7_HTML_DATA_CLEANING_REPORT.md](EX7_HTML_DATA_CLEANING_REPORT.md) |
| `EXP-EX2-CLEANLAB-DENOISING` | **Data-Centric AI** — EX-08 Confident Learning & Label Error Audit (Cleanlab) Report | Done | [EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md](EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md) |
| `EXP-EX2-LORA-RANK32-OPTIMAL` | **Anti-Overfitting Triad** — EX-09 LoRA Rank 32 + Multi-Tier Regularization Report | Done | [EX9_LORA_RANK32_OPTIMAL_REGULARIZATION_REPORT.md](EX9_LORA_RANK32_OPTIMAL_REGULARIZATION_REPORT.md) |
| `EXP-EX2-LORA-LOW-LR-AUDIT` | **Underfitting Diagnostics** — EX-10 LoRA Micro-LR (5e-5) Audit Report | Done | [EX10_LORA_LOW_LR_UNDERFITTING_REPORT.md](EX10_LORA_LOW_LR_UNDERFITTING_REPORT.md) |
| `EXP-EX2-LORA-5E5-STOPPING` | **Early Stopping Telemetry** — EX-11 LoRA 5e-5 Replication & Underfit Confirmation | Done | [EX11_LORA_5E5_EARLY_STOPPING_REPORT.md](EX11_LORA_5E5_EARLY_STOPPING_REPORT.md) |
| `EXP-EX2-LORA-POSTMORTEM` | **Root-Cause Analysis** — EX-12 LoRA Micro-LR (5e-5) Post-Mortem & Sweet-Spot Guide | Done | [EX12_LORA_5E5_DENOISED_EVAL_REPORT.md](EX12_LORA_5E5_DENOISED_EVAL_REPORT.md) |
| `EXP-EX2-HEAD-TAIL-UPGRADE` | **Context & Verdict Retention** — EX-13 Head+Tail Truncation & Peak Checkpoint Resolution | Done | [EX13_HEAD_TAIL_TRUNCATION_REPORT.md](EX13_HEAD_TAIL_TRUNCATION_REPORT.md) |
| `EXP-EX2-LORA-DENOISED-PEAK` | **93.14% Test Accuracy** — EX-14 LoRA on Cleanlab Denoised IMDB with Head-Tail Truncation | Done | [EX14_LORA_DENOISED_HEAD_TAIL_BREAKTHROUGH_REPORT.md](EX14_LORA_DENOISED_HEAD_TAIL_BREAKTHROUGH_REPORT.md) |








