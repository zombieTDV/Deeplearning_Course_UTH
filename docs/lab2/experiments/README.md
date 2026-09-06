# Model Experiments Log

- **Created**: 2026-09-06T13:14:10+07:00
- **Last Updated**: 2026-09-06T13:14:10+07:00

---


This directory contains experiment plans, hyperparameter trial logs, and model comparison documentation in `.md` format.

## 🏗️ Architecture Overview

Experiments are the **research layer** between training and reporting:

```
Training scripts (src/training, src/experiments/*_train.py)
        │  full-state checkpoints + JSONL history (experiments/runs/)
        ▼
This folder: plans, status, results — every number carries 5W1H context
        │
        ▼
Summary reports (SUMMARY_RESULTS.md) → teacher/team presentations
```

Rules: results must follow the 5W1H principle
([agents/rules/RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md)); training
never happens in notebooks ([LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md)).

## 📋 Experiments Index

- [CIFAR_STEM_EXPERIMENT.md](CIFAR_STEM_EXPERIMENT.md) — Native 32x32 Conv Stem Adaptation vs 224x224 Upsampling Baseline Experiment (Completed)
- [EXP_01_OPTUNA_HPO.md](EXP_01_OPTUNA_HPO.md) — Automated Hyperparameter Optimization (Optuna Sweep) Report (Completed — **92.06%**)
- [EXP_02_LR_SCHEDULER_LLRD.md](EXP_02_LR_SCHEDULER_LLRD.md) — Learning Rate Schedulers & Layer-wise LR Decay (LLRD) Report (Completed — **92.78%**)
- [EXP_03_ADVANCED_AUGMENTATIONS.md](EXP_03_ADVANCED_AUGMENTATIONS.md) — Advanced Data Augmentation (RandAugment, CutMix) & Regularization Report (Completed — **92.72%**)
- [EXP_04_STEM_NATIVE_LONG_TRAIN.md](EXP_04_STEM_NATIVE_LONG_TRAIN.md) — Native 32x32 Conv Stem Extended Training Report (Completed — **45.5s/epoch**)
- [EXP_05_MODEL_ARCH_SWEEP.md](EXP_05_MODEL_ARCH_SWEEP.md) — Modern Vision Architecture (ConvNeXt & EfficientNet) Benchmark Report (Completed — **96.42%**)
- [EXP_06_CONVNEXT_ADVANCED_SOTA.md](EXP_06_CONVNEXT_ADVANCED_SOTA.md) — Ultimate ConvNeXt-Tiny SOTA Combination Report (Completed — **97.66%**)
- [EXP_07_RESNET_DENSENET_SOTA.md](EXP_07_RESNET_DENSENET_SOTA.md) — ResNet18 & DenseNet121 Peak Accuracy Fine-Tuning & Ensemble Report (Completed — **96.00%**)
- [CODE_DIFFERENCE_PRACTICE2_VS_EXP07.md](CODE_DIFFERENCE_PRACTICE2_VS_EXP07.md) — Architectural Code Comparison Report (`practice_2.ipynb` vs `exp_07_resnet_densenet_sota.py`)
- [../phases/PRACTICE2_EXP07_UPGRADE_PLAN.md](../phases/PRACTICE2_EXP07_UPGRADE_PLAN.md) — Integration & Planning Guide for Upgrading `practice_2.ipynb` with EXP-07, Loss Curves & ROC-AUC Charts
- [../phases/PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md](../phases/PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md) — EarlyStopping Callback Integration & Misclassified Error Analysis Plan
- [LOGIT_BIAS_SWEEP_STATUS.md](LOGIT_BIAS_SWEEP_STATUS.md) — Class-Logit Bias Sweep Tuning & Threshold Optimization Results
- [SUMMARY_RESULTS.md](SUMMARY_RESULTS.md) — Final Benchmark Report & Loss Curves Visualization
