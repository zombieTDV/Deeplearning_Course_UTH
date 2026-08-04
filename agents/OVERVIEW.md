# OVERVIEW.md

## Project
Practice 2 — Transfer learning with pre-trained architectures on CIFAR-10.

## Purpose
See [PURPOSE.md](PURPOSE.md) for the original exercise brief.

## Plan
- **Dataset:** CIFAR-10 (10 classes, 32x32 RGB, 50k train / 10k test)
- **Models:** ResNet (start with ResNet18), DenseNet (start with DenseNet121),
  both loaded via `torchvision.models` with ImageNet-pretrained weights
- **Approach:** transfer learning first (freeze backbone, replace + train
  final layer), then fine-tuning pass (unfreeze last block(s), lower LR)
- **Monitoring:** TensorBoard for loss/accuracy curves across runs
- **Comparison target:** ResNet18 vs DenseNet121, frozen vs fine-tuned,
  at minimum 2 learning rates and 2 batch sizes each

## Phases
1. [DATA_PREP.md](phases/DATA_PREP.md) — CIFAR-10 loading, transforms, DataLoaders
2. [MODEL.md](phases/MODEL.md) — loading pretrained ResNet/DenseNet, architecture inspection, adapting final layer, freeze/unfreeze strategy
3. [TRAINING_INFO.md](phases/TRAINING_INFO.md) — training loop, hyperparameters, TensorBoard logging
4. [EVAL.md](phases/EVAL.md) — test-set evaluation, metrics, comparison across runs
5. [PRACTICE2_EXP07_UPGRADE_PLAN.md](phases/PRACTICE2_EXP07_UPGRADE_PLAN.md) — Master SOTA Benchmarking Upgrade Plan (LLRD, RandAugment, Soft-Voting 96.00% Ensemble)
6. [PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md](phases/PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md) — EarlyStopping Callback Integration & Misclassified Error Analysis Plan
7. [LOGIT_BIAS_SWEEP_STATUS.md](experiments/LOGIT_BIAS_SWEEP_STATUS.md) — Class-Logit Bias Sweep Tuning & Threshold Optimization Results

## Bug Reports & Troubleshooting
- [bugs/README.md](bugs/README.md) — Master Bug Index & Troubleshooting Directory
- [bugs/BUG_01_DATALOADER_BROKEN_PIPE_PYTHON314.md](bugs/BUG_01_DATALOADER_BROKEN_PIPE_PYTHON314.md) — PyTorch DataLoader `BrokenPipeError` in Multiprocessing Workers under Python 3.14 & Linux Jupyter Environments
- [bugs/BUG_02_PRACTICE2_TRAIN_MODEL_TEST_LOSSES_KEYERROR.md](bugs/BUG_02_PRACTICE2_TRAIN_MODEL_TEST_LOSSES_KEYERROR.md) — `KeyError: 'test_losses'` in Practice 2 Notebook & Build Script

## Known constraint
CIFAR-10 images are 32x32; torchvision pretrained models expect 224x224
(ImageNet-sized) input. This needs an explicit resize in the transform
pipeline — see [DATA_PREP.md](phases/DATA_PREP.md). Flag this early; it's the
most common thing that silently produces garbage results if missed.

## Progress
- [progress/DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md)
- [progress/MODEL_STATUS.md](progress/MODEL_STATUS.md)
- [progress/TRAINING_STATUS.md](progress/TRAINING_STATUS.md)
- [progress/EVAL_STATUS.md](progress/EVAL_STATUS.md)
- [experiments/LOGIT_BIAS_SWEEP_STATUS.md](experiments/LOGIT_BIAS_SWEEP_STATUS.md)
