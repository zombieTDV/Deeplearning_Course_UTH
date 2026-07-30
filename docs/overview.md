# overview.md

## Project
Practice 2 — Transfer learning with pre-trained architectures on CIFAR-10.

## Purpose
See [project_root/purpose.md](../purpose.md) for the original exercise brief.

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
1. [data_prep.md](data_prep.md) — CIFAR-10 loading, transforms, DataLoaders
2. [model.md](model.md) — loading pretrained ResNet/DenseNet, architecture
   inspection, adapting final layer, freeze/unfreeze strategy
3. [training_info.md](training_info.md) — training loop, hyperparameters,
   TensorBoard logging
4. [eval.md](eval.md) — test-set evaluation, metrics, comparison across runs

## Known constraint
CIFAR-10 images are 32x32; torchvision pretrained models expect 224x224
(ImageNet-sized) input. This needs an explicit resize in the transform
pipeline — see [data_prep.md](data_prep.md). Flag this early; it's the
most common thing that silently produces garbage results if missed.

## Progress
- [progress/data_prep_status.md](progress/data_prep_status.md)
- [progress/model_status.md](progress/model_status.md)
- [progress/training_status.md](progress/training_status.md)
- [progress/eval_status.md](progress/eval_status.md)
