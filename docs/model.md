# model.md

## Name
Model — Pretrained ResNet / DenseNet Adaptation

## Background
Covers Practice 2, steps 2–4: loading pretrained models, inspecting
architecture, and adapting the final classification layer for CIFAR-10's
10 classes.

## Goals / Purpose
- Load ResNet18 and DenseNet121 with ImageNet-pretrained weights
- Understand each architecture's structure well enough to know exactly
  which layer to replace and which to freeze
- Produce two adapted models ready for training: one frozen-backbone
  (feature extraction), one partially unfrozen (fine-tuning)

## Input / Output
- **Input:** none (weights fetched via `torchvision.models`)
- **Output:** model objects (`.pt` state dicts saved after training,
  not here — this doc covers architecture setup only)

## How to do it (general plan)
1. Load both models with `pretrained=True` (or `weights=...` on newer
   torchvision versions — check installed version, API changed)
2. Print each model to inspect structure, identify the final
   classification layer
3. Replace the final layer to output 10 classes instead of 1000
4. Freeze all other layers for the transfer-learning variant
5. For the fine-tuning variant, unfreeze the last block(s)

## Pipeline
```
torchvision.models.resnet18(weights=...) → inspect .fc layer →
replace nn.Linear(in_features, 10) → freeze/unfreeze → ready for training

torchvision.models.densenet121(weights=...) → inspect .classifier layer →
replace nn.Linear(in_features, 10) → freeze/unfreeze → ready for training
```

## Detailed experiment plan
- **ResNet18:** final layer is `model.fc` (`nn.Linear(512, 1000)`) →
  replace with `nn.Linear(512, 10)`
- **DenseNet121:** final layer is `model.classifier`
  (`nn.Linear(1024, 1000)`) → replace with `nn.Linear(1024, 10)`
- **Freezing:** set `param.requires_grad = False` for all params except
  the new final layer, for the "feature extraction" variant
- **Fine-tuning variant:** unfreeze last residual block (ResNet) /
  last dense block (DenseNet) in addition to the final layer, use a
  lower LR for these unfrozen layers than for the new final layer
  (this is the standard discriminative fine-tuning trick — worth
  testing but not required for a first pass)
- **Check torchvision version first** — `pretrained=True` is deprecated
  in newer versions in favor of `weights=ResNet18_Weights.DEFAULT`;
  confirm which API applies before writing code, don't assume

## Links
- Related phase docs: [overview.md](overview.md), [data_prep.md](data_prep.md),
  [training_info.md](training_info.md)
- Progress tracking: [progress/model_status.md](progress/model_status.md)
