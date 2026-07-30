# data_prep.md

## Name

Data Preparation — CIFAR-10

## Background

Pretrained torchvision models (ResNet, DenseNet) expect 224x224 3-channel
input normalized with ImageNet mean/std. CIFAR-10 is natively 32x32, so
the transform pipeline must resize and normalize correctly, or training
will silently underperform without erroring.

## Goals / Purpose

- Load CIFAR-10 train/test splits
- Build a transform pipeline compatible with pretrained model input
  expectations
- Produce working `DataLoader`s for train/val/test

## Input / Output

- **Input:** CIFAR-10 (auto-downloaded via `torchvision.datasets.CIFAR10`)
- **Output:** `train_loader`, `val_loader`, `test_loader` — batched,
  transformed, ready to feed into training loop

## How to do it (general plan)

1. Download CIFAR-10 via `torchvision.datasets.CIFAR10(download=True)`
2. Split off a validation set from the training set (e.g. 45k/5k)
3. Define transforms: resize to 224x224, convert to tensor, normalize
   with ImageNet mean/std
4. Wrap in `DataLoader` with appropriate batch size, shuffle for train only

## Pipeline

```
torchvision.datasets.CIFAR10 → transforms.Compose([...]) → 
torch.utils.data.random_split (train/val) → DataLoader (train/val/test)
```

## Detailed experiment plan

- **Resize:** `transforms.Resize(224)` — required, not optional, for
  pretrained ResNet/DenseNet compatibility
- **Normalization:** use ImageNet stats
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
  (not CIFAR-10's own stats — the pretrained weights were trained on
  ImageNet-normalized data)
- **Augmentation (train only):** apply **after** resize, not before —
  `Resize(224)` → `RandomHorizontalFlip()` → `RandomCrop(224, padding=16)`
  → `ToTensor()` → `Normalize(...)`. Cropping the native 32x32 image
  first (before upsampling) gives negligible augmentation signal once
  diluted across 224x224; cropping at the working resolution is the
  correction from the original plan.
- **Batch size:** start at 64, this is one of the hyperparameters to
  sweep later (see [training_info.md](training_info.md))
- **Known gotcha:** resizing 32x32 → 224x224 upsamples, doesn't add
  real detail — expect this to cap accuracy somewhat vs native
  ImageNet-resolution images. Worth noting in eval writeup, not a bug.

## Single split policy (project-wide)

The train/val/test split is generated **once** and persisted to disk —
never regenerated on the fly, even with a fixed seed. Relying on seed
reproducibility across environments/library versions is fragile; a
persisted artifact is not.

- Split indices saved to `data/processed/cifar10_split_seed42.json`
  (lists of indices per split, plus the seed and ratio used)
- This file is treated as **read-only** the moment it's created — no
  script (data prep, model, training, eval) may regenerate or mutate it
- Every later phase (`model.md`, `training_info.md`, `eval.md`) loads
  the same split file — there is exactly one train/val/test split for
  the entire project, shared across every experiment and model compared
- If the split ever needs to change, that's a deliberate, logged
  decision — increment to `cifar10_split_v2.json`, don't overwrite v1

## Links

- Related phase docs: [overview.md](overview.md), [model.md](model.md)
- Progress tracking: [progress/data_prep_status.](progress/data_prep_status.md)

# data_prep.md

## Name

Data Preparation — CIFAR-10

## Background

Pretrained torchvision models (ResNet, DenseNet) expect 224x224 3-channel
input normalized with ImageNet mean/std. CIFAR-10 is natively 32x32, so
the transform pipeline must resize and normalize correctly, or training
will silently underperform without erroring.

## Goals / Purpose

- Load CIFAR-10 train/test splits
- Build a transform pipeline compatible with pretrained model input
  expectations
- Produce working `DataLoader`s for train/val/test

## Input / Output

- **Input:** CIFAR-10 (auto-downloaded via `torchvision.datasets.CIFAR10`)
- **Output:** `train_loader`, `val_loader`, `test_loader` — batched,
  transformed, ready to feed into training loop

## How to do it (general plan)

1. Download CIFAR-10 via `torchvision.datasets.CIFAR10(download=True)`
2. Split off a validation set from the training set (e.g. 45k/5k)
3. Define transforms: resize to 224x224, convert to tensor, normalize
   with ImageNet mean/std
4. Wrap in `DataLoader` with appropriate batch size, shuffle for train only

## Pipeline

```
torchvision.datasets.CIFAR10 → transforms.Compose([...]) → 
torch.utils.data.random_split (train/val) → DataLoader (train/val/test)
```

## Detailed experiment plan

- **Resize:** `transforms.Resize(224)` — required, not optional, for
  pretrained ResNet/DenseNet compatibility
- **Normalization:** use ImageNet stats
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
  (not CIFAR-10's own stats — the pretrained weights were trained on
  ImageNet-normalized data)
- **Augmentation (train only):** start simple —
  `RandomHorizontalFlip`, `RandomCrop(32, padding=4)` before resize
- **Batch size:** start at 64, this is one of the hyperparameters to
  sweep later (see [training_info.md](training_info.md))
- **Known gotcha:** resizing 32x32 → 224x224 upsamples, doesn't add
  real detail — expect this to cap accuracy somewhat vs native
  ImageNet-resolution images. Worth noting in eval writeup, not a bug.

## Links

- Related phase docs: [overview.md](overview.md), [model.md](model.md)
- Progress tracking: [progress/data_prep_status.md](progress/data_prep_status.md)
