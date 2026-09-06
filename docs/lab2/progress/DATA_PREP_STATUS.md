# DATA_PREP_STATUS.md

**Phase:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
**Last updated:** 2026-07-30

## Status: Done

## Log

- 2026-07-29: File created, not started
- 2026-07-30: Wrote `src/data/load_cifar10.py` — split-once persistence,
  `get_cifar10_loaders(batch_size=64, num_workers=2)` returning
  `(train_loader, val_loader, test_loader)` with ImageNet-compatible
  transforms (Resize(224) → RandomHorizontalFlip → RandomCrop(224,padding=16)
  → ToTensor → Normalize(ImageNet stats) for train; no augmentation for val/test)
- 2026-07-30: Wrote `notebooks/02_data_prep.ipynb` — practice deliverable
  following NOTEBOOK_HEADER_CONVENTION.md; calls `get_cifar10_loaders()`,
  visualises raw/transformed samples, class distribution, split sizes

## Blockers (if any)

- None

## Next step

- Proceed to [MODEL.md](../phases/MODEL.md): load pretrained ResNet18/DenseNet121,
  inspect architecture, replace final layer, freeze/unfreeze strategy
