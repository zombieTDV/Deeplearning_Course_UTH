# data_prep_status.md

**Phase:** [../data_prep.md](../data_prep.md)
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
  following notebook_header_convention.md; calls `get_cifar10_loaders()`,
  visualises raw/transformed samples, class distribution, split sizes
- 2026-07-30: Wrote smoke test script `smoke_test_data_prep.py` — verifies
  split file creation, batch shapes (64,3,224,224), no NaNs, reuse on
  second call. Smoke test requires CIFAR-10 download (~170 MB) on first run.

## Blockers (if any)

- None

## Next step

- Proceed to [model.md](../model.md): load pretrained ResNet18/DenseNet121,
  inspect architecture, replace final layer, freeze/unfreeze strategy
