# src/data — Data Pipeline Layer

## Architecture Overview (LAB2)

The data layer is the first stage of the LAB2 pipeline:

```
data/raw (CIFAR-10) → transforms.py → dataloader.py → train/val/test loaders
                          │
data/processed/cifar10_split_seed42.json  (fixed 45k/5k/10k split, seed 42)
```

- **transforms.py** — augmentation pipelines: standard ImageNet resize/normalize,
  plus the advanced SOTA pipeline (RandAugment + RandomErasing) for training.
  Also exposes `IMAGENET_MEAN`/`IMAGENET_STD` constants.
- **dataloader.py** — the **canonical loader**. `get_cifar10_loaders()` reads the
  persisted split file (generated once, reused forever), gates downloads via
  `_cifar10_present()`, and returns `(train, val, test)` DataLoaders.
- **statistics.py / inspection.py** — dataset statistics and validation helpers.
- **config.py** — configuration loader.
- **load_cifar10.py / dataset.py** — **legacy** duplicates, kept for backward
  compatibility only; new code must import from `dataloader.py`/`transforms.py`.

Data is stored once under `data/raw/` (single source of truth) — no re-download
and no duplicate copies under an external data directory.

## Usage

```python
from src.data.dataloader import get_cifar10_loaders
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)
```

## Performance notes

- **`num_workers` is 0 by default** (main-process loading) — the safe choice on
  Python 3.14, where multiprocess workers caused `BrokenPipeError`
  ([BUG-01](../../agents/bugs/BUG_01_DATALOADER_BROKEN_PIPE_PYTHON314.md)).
  Set it explicitly (or edit `configs/data.yaml`) to re-enable workers on a
  stable runtime.
- Loaders build **two** raw CIFAR-10 instances (one train shared by train/val
  via `_ApplyTransform`, one test) instead of three — see
  `src/data/dataloader.py`.

See [agents/phases/DATA_PREP.md](../../agents/phases/DATA_PREP.md) and
[agents/phases/DATALOADER.md](../../agents/phases/DATALOADER.md).
