# src/training — Training Layer (script-only)

- **Created**: 2026-09-06T13:14:10+07:00
- **Last Updated**: 2026-09-06T13:14:10+07:00

---


## Architecture Overview (LAB2)

All model training runs here as Python scripts — **notebooks never train**:

```
train_lab2_models.py (CLI)          # 6 variants: ResNet18/DenseNet121 × frozen/finetune/sota
        │
        └── train_model.py          # training loop, validation, EarlyStopping
                ├── save_checkpoint() / load_checkpoint_state()   # full state
                ├── capture/restore_rng_state()                   # exact resume
                └── RunLogger (src/utils/run_logger.py)           # progress + logs
```

Artifacts per run (`experiments/runs/<ts>_<run>/`):

| Path | Content |
|---|---|
| `checkpoints/<run>_best.pt` | best validation-loss state (full dict) |
| `checkpoints/<run>_last.pt` | every-epoch state — the resume source |
| `logs/<run>.log` | timestamped log (gzip rotation at 20 MB) |
| `metrics/<run>_config.json` | hyperparameters + 5W1H description |
| `metrics/<run>_history.jsonl` | one JSON line per epoch |
| `tensorboard/` | optional TensorBoard events (`--tb`) |

## Usage

```bash
python -m src.training.train_lab2_models [--modes frozen finetune sota] [--epochs 20]
                                         [--seed 42] [--resume] [--tb] [--smoke]
```

- **Resume**: re-run with `--resume`; restores model + optimizer + scheduler +
  RNG state + history and continues at the saved epoch.
- **Monitoring**: console progress line always on; TensorBoard via `--tb`.

Authoritative rules: [agents/rules/LOGGING_CHECKPOINT_RULES.md](../../agents/rules/LOGGING_CHECKPOINT_RULES.md).

## Module reference

- `train_model.py` — `train_model()`, `train_one_epoch()`, `validate()`,
  `EarlyStopping`, `save_checkpoint()`, `load_checkpoint_state()`,
  `load_model_weights()`, `capture_rng_state()`, `restore_rng_state()`.
- `train_lab2_models.py` — CLI entry point; builds variants, optimizers
  (incl. LLRD groups), schedulers, and orchestrates logged runs.
