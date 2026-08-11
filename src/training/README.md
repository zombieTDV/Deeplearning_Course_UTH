# src/training — Training Layer (scripts only)

What belongs here:

- `train_model.py` — the generic training loop: full-state checkpointing
  (model + optimizer + scheduler + RNG + history + config), early stopping,
  resume, TensorBoard hook.
- `<feature>_train.py` — per-feature CLI entry points that wire data + model +
  loop together and are run with `python -m src.training.<feature>_train`.
- `run_logger.py` / `checkpoint_utils.py` — shared helpers owned by
  [src/utils/README.md](../utils/README.md); consume them from there (do not
  re-implement here).

## Hard rule

**Training runs only from these scripts.** Notebooks never contain a training
loop; they load artifacts produced here. See
[agents/rules/LOGGING_CHECKPOINT_RULES.md](../../agents/rules/LOGGING_CHECKPOINT_RULES.md).

See [agents/phases/TRAINING_INFO.md](../../agents/phases/TRAINING_INFO.md).

> **Status:** `train_model.py` and `<feature>_train.py` are **planned**
> (roadmap Phase 6); not implemented yet — audit finding `ARC-1`
> ([agents/CODEBASE_AUDIT_REPORT.md](../../agents/CODEBASE_AUDIT_REPORT.md)).
