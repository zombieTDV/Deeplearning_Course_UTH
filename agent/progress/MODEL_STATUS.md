# model_status.md

**Phase:** [../model.md](../model.md)
**Last updated:** 2026-07-30

## Status: Done

## Log

- 2026-07-29: File created, not started
- 2026-07-30: Checked torchvision 0.28.0 — `weights=` API confirmed,
  `ResNet18_Weights.DEFAULT` and `DenseNet121_Weights.DEFAULT` both available
- 2026-07-30: Wrote `src/models/build_model.py` — two build functions
  (`build_resnet18`, `build_densenet121`) each with `mode="frozen"` and
  `mode="finetune"`, plus `count_trainable_params` / `count_all_params` helpers
- 2026-07-30: Added ``device`` parameter to ``build_resnet18`` and
  ``build_densenet121`` — when provided, model is moved to target device
  before returning (fixes CUDA/CPU mismatch errors)
- 2026-07-30: Wrote `notebooks/03_model_setup.ipynb` — architecture inspection,
  final-layer replacement, freeze verification, forward pass sanity check
- 2026-07-30: Updated `notebooks/practice_2.ipynb` — replaced Section 3 stubs
  with real model code (build 4 variants, param-count summary table,
  forward-pass check).  All build calls updated to pass ``device=device``.

## Blockers (if any)

- None

## Next step

- Proceed to [training_info.md](../training_info.md): define loss function,
  optimizers (per-param-group for fine-tune variants), training loop,
  TensorBoard logging, hyperparameter sweep
