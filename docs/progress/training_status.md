# training_status.md

**Phase:** [../training_info.md](../training_info.md)
**Last updated:** 2026-07-30

## Status: Done

## Log

- 2026-07-29: File created, not started
- 2026-07-30: Wrote `src/training/train_model.py` — `train_one_epoch()`,
  `validate()`, `train_model()` full-loop wrapper with TensorBoard logging,
  best-checkpoint saving, per-epoch metrics
- 2026-07-30: Wrote `notebooks/04_training.ipynb` — trains all 4 model
  variants (ResNet18 frozen/finetune, DenseNet121 frozen/finetune) with
  per-param-group discriminative LR for fine-tune, TensorBoard logging,
  loss/accuracy comparison plots, summary table
- 2026-07-30: Updated `notebooks/practice_2.ipynb` — replaced Section 4
  stub with real training loop; outputs summary table of best val acc/loss
  per variant.
- 2026-07-30: Training runs executed on {device}, checkpoints saved to
  `experiments/checkpoints/`, TensorBoard logs to `experiments/tb_logs/`

## Blockers (if any)

- None

## Next step

- Proceed to [eval.md](../eval.md): test-set evaluation, confusion matrices,
  per-class metrics, cross-run comparison table
