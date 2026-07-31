# training_info.md

## Name
Training — Hyperparameters & TensorBoard Monitoring

## Background
Covers Practice 2, steps 6 + the hyperparameter/TensorBoard exercises:
loss function, optimizer, training loop, and monitoring.

## Goals / Purpose
- Train both adapted models (ResNet18, DenseNet121) on CIFAR-10
- Sweep at least learning rate and batch size
- Log everything to TensorBoard for comparison across runs

## Input / Output
- **Input:** adapted model from [model.md](model.md),
  DataLoaders from [data_prep.md](data_prep.md)
- **Output:** trained model checkpoints (`experiments/checkpoints/<run_name>_best.pt`),
  TensorBoard logs (`experiments/tb_logs/<run_name>/` via ``SummaryWriter``)

## How to do it (general plan)
1. Define loss function: `nn.CrossEntropyLoss()`
2. Ensure model is on the same device as input tensors
   (pass ``device=device`` to ``build_resnet18`` / ``build_densenet121``)
3. Define optimizer — only pass `requires_grad=True` params to it
4. Set up `SummaryWriter` for TensorBoard
4. Write training loop: forward, loss, backward, step, log
5. Run smoke test first (see `docs/smoke_test_checklist.md`) before any
   full run
6. Run full training sweep

## Pipeline
```
model (from model.md) + train_loader/val_loader (from data_prep.md) →
loss_fn + optimizer → training loop (per epoch: train step, val step,
log to TensorBoard) → checkpoint saved per run
```

## Detailed experiment plan
- **Optimizer:** start with `Adam`, compare against `SGD + momentum`
  if time allows
- **Learning rate sweep:** `1e-3`, `1e-4` at minimum (frozen backbone
  can tolerate higher LR since only the final layer trains; fine-tuned
  variant needs lower LR to avoid destroying pretrained features)
- **Batch size sweep:** `32`, `64`
- **Epochs:** enough for the loss curve to visibly plateau — check
  TensorBoard rather than hardcoding a number blindly; 10–15 is a
  reasonable starting ceiling for CIFAR-10 transfer learning
- **TensorBoard logging per step/epoch:**
  - `train/loss`, `train/accuracy`
  - `val/loss`, `val/accuracy`
  - learning rate (if using a scheduler)
- **Run naming:** `<model>_<lr>_<batch>_<date>` per naming_convention.md,
  so TensorBoard runs don't collide and stay comparable side-by-side
- **Known gotcha:** forgetting `model.eval()` / `model.train()` mode
  switches between train and val loops silently breaks BatchNorm
  behavior — double check this in the smoke test

## Links
- Related phase docs: [overview.md](overview.md), [model.md](model.md),
  [eval.md](eval.md)
- Progress tracking: [progress/training_status.md](progress/training_status.md)
