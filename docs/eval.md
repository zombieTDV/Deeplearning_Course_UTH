# eval.md

## Name
Evaluation — Test Set Performance & Model Comparison

## Background
Covers Practice 2, step 7: evaluate trained models on the held-out
CIFAR-10 test set, and compare across the ResNet18 vs DenseNet121 and
frozen vs fine-tuned runs from [training_info.md](training_info.md).

## Goals / Purpose
- Get a fair, held-out accuracy number for each trained variant
- Compare ResNet18 vs DenseNet121, frozen vs fine-tuned, across the
  learning rate / batch size sweep
- Identify which configuration generalizes best, not just which fits
  train data best

## Input / Output
- **Input:** trained checkpoints from `experiments/<run_id>/model.pt`,
  `test_loader` from [data_prep.md](data_prep.md)
- **Output:** a comparison table (accuracy, and optionally per-class
  precision/recall) across all runs

## How to do it (general plan)
1. Load each checkpoint, set `model.eval()`
2. Run inference over `test_loader` with `torch.no_grad()`
3. Compute overall accuracy; optionally per-class metrics /
   confusion matrix
4. Log final test metrics to TensorBoard (as a single point, or as
   text summary) alongside the training curves for that run
5. Collect results into one comparison table across all runs

## Pipeline
```
load checkpoint → model.eval() → test_loader → predictions →
accuracy / confusion matrix → append to comparison table
```

## Detailed experiment plan
- **Primary metric:** overall test accuracy
- **Secondary:** per-class accuracy (CIFAR-10 has 10 balanced classes,
  so per-class breakdown can reveal if a model struggles on specific
  categories, e.g. cat vs dog confusion is a classic CIFAR-10 failure)
- **Comparison table columns:** model, frozen/fine-tuned, LR, batch
  size, epochs trained, val accuracy, test accuracy
- **Known gotcha:** never tune anything based on test set results —
  if a decision needs to be made (e.g. picking best LR), that decision
  should already be settled from val accuracy in training_info.md;
  test set is for final reporting only

## Links
- Related phase docs: [overview.md](overview.md), [training_info.md](training_info.md)
- Progress tracking: [progress/eval_status.md](progress/eval_status.md)
