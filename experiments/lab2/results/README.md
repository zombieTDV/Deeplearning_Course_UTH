# experiments/results — Metrics Index & Descriptions

Every file below is described with the 5W1H principle (What / Why / When / Where
/ Who / How) — see [agents/rules/RESULTS_REPORTING.md](../../agents/rules/RESULTS_REPORTING.md).
Training-state logs and checkpoints live in `experiments/runs/`, NOT here.

## Metric files

| File | 5W1H description |
|---|---|
| `training_history.json` | **What**: per-run validation/train loss + accuracy curves of the 6 LAB2 variants. **Why**: reproduce loss-curve plots without re-training. **When**: written by `src/training/train_lab2_models.py` after each run. **Where**: combined from `experiments/runs/*/metrics/*_history.jsonl`. **Who**: LAB2 team. **How**: one entry per run: `{train_losses, val_losses, val_accs, best_epoch, best_val_acc, checkpoint}`. |
| `test_metrics.json` | **What**: test-set loss, top-1 accuracy, per-class accuracy of every evaluated variant + ensemble. **Why**: the deliverable's headline generalization numbers. **When**: after evaluation in `notebooks/practice_2.ipynb`. **Where**: written from `src/eval/evaluate_model.py` results. **Who**: LAB2 team → teacher. **How**: official 10k test split, no TTA (TTA results are separate), argmax over softmax. |
| `comparison_table.txt` | **What**: human-readable summary table of `test_metrics.json`. **Why**: quick glance for reports. **Where**: generated alongside `test_metrics.json`. |
| `sota_benchmark.json` | **What**: benchmark of the SOTA checkpoints (single models + ensemble, isolated cat/dog subset + cross-errors). **Why**: LAB2 SOTA verification. **How**: see `src/experiments/benchmark_sota.py`. |
| `catdog_confusion_reduction.json` / `..._sota.json` | **What**: results of the cat/dog confusion-reduction strategies (S1/S2/S3). **Why**: documents that post-hoc strategies did not beat the ensemble baseline. **How**: see `src/experiments/catdog_confusion_reduction.py`. |
| `feature_level_tta.json` | **What**: TTA (2-view) and top-block-unfreeze experiments on the ensemble. **Why**: measures the +0.25% TTA gain. **How**: see `src/experiments/feature_level_tta.py`. |
| `logit_bias_sweep_results.json` | **What**: 2D grid search of cat/dog logit biases (val) + zero-leakage test benchmark. **Why**: decision-threshold optimization. **Where**: `notebooks/practice_2_logit_bias_sweep.ipynb`. |
| `resnet_densenet_sota_ensemble_results.json` | **What**: ensemble evaluation metrics (isolated/cross cat-dog breakdown). |
| `stacking_mlp/` | **What**: stacking meta-model artifacts — `stacking_mlp_artifacts.npz` (val/test features + ensemble & MLP test probabilities), `mlp_{notta,tta}.pt`, `config.json`, `history.jsonl`. **Why**: notebook analysis of the 97.21% best configuration without re-training. **How**: `python -m src.experiments.stacking_mlp_train`. **Note**: `.npz`/`.pt` are regenerable by that script and are gitignored (kept out of git). |
| `moe_phase1/` | **What**: Phase-1 soft-router artifacts — `moe_phase1_artifacts.npz` (expert probabilities + gates), `router_phase1.pt`, `config.json`, `history.jsonl`. **Why**: MoE router analysis without re-training. **How**: `python -m src.experiments.moe_router_train`. **Note**: `.npz`/`.pt` are regenerable by that script and are gitignored (kept out of git). |
| `umap_features_summary.json` | **What**: UMAP clustering separability metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) comparing raw pixels against frozen, fine-tuned, SOTA models and SOTA-Ensemble penultimate features. **Why**: visual and quantitative proof of learned semantic class separation. **How**: `python src/eda/umap_cifar10_features.py --all`. |
| `umap_embeddings_*.npy` / `umap_labels_*.npy` | **What**: 2D UMAP coordinates and ground-truth labels for CIFAR-10 test set representations across models. **Why**: fast re-plotting without re-running dimensionality reduction. |

## Storage rules

- One JSON/JSONL file per **logical unit** (run history, config, experiment), never per epoch.
- Large probability/feature arrays use **compressed NPZ** (`np.savez_compressed`).
- Files > 10 MB may be gzip-archived (`file.json.gz`); keep this index current.
- Never write training state into this folder — it belongs in `experiments/runs/`.
