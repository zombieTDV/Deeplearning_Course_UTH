# experiments/results — Metrics Index & Descriptions

Every file below is described with the 5W1H principle (What / Why / When /
Where / Who / How) — see
[agents/rules/RESULTS_REPORTING.md](../../agents/rules/RESULTS_REPORTING.md).
Training-state logs and checkpoints live in `experiments/runs/`, NOT here.

## Metric files

| File | 5W1H description |
|---|---|
| [`baseline_imdb_sentiment.json`](baseline_imdb_sentiment.json) | **What**: zero-shot baseline metrics on the IMDB test split (25,000 samples): accuracy 89.07%, majority-class floor 50.00%, ROC-AUC 0.9587. **Why**: reference floor for Exercise 2 finetuning deltas. **When**: 2026-08-12, by [`src/experiments/baseline_imdb_sentiment.py`](../../src/experiments/baseline_imdb_sentiment.py). **Where**: this file + [`experiments/plots/baseline_zero_shot_roc_curve.png`](../plots/baseline_zero_shot_roc_curve.png). **Who**: bush-le + AI agent → coursework. **How**: HF pipeline `distilbert-base-uncased-finetuned-sst-2-english`, GPU, seed 42, no training; test evaluated once. |
| [`imdb_dataset_eda.json`](imdb_dataset_eda.json) | **What**: IMDB EDA statistics — per-split counts, label balance (50/50), review-length word-count stats. **Why**: documents dataset geometry before preprocessing (pipeline §3). **When**: 2026-08-12, by [`src/data/eda_imdb.py`](../../src/data/eda_imdb.py). **Where**: this file + [`experiments/plots/imdb_label_distribution.png`](../plots/imdb_label_distribution.png), [`experiments/plots/imdb_review_length_distribution.png`](../plots/imdb_review_length_distribution.png). **Who**: bush-le → coursework. **How**: `datasets.load_dataset("stanfordnlp/imdb")`; no test-set statistics used in training. |
| [`imdb_sentiment_eval.json`](imdb_sentiment_eval.json) | **What**: Finetuned DistilBERT test evaluation — accuracy 90.70%, ROC-AUC 0.9653, F1 macro 0.9069, peak VRAM 264.62 MB (target ≤3.5 GB). **Why**: Exercise 2 generalization benchmark on sealed 25k IMDB test split vs zero-shot baseline (89.07%). **When**: 2026-08-12, by [`src/eval/evaluate_model.py`](../../src/eval/evaluate_model.py). **Where**: this file + [`experiments/plots/imdb_finetuned_confusion_matrix.png`](../plots/imdb_finetuned_confusion_matrix.png), [`experiments/plots/imdb_finetuned_roc_curve.png`](../plots/imdb_finetuned_roc_curve.png). **Who**: bush-le + AI agent → coursework. **How**: single held-out evaluation pass using softmax probabilities on `distilbert-finetune_best.pt` checkpoint. |


## Storage rules

- One JSON/JSONL file per **logical unit** (run history, config, experiment), never per epoch.
- Large probability/feature arrays use **compressed NPZ** (`np.savez_compressed`).
- Files > 10 MB may be gzip-archived (`file.json.gz`); keep this index current.
- Never write training state into this folder — it belongs in `experiments/runs/`.
