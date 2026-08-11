# FEATURE_SPLIT.md — Phase 3: Feature Engineering & Split (agents/phases)

---

## Header

- **Title:** Feature Engineering & Split
- **Execution order:** 3 of 8
- **Date created:** 2026-08-11
- **Last updated:** 2026-08-11
- **Description:** Tokenize IMDB with the `distilbert-base-uncased` tokenizer (the DL-equivalent of feature engineering) and set up train/val/test splits with the leakage golden rules applied.
- **Status:** To Do

## Background

Pipeline stages §6–§7 and §9–§10 of [ML_PIPELINE_REFERENCE_v3.md](../ML_PIPELINE_REFERENCE_v3.md): for pretrained NLP, tokenization replaces classical feature engineering (scaling/encoding are N/A), and the split must happen with leakage rules (split first; test sealed; no statistics fit on test). The tokenizer is static (pretrained, not fit on the data), so it does not leak. This phase covers roadmap tasks T5–T6 and milestone M3.

## Goals / Purpose

- What "done" looks like, concretely:
  - Tokenized dataset (`input_ids`, `attention_mask`) for train, validation, and test.
  - Validation split carved from train for the `Trainer`.
  - Leakage golden rules documented in this doc.
  - Preprocessing wrapped in `src/data/prepare_imdb.py` so training consumes the same tokenized data.
- What this phase explicitly does NOT try to solve:
  - No model selection rationale (Phase 5), no training loop (Phase 6).

## Input / Output

- **Input:** raw IMDB from Phase 2; `distilbert-base-uncased` tokenizer (from the HF Hub, cached).
- **Output:** tokenized `datasets.Dataset` (train/val/test, torch format); `max_length` decision; `src/data/prepare_imdb.py`; this doc.

## How to do it (general plan)

1. Load the tokenizer: `AutoTokenizer.from_pretrained("distilbert-base-uncased")` — static, no data fitting.
2. Map the dataset with `padding`/`truncation` and `max_length` (e.g. 256); keep `text`/`label`.
3. Carve a validation split from train (e.g. `train_test_split(test_size=0.1, seed=42)` or the Trainer's built-in eval split).
4. Verify split sizes and that no test statistics are used anywhere in preprocessing (§10 golden rules).
5. Set `dataset.set_format("torch")` for Trainer consumption.
6. Document the tokenization-as-FE rationale and the leakage rules in this doc.

## Pipeline

```
src/data/prepare_imdb.py: load_dataset("imdb")
  → AutoTokenizer("distilbert-base-uncased").map(pad/trunc, max_length=256)
  → train/val split → tokenized Dataset (torch format)
  → consumed by src/training/imdb_sentiment_train.py (Phase 6)
```

## Detailed plan / gotchas

- `max_length` trades VRAM against signal: 256 tokens covers the vast majority of IMDB reviews; use 128 if VRAM is tight (roadmap risk R1).
- Never recompute tokenizer or split statistics on the test set — it is evaluated exactly once at the end (golden rule 4, [§10](../ML_PIPELINE_REFERENCE_v3.md#10-train--test-split-and-data-leakage)).
- Keep the label names aligned with the model's `id2label`/`label2id` mapping (roadmap risk R6).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 3, §5 T5–T6)
- Progress tracking: [../progress/FEATURE_SPLIT_STATUS.md](../progress/FEATURE_SPLIT_STATUS.md)
- Related phases: [DATA_PREP.md](DATA_PREP.md), [MODEL.md](MODEL.md), [TRAINING_INFO.md](TRAINING_INFO.md)
- Reference: [ML_PIPELINE_REFERENCE_v3.md §10](../ML_PIPELINE_REFERENCE_v3.md#10-train--test-split-and-data-leakage)
