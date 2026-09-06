# FEATURE_SPLIT_STATUS.md — Phase 3: Feature Engineering & Split (agents/progress)

---

## Header

- **Title:** Feature Engineering & Split
- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-08-12T00:00:00+07:00
- **Description:** Tracks tokenization, train/val/test split, and leakage rules.
- **Status:** Done
- **Phase doc:** [../phases/FEATURE_SPLIT.md](../phases/FEATURE_SPLIT.md)

## Log

- 2026-08-11: status doc created — phase not started.
- 2026-08-12: `src/data/prepare_imdb.py` implemented and verified; tokenized splits cached to `data/processed/imdb_tokenized`.

## Blockers (if any)

- (none)

## Decisions

- static tokenizer `distilbert-base-uncased`, max_length=256, val_size=0.1.

## Next step

- Proceed to Phase 6 full training run.

## Links

- Phase doc: [../phases/FEATURE_SPLIT.md](../phases/FEATURE_SPLIT.md)
