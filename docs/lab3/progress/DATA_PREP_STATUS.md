# DATA_PREP_STATUS.md — Phase 2: Data Survey & Cleaning (agents/progress)

---

## Header

- **Title:** Data Survey, Preprocessing & Data-Centric AI Cleaning
- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-08-15T00:00:00+07:00
- **Description:** Tracks IMDB EDA, HTML noise stripping, Head+Tail Truncation, and Cleanlab Confident Learning.
- **Status:** Done
- **Phase doc:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)

## Log

- 2026-08-11: Status doc created — phase not started.
- 2026-08-12: Implemented `src/data/eda_imdb.py` to analyze [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb). Generated class balance plot (`experiments/plots/imdb_label_distribution.png`), length distribution plot (`experiments/plots/imdb_review_length_distribution.png`), and persisted statistics JSON (`experiments/results/imdb_dataset_eda.json`).
- 2026-08-15: Completed EX-07 HTML tag cleaning (`clean_text()` in `src/data/prepare_imdb.py`), stripping `<br />` artifacts and reclaiming 18.4 wasted tokens per review.
- 2026-08-15: Completed EX-13 Head + Tail Truncation (`head_tail_tokenize()`), preserving 128 head tokens (premise) + 384 tail tokens (verdict) to eliminate verdict truncation on reviews > 512 tokens.
- 2026-08-15: Integrated Cleanlab Confident Learning with 5-Fold Out-Of-Fold (OOF) cross-validation in `src/data/cleanlab_denoiser.py` and exported pristine dataset to `data/processed/imdb_denoised_512` (22,388 clean train samples, 2,500 val, 25,000 test).

## Decisions

- Retain 50/50 balanced splits.
- Preprocess texts with `clean_text()` and tokenize via `head_tail_tokenize(max_length=512, head_ratio=0.25)`.
- Use independent 5-Fold OOF Audit Model for Cleanlab label error filtering to prevent data leakage.

## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- EX-07 Report: [../experiments/EX7_HTML_DATA_CLEANING_REPORT.md](../experiments/EX7_HTML_DATA_CLEANING_REPORT.md)
- EX-08 Report: [../experiments/EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md](../experiments/EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md)
- EX-13 Report: [../experiments/EX13_HEAD_TAIL_TRUNCATION_REPORT.md](../experiments/EX13_HEAD_TAIL_TRUNCATION_REPORT.md)
