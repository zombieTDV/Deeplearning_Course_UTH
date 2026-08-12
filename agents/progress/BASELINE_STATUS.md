# BASELINE_STATUS.md — Phase 4: Baseline Model (agents/progress)

---

## Header

- **Title:** Baseline Model (Exercise 1 — Zero-Shot Sentiment Analysis)
- **Date created:** 2026-08-11
- **Last updated:** 2026-08-11
- **Description:** Tracks the Ex 1 zero-shot baseline and majority-class floor.
- **Status:** Done
- **Phase doc:** [../phases/BASELINE.md](../phases/BASELINE.md)

## Log

- 2026-08-11: status doc created — phase not started.
- 2026-08-12: created branch plan `agents/experiments/EX1_SENTIMENT_BASELINE.md` for `feature/ex1-sentiment-baseline`.
- 2026-08-12: implemented `src/experiments/baseline_imdb_sentiment.py` using [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) on [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb). Achieved 89.07% accuracy vs 50.00% majority baseline floor.

## Blockers (if any)

- (none)

## Decisions

- Implement zero-shot evaluation CLI in `src/experiments/baseline_imdb_sentiment.py`.

## Next step

- Proceed to Phase 5 / Phase 6 (Exercise 2: Finetune [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) on IMDB dataset).

## Links

- Phase doc: [../phases/BASELINE.md](../phases/BASELINE.md)
