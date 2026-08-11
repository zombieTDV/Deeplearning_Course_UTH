# DATA_PREP.md — Phase 2: Data Survey & Cleaning (agents/phases)

---

## Header

- **Title:** Data Survey & Cleaning
- **Execution order:** 2 of 8
- **Date created:** 2026-08-11
- **Last updated:** 2026-08-11
- **Description:** Run EDA on IMDB (label balance, review lengths, sample inspection) and document why the cleaning and class-imbalance pipeline stages are N/A for this dataset.
- **Status:** To Do

## Background

Pipeline stages §3–§5 and §8 of [ML_PIPELINE_REFERENCE_v3.md](../ML_PIPELINE_REFERENCE_v3.md): EDA must come first (Rule #0 — start from the data), and cleaning/imbalance handling must be explicitly checked rather than assumed. IMDB is pre-cleaned (no missing values or outliers to treat in the classical sense) and is balanced 50/50 by construction — but this must be verified and documented, not silently skipped. This phase covers roadmap tasks T3–T4.

## Goals / Purpose

- What "done" looks like, concretely:
  - EDA statistics recorded: per-split sample counts, label distribution, review-length stats, representative samples.
  - Cleaning (§4–§5) and class-imbalance (§8) stages confirmed N/A with written rationale.
  - This phase doc updated with the recorded numbers.
- What this phase explicitly does NOT try to solve:
  - No tokenization or splitting (Phase 3), no model choice (Phase 5).

## Input / Output

- **Input:** IMDB dataset from Hugging Face `datasets` (`load_dataset("imdb")`); dataset card.
- **Output:** EDA statistics; this doc; raw dataset cached under `data/external/` or the HF cache.

## How to do it (general plan)

1. Load IMDB: `datasets.load_dataset("imdb")`; record train/test sample counts.
2. Verify label distribution per split (expect ~50/50 positive/negative) — §8 balance check.
3. Compute review-length stats (characters/tokens: mean, median, percentiles) — §3 univariate survey.
4. Inspect representative samples from each class (qualitative sanity check) — §3.2.
5. Document why missing-value handling (§4) and outlier treatment (§5) are N/A (pre-cleaned text data).
6. Record everything in this doc; hand the raw dataset to Phase 3.

## Pipeline

```
load_dataset("imdb") → pandas/numpy stats (counts, label balance, length stats)
  → sample inspection → DATA_PREP.md (EDA + N/A rationale)
  → raw IMDB → FEATURE_SPLIT.md (Phase 3)
```

## Detailed plan / gotchas

- The IMDB test set is sealed until final evaluation (leakage golden rule, [§10](../ML_PIPELINE_REFERENCE_v3.md#10-train--test-split-and-data-leakage)) — never compute anything from it that informs training.
- Download is network-dependent (~80 MB): use the HF cache and set `HF_ENDPOINT` if blocked (roadmap risk R2).
- EDA feeds model selection (T8) and preprocessing decisions (T5–T6).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 2, §5 T3–T4)
- Progress tracking: [../progress/DATA_PREP_STATUS.md](../progress/DATA_PREP_STATUS.md)
- Related phases: [SETUP.md](SETUP.md), [FEATURE_SPLIT.md](FEATURE_SPLIT.md)
- Reference: [ML_PIPELINE_REFERENCE_v3.md §3–§8](../ML_PIPELINE_REFERENCE_v3.md)
