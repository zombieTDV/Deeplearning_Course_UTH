# REPORT_STATUS.md — Phase 8: Error Analysis, Interpretability & Reporting (agents/progress)

---

## Header

- **Title:** Error Analysis, Interpretability & Reporting
- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-08-15T00:00:00+07:00
- **Description:** Tracks error analysis, interpretability, interactive demo notebooks, and 5W1H experiment reports.
- **Status:** Done
- **Phase doc:** [../phases/REPORT.md](../phases/REPORT.md)

## Log

- 2026-08-11: Status doc created — phase not started.
- 2026-08-12: Demo notebook `notebooks/02_ex2_finetune.ipynb` generated. Final Ex 2 report documented in `agents/experiments/EX2_IMDB_FINETUNE.md`.
- 2026-08-13: Refactored notebooks into a clean 2-notebook suite (`notebooks/01_ex1_sentiment_baseline.ipynb` for Zero-Shot Baseline & `notebooks/02_ex2_finetune.ipynb` for Fine-Tuning presets `EXP-00` to `EXP-07`). Documented EXP-06 breakthrough milestone (**93.23% Test Accuracy**, **0.9323 Macro F1**, **0.9742 ROC-AUC**) in `agents/experiments/EX6_512_TOKENS_BREAKTHROUGH_REPORT.md`.
- 2026-08-15: Completed EX-07 (HTML `<br />` artifact stripping) and EX-08 Data-Centric AI (Cleanlab Confident Learning).
- 2026-08-15: Completed EX-09 (LoRA Rank 32 + Multi-Tier Regularization), EX-10..EX-12 (LoRA learning rate sensitivity diagnostics & post-mortem).
- 2026-08-15: Completed EX-13 (Head + Tail Truncation and Independent 5-Fold OOF Cleanlab Checkpoint Selection Architecture). Synchronized all 11 cells in `notebooks/02_ex2_finetune.ipynb` with clickable markdown links.
- 2026-08-15: Completed EX-14 (LoRA PEFT on Denoised IMDB with Head-Tail Truncation achieving **93.14% Test Accuracy**, **0.9314 Macro F1**, **0.9662 ROC-AUC** under 1.11 GB peak VRAM).

## Decisions

- Notebooks serve as clean high-level dashboards calling modular `src` classes (`IMDBDatasetEDA`, `IMDBCleanlabAuditor`, `IMDBTrainer`, `IMDBEvaluator`, `IMDBPlotter`, `SentimentPredictor`, `ErrorAuditor`, `visualize_head_tail_truncation`).
- Comparative analysis demonstrates **+4.16% accuracy gain** (93.23% vs 89.07% zero-shot baseline) and **-511 misclassification reduction**.
- Data-Centric AI 5-Fold OOF Confident Learning provides 100% zero-memorization guarantee for clean training split creation.
- LoRA PEFT ($r=32, \alpha=64$) on Denoised IMDB achieves full fine-tuning parity ($93.14\%$ vs $93.23\%$) with $38\times$ parameter reduction and $52\%$ VRAM reduction.

## Links

- Phase doc: [../phases/REPORT.md](../phases/REPORT.md)
- Breakthrough Report: [../experiments/EX6_512_TOKENS_BREAKTHROUGH_REPORT.md](../experiments/EX6_512_TOKENS_BREAKTHROUGH_REPORT.md)
- HTML Cleaning Report: [../experiments/EX7_HTML_DATA_CLEANING_REPORT.md](../experiments/EX7_HTML_DATA_CLEANING_REPORT.md)
- Cleanlab Denoising Report: [../experiments/EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md](../experiments/EX8_CLEANLAB_DATA_CENTRIC_DENOISING_REPORT.md)
- Head + Tail & OOF Report: [../experiments/EX13_HEAD_TAIL_TRUNCATION_REPORT.md](../experiments/EX13_HEAD_TAIL_TRUNCATION_REPORT.md)
- LoRA Denoised Breakthrough: [../experiments/EX14_LORA_DENOISED_HEAD_TAIL_BREAKTHROUGH_REPORT.md](../experiments/EX14_LORA_DENOISED_HEAD_TAIL_BREAKTHROUGH_REPORT.md)
