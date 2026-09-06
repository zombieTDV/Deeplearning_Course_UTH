# TRAINING_INFO_STATUS.md — Phase 6: Finetuning (Train Model) (agents/progress)

---

## Header

- **Title:** Finetuning with the HF Trainer (Exercise 2)
- **Created**: 2026-08-11T00:00:00+07:00
- **Last Updated**: 2026-08-13T00:00:00+07:00
- **Description:** Tracks `distilbert-base-uncased` finetuning via HF Trainer, LLRD, Cosine Scheduler, and 512 tokens expansion.
- **Status:** Done
- **Phase doc:** [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md)

## Log

- 2026-08-11: Status doc created — phase not started.
- 2026-08-12: Baseline finetuning run completed (`EXP-00`, 91.19% Test Acc). Anti-overfitting experiments executed (`EXP-01` to `EXP-05`).
- 2026-08-13: EXP-06 512 tokens sequence length expansion run completed (4 epochs, batch size 8, LLRD decay 0.9, Cosine scheduler, LR 2.5e-5). Achieved **93.23% Test Accuracy** and **0.9742 ROC-AUC**. Peak VRAM 2.14GB $\le$ 3.5GB budget ceiling. Added EXP-07 advanced hyperparameter fine-tuning preset.

## Blockers (if any)

- (none)

## Decisions

- AdamW lr=2.5e-5, LLRD decay=0.9, max_length=512, batch_size=8, grad_accum=2, fp16=True, seed=42.
- SWA parameter averaging supported across best and last checkpoints.

## Next step

- Maintained & locked for graded submission.

## Links

- Phase doc: [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md)
- Winning Config: [../../configs/config_imdb_sentiment_512.yaml](../../configs/config_imdb_sentiment_512.yaml)
