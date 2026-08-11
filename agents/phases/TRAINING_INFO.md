# TRAINING_INFO.md — Phase 6: Finetuning (Train Model) (agents/phases)

---

## Header

- **Title:** Finetuning with the HF Trainer (Exercise 2)
- **Execution order:** 6 of 8
- **Date created:** 2026-08-11
- **Last updated:** 2026-08-11
- **Description:** Implement the script-only finetuning CLI using the HF `Trainer` with full-state checkpoints, resume support, and logging per repo rules; smoke-test, then run the full finetune on GPU.
- **Status:** To Do

## Background

Exercise 2 steps 5–7 (define training arguments, create a Trainer, finetune) plus the repo's script-only rule: training runs only from `src/training/*.py`, and every run persists checkpoints, logs, config, and history per [LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md). This phase covers roadmap tasks T10–T12 and milestone M4.

## Goals / Purpose

- What "done" looks like, concretely:
  - `src/training/imdb_sentiment_train.py` CLI with `--epochs`, `--seed`, `--resume`, `--force-resume`, `--tb`, `--smoke`.
  - Smoke test passes on a tiny subset (VRAM verified ≤6 GB).
  - Full run persisted under `experiments/runs/<ts>_<run>/` with checkpoints, logs, config, history JSONL.
- What this phase explicitly does NOT try to solve:
  - No hyperparameter sweeps (out of scope), no evaluation reporting (Phase 7).

## Input / Output

- **Input:** tokenized splits (Phase 3), config (Phase 5).
- **Output:** run directory `experiments/runs/<ts>_<run>/` with `checkpoints/<run>_best.pt`, `checkpoints/<run>_last.pt`, `logs/<run>.log`, `metrics/<run>_config.json`, `metrics/<run>_history.jsonl`, optional `tensorboard/`.

## How to do it (general plan)

1. Implement the CLI entry point wrapping `Trainer` + `TrainingArguments` (Exercise 2 steps 5–6).
2. Integrate full-state checkpoints: model, optimizer, scheduler, RNG, history, best metrics, `early_stop_triggered`, config, timestamp, commit (see §3 of the logging rules).
3. Write `_last.pt` after every epoch (resume source) and `_best.pt` on the best tracked metric.
4. Run the smoke test on a tiny subset per [SMOKE_TEST_CHECKLIST.md](../templates/SMOKE_TEST_CHECKLIST.md); verify VRAM ≤6 GB.
5. Launch the full run on GPU (produces the Exercise 2 finetuned model).

## Pipeline

```
python -m src.training.imdb_sentiment_train --epochs 3 --seed 42 --tb
  → Trainer finetune (distilbert-base-uncased, IMDB)
  → experiments/runs/<ts>_<run>/ {checkpoints/, logs/, metrics/, tensorboard/}

python -m src.training.imdb_sentiment_train --resume     # exact resume from _last.pt
python -m src.training.imdb_sentiment_train --force-resume  # rewind from _best.pt
```

## Detailed plan / gotchas

- Resume loads `_last.pt` (model → optimizer → scheduler → RNG → history → best metrics) and continues at `epoch + 1`; `_best.pt` is used only for `--force-resume` rewind ([§5 of the logging rules](../rules/LOGGING_CHECKPOINT_RULES.md#5-resume-procedure)).
- If the last checkpoint has `early_stop_triggered`, plain `--resume` halts; use `--force-resume` to rewind from the best epoch.
- Checkpoints must load with `weights_only=True` (§3 of the logging rules).
- fp16 + gradient accumulation keep VRAM ≤6 GB (roadmap risk R1).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 6, §5 T10–T12)
- Progress tracking: [../progress/TRAINING_INFO_STATUS.md](../progress/TRAINING_INFO_STATUS.md)
- Related phases: [MODEL.md](MODEL.md), [EVAL.md](EVAL.md)
- Rules: [LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md), [SMOKE_TEST_CHECKLIST.md](../templates/SMOKE_TEST_CHECKLIST.md)
