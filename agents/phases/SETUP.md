# SETUP.md — Phase 1: Setup & Problem Framing (agents/phases)

---

## Header

- **Title:** Setup & Problem Framing
- **Execution order:** 1 of 8
- **Date created:** 2026-08-11
- **Last updated:** 2026-08-11
- **Description:** Install the Hugging Face stack (`transformers`, `datasets`, `evaluate`), verify GPU/VRAM limits, and lock the problem framing for Practice 3.
- **Status:** Done

## Background

Exercise 1 step 1 requires installing the Hugging Face `transformers` library, and Exercise 2 adds `datasets` and `evaluate`. The repo's `requirements.txt` currently has no HF dependencies, so the first step is to add and verify them inside the existing `.venv`. The project constraint of multi-GPU team hardware (8 GB and 4 GB machines, target ≤3.5 GB VRAM) must be verified up front so later phases (batch size, model choice) stay within budget. This phase covers roadmap tasks T1–T2 and milestone M1.

## Goals / Purpose

- What "done" looks like, concretely:
  - HF stack imports cleanly inside `.venv`.
  - GPU/VRAM reported (≤3.5 GB target, 4 GB ceiling).
  - Problem framing locked (supervised binary sentiment classification; Ex 1 = zero-shot baseline, Ex 2 = finetune).
  - This phase doc written and reviewed.
- What this phase explicitly does NOT try to solve:
  - No dataset download, no model choice, no training code.

## Input / Output

- **Input:** `requirements.txt`, `.venv`, GPU(s) (8 GB + 4 GB team machines).
- **Output:** updated `requirements.txt` with pinned HF versions; verified environment; VRAM report; this doc.

## How to do it (general plan)

1. Activate `.venv`; add `transformers`, `datasets`, `evaluate` to `requirements.txt` with pinned versions (task T1).
2. Install and verify imports: `python -c "import transformers, datasets, evaluate"`.
3. Run a GPU/VRAM check (e.g. `nvidia-smi`); confirm usable VRAM and that `torch.cuda.is_available()` is true.
4. Lock the problem framing against [PURPOSE.md](../PURPOSE.md) (task T2).
5. Write and review this phase doc.

## Pipeline

```
.venv activation → pip install -r requirements.txt
  → import smoke (transformers, datasets, evaluate)
  → nvidia-smi VRAM check
  → PURPOSE.md framing locked → SETUP.md approved
```

## Detailed plan / gotchas

- Pin HF versions in `requirements.txt` at T1 to avoid API drift in later phases (roadmap risk R3).
- Ensure the installed `torch` build matches the CUDA driver.
- Target VRAM ≤3.5 GB: this constrains batch size and `max_length` in [FEATURE_SPLIT.md](FEATURE_SPLIT.md) and [TRAINING_INFO.md](TRAINING_INFO.md) (roadmap risk R1).
- File and run naming follows [NAMING_CONVENTION.md](../rules/NAMING_CONVENTION.md).

## Links

- Roadmap: [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) (§4 Phase 1, §5 T1–T2)
- Progress tracking: [../progress/SETUP_STATUS.md](../progress/SETUP_STATUS.md)
- Related phases: [DATA_PREP.md](DATA_PREP.md), [BASELINE.md](BASELINE.md)
- Rules: [LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md)
