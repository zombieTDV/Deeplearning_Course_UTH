# src/utils — Shared Helpers

Cross-cutting utilities used by the other layers:

- `run_logger.py` — real-time logging + rotating files + JSONL history.
- `checkpoint_utils.py` — safe loading (`weights_only=True`), best/last
  checkpoint discovery, run registry lookups.
- Other small helpers as needed (device selection, seeds, path helpers).

Do not put domain logic here — it belongs in the layer that owns it.

> **Status:** `run_logger.py` and `checkpoint_utils.py` are **planned**
> (roadmap Phases 5–6). This file is the **canonical owner** of both — other
> layers consume, never re-implement (audit findings `AQ-1` / `ARC-1`, see
> [agents/CODEBASE_AUDIT_REPORT.md](../../agents/CODEBASE_AUDIT_REPORT.md)).
