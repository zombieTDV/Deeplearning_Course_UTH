# src/models — Model Layer

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---


What belongs here:

- `build_model.py` — model builders (e.g. torchvision pretrained models,
  custom architectures), each exposing a `mode` argument for
  frozen / finetune / full-training variants.
- Freeze/unfreeze helpers, layer-wise LR param groups.

## Rules

- Builders are pure constructors: no training logic, no file I/O beyond
  loading pretrained weights.
- Parameter groups for layer-wise LR decay (LLRD) belong here, not in the
  training loop.

See [agents/phases/MODEL.md](../../agents/phases/MODEL.md).

> **Status:** planned (roadmap Phase 5); `build_model.py` not implemented yet —
> audit finding `ARC-1`
> ([agents/CODEBASE_AUDIT_REPORT.md](../../agents/CODEBASE_AUDIT_REPORT.md)).
