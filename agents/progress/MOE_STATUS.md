# MOE_STATUS.md

**Phase / area:** Mixture of Experts (MoE) on the SOTA ensemble — learned
per-input gating between ResNet18-sota and DenseNet121-sota.
**Last updated:** 2026-08-05

Status legend: `TODO` · `IN PROGRESS` · `DONE` · `ON HOLD` · `BLOCKED`

## Status: IN PROGRESS (Phase 1 implemented & measured; Phase 2 gated)

## Log

- 2026-08-05: Wrote MoE plan + `notebooks/practice_2_moe.ipynb`
  (`agents/experiments/MOE_EXPERIMENT.md`, `agents/phases/MOE.md`).
- 2026-08-05: **Phase 1 run** (frozen experts + soft router, trained on val).
  - Router val acc: 96.96%.
  - Test: MoE router (no TTA) **96.53%** | +TTA **96.89%**.
  - vs baselines: soft-voting 96.87% | +TTA 97.12%. → **Phase 1 UNDERPERFORMS
    fixed soft-voting.**
- 2026-08-05: **Phase 2** implemented but gated off (`RUN_PHASE2=False`) — not
  run yet (heavy, and experts are already strong / earlier top-block fine-tune
  degraded them).

## Key decisions

1. Use **soft gating** (default) over top-k hard routing — hard routing discards
   ensemble-averaging benefit.
2. Train router on **validation** split, evaluate on **test** (no test leakage).
3. Phase 1 negative result → **strengthen soft gating** before spending compute
   on Phase 2: raise load-balance `alpha`, add gate entropy regularization,
   and/or put a temperature on router logits to stop near-top-1 collapse.

## Next steps

- [ ] Fix Phase 1 soft gating (alpha / entropy / router-logit temperature).
- [ ] Re-measure: goal ≥ 96.87% (no TTA) and ≥ 97.12% (+TTA), ideally beating
      MLP-stacking 97.21%.
- [ ] Fix final-summary-cell display bug (the "+TTA" row shows the no-TTA number).
- [ ] Run Phase 2 only if Phase 1 soft gating is competitive.

## Related
- Notebook: `notebooks/practice_2_moe.ipynb`
- Experiment record: `agents/experiments/MOE_EXPERIMENT.md`
- Phase doc: `agents/phases/MOE.md`
