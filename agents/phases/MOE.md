# MOE.md

**Phase / area:** Mixture of Experts (MoE) on the CIFAR-10 SOTA ensemble.
**Last updated:** 2026-08-05

## Purpose

Add a learned **router/gating network** to combine the two SOTA experts
(ResNet18-sota + DenseNet121-sota) per input, instead of a fixed 0.5/0.5
soft-voting rule. Intended to squeeze a small gain on top of the current best
(96.87% → 97.12% +TTA → 97.21% +MLP-stacking).

## Formulation

$$p(x) = \sum_{i} g_i(x)\, p_i(x), \qquad g(x) = \operatorname{softmax}(\operatorname{Router}([p_1(x), p_2(x)]))$$

- **Experts** = whole pretrained models (ResNet18-sota, DenseNet121-sota).
- **Router** = small MLP `20 -> 64 -> ReLU -> 2`.
- **Input** = concatenated expert softmax (20-dim).
- **Gating** = soft (default). Top-k hard routing optional but discouraged (it
  discards ensemble-averaging robustness).

## Phases

- **Phase 1 — frozen experts + soft router (IN PROGRESS).** Router trained on
  the validation split; experts frozen. Cheap.
- **Phase 2 — joint fine-tune (gated).** Unfreeze expert top blocks
  (`layer4+fc`, `denseblock4+norm5+classifier`) and train router + blocks on the
  train split with LLRD + load-balance loss. Heavy.

## Training protocol (leakage-aware)

- Router trained on **validation** (5k); evaluated on **test** (10k).
- Optionally hold out a small meta-val for router early stopping (experts used
  the val split during their own early stopping).
- Always report **test** as the source of truth.

## Results so far (Phase 1)

- Router val acc 96.96%; test 96.53% (no TTA) / 96.89% (+TTA) — **below**
  fixed soft-voting (96.87% / 97.12%).
- Cause: router collapses to near-hard per-class routing, losing the
  ensemble-averaging benefit.

## Open issues / to fix

1. **Soft-gating collapse** → stronger load-balance loss + gate entropy
   regularizer + router-logit temperature.
2. **Summary-cell display bug** in the notebook final table (+TTA row shows the
   no-TTA baseline).
3. **Phase 2** not run; likely marginal given experts are already strong.

## Known risks

- **Router collapse** (all inputs → one expert): mitigated by load-balance loss
  + entropy reg.
- **Diminishing returns** at 2 experts / 10 classes / ~97%: MoE's big wins
  (huge scale, sparsity) do not apply here; realistic gain is small.
- **Joint fine-tune can degrade strong experts** (observed in earlier
  feature-level experiments).
- **Hard/top-1 routing can hurt** accuracy by discarding averaging.
- **Local GPU / slow training**: keep experts frozen in Phase 1; cache base
  probabilities so the router trains cheaply.

## Related
- Notebook: `notebooks/practice_2_moe.ipynb`
- Status: `agents/progress/MOE_STATUS.md`
- Experiment record: `agents/experiments/MOE_EXPERIMENT.md`
