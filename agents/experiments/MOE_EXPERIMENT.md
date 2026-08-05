# MOE_EXPERIMENT.md

**Experiment:** Mixture of Experts (learned router) on the CIFAR-10 SOTA
ensemble (ResNet18-sota + DenseNet121-sota).
**Last updated:** 2026-08-05

## Objective

Beat the fixed `0.5·P_ResNet + 0.5·P_DenseNet` soft-voting baseline
(96.87% test; 97.12% with TTA; MLP-stacking 97.21%) by learning a per-input
gate over the two frozen experts:

$$p(x) = g_1(x)\,p_1(x) + g_2(x)\,p_2(x), \qquad g(x) = \mathrm{softmax}(\mathrm{Router}([p_1,p_2]))$$

## Setup

- Experts: `ResNet18-sota_best.pt`, `DenseNet121-sota_best.pt` (loaded from
  `experiments/checkpoints`).
- Router: `MLP(20 -> 64 -> ReLU -> 2)`; input = concatenated expert softmax (20-dim).
- Router trained on the **validation** split (5k); evaluated on **test** (10k).
- Loss: CE on the gated mixture + `alpha * load_balance(gates)` (`alpha=0.01`).

## Phase 1 — frozen experts + soft router (DONE, measured)

| Method | Full acc | Isolated | Cross-conf |
|---|---|---|---|
| Soft-voting (no TTA) | 96.87% | 93.45% | 86 |
| **Ensemble + MoE router (no TTA)** | **96.53%** | 92.80% | 87 |
| Soft-voting + TTA | 97.12% | 94.00% | 76 |
| **Ensemble + MoE router + TTA** | **96.89%** | 93.75% | 79 |

Router val acc: 96.96%. **Result: Phase 1 MoE router underperforms fixed
soft-voting** (by -0.34% no TTA, -0.23% with TTA).

### Root-cause analysis (from routing stats)
The router collapsed to **near-hard per-class assignment**:

| Class | g_E1 (ResNet) | g_E2 (DenseNet) |
|---|---|---|
| airplane | 0.79 | 0.21 |
| automobile | 0.01 | 0.99 |
| cat | 0.02 | 0.98 |
| deer | 0.01 | 0.99 |
| dog | 0.61 | 0.39 |
| frog | 0.00 | 1.00 |
| ship | 0.07 | 0.93 |
| truck | 0.89 | 0.11 |

The router effectively routes each class to its single stronger expert,
**discarding the ensemble-averaging robustness** that made soft-voting strong.
The `alpha=0.01` load-balancing loss was too weak to keep gates soft.

## Phase 2 — joint fine-tune (gated off, NOT run)

Unfreezes `layer4+fc` (ResNet) and `denseblock4+norm5+classifier` (DenseNet),
trains router + those blocks on the train split (LLRD) with the load-balance
loss. Implemented behind `RUN_PHASE2=False`. Not run — experts already strong,
and earlier feature-level top-block fine-tune degraded them.

## Next experiment variants (to make MoE beat soft-voting)

1. **Stronger soft gating:** raise `alpha` (0.1), add gate entropy regularizer,
   and/or temperature on router logits (divide by `T~2`).
2. **Top-2 routing:** always combine both experts; only learn the blend.
3. Re-measure; target ≥ 96.87% (no TTA) / ≥ 97.12% (+TTA), ideally > 97.21%.

## How to run
- Phase 1: run `notebooks/practice_2_moe.ipynb` (all cells; Phase 2 auto-skipped).
- Phase 2: set `RUN_PHASE2 = True` (optionally `PHASE2_SUBSET` for a smoke run)
  in cell 10, then run. Runtime > 5 min — run manually, not via automation.
