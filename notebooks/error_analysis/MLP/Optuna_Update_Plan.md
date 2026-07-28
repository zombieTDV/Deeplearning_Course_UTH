# Optuna Update Plan — MLP Hyperparameter Optimization

- **Motivation/Background**: Phase 4 Optuna search (30 trials) completed with 19/30 pruned and 11/30 complete; the search space contained dead parameter zones that wasted TPE budget, and dashboard analysis shows dropout + batch_norm dominate final validation accuracy. A refined search space is needed to improve trial efficiency and convergence.
- **Purpose**: Define a reduced, evidence-driven search space for the next MLP Optuna run, eliminating dead zones, fixing low-importance HPs to constants, and concentrating TPE budget on the 2–3 parameters that matter most.
- **Overview Pipeline**: The Optuna DB was analyzed via FANOVA importance, categorical dead-zone detection, and best-trial pattern clustering. Results inform a new search space proposal, removal candidates, and implementation steps for the next run.
- **Detailed Plan**: Section 1 — Current Search Space Audit (HPs, ranges, dead zones); Section 2 — Insights from Optuna DB Analysis (importance rankings, best-trial convergence, trial #17 anomaly); Section 3 — Proposed Search Space Refinements (narrow, fix-to-constant, drop); Section 4 — Implementation Plan (code diff, n_trials, pruner tuning); Section 5 — Expected Outcomes and Validation.
- **References**: Optuna 4.9.0 (SQLite storage, TPESampler, MedianPruner, FANOVA), PyTorch 2.x, torchvision (Fashion-MNIST), scikit-learn, `notebooks/error_analysis/MLP/phase4_mlp_optuna_merged.ipynb`, `outputs/error_analysis/MLP/phase4_optuna/optuna_study.db`.

---

## Table of Contents

- [Current Search Space Audit](#current-search-space-audit)
  - [Current Search Space (Phase 4)](#current-search-space-phase-4)
  - [Dead Parameter Zones](#dead-parameter-zones)
- [Insights from Optuna DB Analysis](#insights-from-optuna-db-analysis)
  - [Hyperparameter Importance Rankings](#hyperparameter-importance-rankings)
  - [Top Reliable Trial Patterns](#top-reliable-trial-patterns)
  - [Trial #17 Duration Anomaly](#trial-17-duration-anomaly)
- [Proposed Search Space Refinements](#proposed-search-space-refinements)
  - [Parameters to Drop](#parameters-to-drop)
  - [Parameters to Fix as Constants](#parameters-to-fix-as-constants)
  - [Parameters to Narrow](#parameters-to-narrow)
  - [Parameters to Keep Full Range](#parameters-to-keep-full-range)
  - [Proposed Search Space Summary](#proposed-search-space-summary)
- [Implementation Plan](#implementation-plan)
  - [Objective Function Changes](#objective-function-changes)
  - [Pruner and Study Config Tuning](#pruner-and-study-config-tuning)
  - [Expected Trial Budget](#expected-trial-budget)
- [Expected Outcomes and Validation](#expected-outcomes-and-validation)
  - [Expected Improvements](#expected-improvements)
  - [Validation Strategy](#validation-strategy)
  - [Risks and Mitigations](#risks-and-mitigations)

---

## Current Search Space Audit

### Current Search Space (Phase 4)

| HP | Type | Range | Conditional |
|----|------|-------|-------------|
| `n_layers` | int | [1, 4] | No |
| `units_i` | int (log) | 64–1024 (≤3 layers), 64–512 (4 layers) | On `n_layers` |
| `activation` | categorical | ReLU, LeakyReLU, ELU, GELU | No |
| `dropout` | float | [0.0, 0.5] step=0.05 | No |
| `batch_norm` | categorical | none, before_act, after_act | No |
| `batch_size` | categorical | 64, 128, 256, 512 | No |
| `lr` | float (log) | [1e-4, 1e-2] | No |
| `weight_decay` | float (log) | [1e-6, 1e-3] | No |
| `optimizer` | categorical | Adam, AdamW, SGD | No |
| `scheduler` | categorical | none, cosine, step | No |
| `step_size` | int | [5, 10] | On `scheduler=step` |
| `gamma` | float | [0.1, 0.5] | On `scheduler=step` |
| `momentum` | float | [0.8, 0.99] | On `optimizer=SGD` |

**Total HP dimensions**: 9 unconditional + 3 conditional = up to 12 simultaneous explore dimensions for TPE.

### Dead Parameter Zones

Zones where **100% of trials were pruned** or only a single outlier survived:

| Zone | Trials | Complete | Finding |
|------|--------|----------|---------|
| `scheduler=none` | 5 | 0 | Never survives — no LR decay causes plateau mid-training |
| `batch_size=256` | 4 | 0 | Pruning artifact: slower epoch → pruned before convergence |
| `batch_norm=none` | 5 | 0 | No BatchNorm destabilizes deeper MLPs |
| `optimizer=SGD` | 5 | 1 | Only #2 survived (lr=1.5e-4); 4/5 pruned |
| `4 layers` | 5 | 2 | Never beats top 2-layer configs |
| `activation=ELU` | 3 | 1 | #2 SGD outlier survives; otherwise fully pruned |
| `activation=LeakyReLU` | 5 | 2 | Both complete but below GELU/ReLU ceiling |

---

## Insights from Optuna DB Analysis

### Hyperparameter Importance Rankings

Computed via FANOVA on all 30 trials targeting validation accuracy (user_attr), matching Optuna dashboard:

| Rank | HP | Importance | Interpretation |
|------|----|:----------:|----------------|
| 1 | `dropout` | **0.32** | Dominates — model is overfitting-sensitive |
| 2 | `batch_norm` | **0.28** | Placement (none/before/after) controls training stability |
| 3 | `lr` | 0.10 | Tied with scheduler — coupled dynamics |
| 4 | `scheduler` | 0.10 | StepLR vs cosine vs none |
| 5 | `activation` | 0.05 | Diminishing returns beyond GELU |
| 6 | `n_layers` | 0.05 | Depth matters little once ≥2 |
| 7 | `optimizer` | 0.04 | Adam vs AdamW vs SGD — SGD is the only differentiator |
| 8 | `weight_decay` | 0.03 | Marginal for 30-epoch training |
| 9 | `batch_size` | 0.02 | Noise — fix to 512 |

**Key insight**: The top 2 HPs (dropout, batch_norm) account for **60%** of importance. The bottom 5 HPs collectively account for only 19% — they're wasting TPE budget.

### Top Reliable Trial Patterns

Top 3 complete trials (excluding #17 anomaly), all converging in 5–7 minutes:

| Trial | Value | Val Acc | Common pattern |
|-------|-------|---------|----------------|
| **#14** | 0.9594 | 90.3% | GELU, Adam, step, after_act, 2 layers, lr=1.3e-3, drop=0.35 |
| **#26** | 0.9588 | 90.4% | GELU, Adam, step, after_act, 2 layers, lr=3.2e-4, drop=0.40 |
| **#25** | 0.9576 | 90.3% | GELU, Adam, step, after_act, 2 layers, lr=3.3e-4, drop=0.40 |

**Convergence cluster**:
- GELU activation, Adam optimizer, step scheduler, after_act BatchNorm
- 2 hidden layers, batch_size=512
- lr ∈ [3e-4, 1.3e-3], dropout ∈ [0.30, 0.40], weight_decay ∈ [5e-6, 2e-5]

### Trial #17 Duration Anomaly

Trial #17 (0.9596, the official best) took **7.4 hours** vs. 5–7 minutes for identical config family peers — an **81× slowdown**. Likely a hardware hiccup (swap thrashing, CPU throttling). Its value is statistically tied with #14 (0.9594) and should be treated as suspect.

---

## Proposed Search Space Refinements

### Parameters to Drop

Completely removed from the search — no evidence of any productive configuration:

| HP | Reason |
|----|--------|
| `scheduler=none` | 0/5 complete. Always pruned. |
| `batch_size=256` | 0/4 complete. Likely a pruning interaction. |
| `batch_norm=none` | 0/5 complete. Never optimal. |
| `optimizer=SGD` | 1/5 complete. Requires lr < 2e-4 to survive — too narrow to justify search budget. |

### Parameters to Fix as Constants

Removed from search; set to the empirically optimal value:

| HP | Fixed value | Evidence |
|----|:-----------:|----------|
| `optimizer` | **Adam** | 8/11 complete, avg value 0.9567. AdamW close but never beats top Adam. |
| `activation` | **GELU** | 7/11 complete, all top-3 use GELU. ReLU second at 0.05 importance. |
| `weight_decay` | **1e-5** | Center of best-trial WD distribution [5e-6, 2e-5]. Importance only 0.03. |
| `batch_size` | **512** | 8/11 complete. Importance 0.02 — statistical noise. |

### Parameters to Narrow

Reduced range to focus TPE sampling on the empirically productive zone:

| HP | Old range | New range | Reason |
|----|-----------|-----------|--------|
| `lr` | [1e-4, 1e-2] log | **[1e-4, 2e-3]** log | No complete trial used lr > 2e-3. High end was wasted space. |
| `dropout` | [0.0, 0.5] step=0.05 | **[0.2, 0.5]** step=0.05 | Best cluster at 0.30–0.40. Low dropout (0.0–0.15) underperforms. |
| `n_layers` | [1, 4] | **[1, 3]** | 4-layer networks never beat 2-layer. Restricting to 3 saves n_layers=4 conditional sampling. |
| `scheduler` | [none, cosine, step] | **[step, cosine]** | `none` is dead. Keep step + cosine for coverage. |

### Parameters to Keep Full Range

| HP | Range | Reason |
|----|-------|--------|
| `batch_norm` | [before_act, after_act] | #2 importance (0.28). Both placements have value. |
| `units_i` | 64–1024 log (≤3 layers) | Still productive to tune layer capacity. |
| `step_size` | [5, 10] (on scheduler=step) | Coupled with gamma — keep full range. |
| `gamma` | [0.1, 0.5] (on scheduler=step) | Coupled with step_size — keep full range. |

### Proposed Search Space Summary

After applying all refinements:

| HP | Type | Range |
|----|------|-------|
| `n_layers` | int | [1, 3] |
| `units_i` | int (log) | 64–1024 |
| `dropout` | float (step=0.05) | [0.2, 0.5] |
| `batch_norm` | categorical | before_act, after_act |
| `lr` | float (log) | [1e-4, 2e-3] |
| `scheduler` | categorical | step, cosine |
| `step_size` | int | [5, 10] (conditional on step) |
| `gamma` | float | [0.1, 0.5] (conditional on step) |

**Fixed constants**: optimizer=Adam, activation=GELU, weight_decay=1e-5, batch_size=512

**Dropped entirely**: SGD, scheduler=none, batch_size other than 512, batch_norm=none, ELU, LeakyReLU, 4-layer networks, weight_decay search, optimizer search, activation search

**Net reduction**: 12 conditional dimensions → **6 effective dimensions** (4 unconditional + 2 conditional)

---

## Implementation Plan

### Objective Function Changes

1. Remove `batch_size` from `trial.suggest_categorical` — use `BATCH_SIZE = 512` constant
2. Remove `weight_decay` from `trial.suggest_float` — use `WEIGHT_DECAY = 1e-5` constant
3. Remove `optimizer` from `trial.suggest_categorical` — use `optim.Adam` directly
4. Remove `activation` and `build_mlp` act_name selection — hardcode `nn.GELU()`
5. Collapse `n_layers` range to [1, 3]
6. Collapse `units_i` to a single range (64–1024, no need for conditional low/high switching)
7. Narrow `lr` to [1e-4, 2e-3] log
8. Narrow `dropout` to [0.2, 0.5]
9. Restrict `scheduler` to [step, cosine]
10. Drop `momentum` (no SGD)
11. Drop `batch_norm` "none" option — restrict to [before_act, after_act]

### Pruner and Study Config Tuning

| Setting | Current | Proposed | Rationale |
|---------|---------|----------|-----------|
| `n_startup_trials` | 5 | **5** | Keep — gives TPE enough data before pruning |
| `n_warmup_steps` | 3 | **5** | Increase — batch_size=512 epochs train fast; 3 steps is too early to judge |
| `interval_steps` | 1 | **1** | Keep — check every epoch |
| `n_trials` | 30 | **20** | Reduced space needs fewer trials to converge |
| `EPOCHS_PER_TRIAL` | 30 | **30** | Keep — matches final retrain length |

### Expected Trial Budget

- **6 effective dimensions**: ~15–20 trials for TPE convergence (vs. ~40–50 for 12 dimensions)
- **Set n_trials=20** with `show_progress_bar=True`
- Expected runtime: 20 trials × 5–7 min = **~2 hours** (vs. 30 trials × variable duration = ~4+ hours in Phase 4)
- Lower risk of pruner waste: narrower space means fewer dead draws in early trials

---

## Expected Outcomes and Validation

### Expected Improvements

| Metric | Phase 4 (best) | Phase 5 target | Rationale |
|--------|:--------------:|:--------------:|-----------|
| Top-1 macro PR-AUC | ~0.959 | **~0.962–0.965** | Tightened search avoids dead zones; all trials produce viable configs |
| Complete trial ratio | 11/30 (37%) | **~16/20 (80%)** | Dead zones removed; fewer prunes |
| Search time | ~4+ hours | **~2 hours** | Fewer trials + no 7-hour outlier |
| Top-1 test accuracy (30-epoch retrain) | 90.00% | **~90.3–90.5%** | Better HP focus → better final config |

### Validation Strategy

1. Run 20-trial search with the refined space
2. Compare against Phase 4 best config (#14) and top-3 retrain accuracies
3. Verify no regression in complete/pruned ratio
4. Cross-check importance ranking post-run — expect dropout + batch_norm to remain dominant, with lr dropping due to narrowed range

### Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|:----------:|------------|
| Fixing `weight_decay=1e-5` misses a better high-WD config | Low | WD importance is 0.03; the top trials cluster tightly at 5e-6 to 2e-5 |
| Fixing `batch_size=512` misses a small-batch generalization benefit | Low | BS importance is 0.02; 256 = 0/4 complete; 64/128 produce fewer complete trials |
| Removing `scheduler=none` misses a config where no scheduler outperforms | None | 0/5 complete — empirically dead |
| Dropping `optimizer=SGD` misses a converged SGD baseline | Very low | 1/5 complete, value=0.9335 vs. Adam top at 0.959. Not competitive. |
| `n_warmup_steps=5` prunes too late (wastes epochs on bad trials) | Medium | EPOCHS_PER_TRIAL=30 → warmup=5 is 17% of training — reasonable |
