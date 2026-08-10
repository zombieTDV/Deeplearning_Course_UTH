# RESULTS_REPORTING.md — 5W1H Rules for Every Reported Result

- **Motivation/Background**: Results shown to teammates or teachers were often bare numbers (e.g. `test_acc=97.21%`) with no context, making them unverifiable and hard to interpret.
- **Purpose**: Make the 5W1H principle (What, Why, When, Where, Who, How) mandatory for every metric presented in docs, notebooks, or reports.
- **Overview Pipeline**: Derived from teacher feedback on LAB2; applies to every result in `agents/experiments/*.md`, `experiments/results/*`, and analysis notebooks.
- **Detailed Plan**: §1 the 5W1H rule; §2 the required block format; §3 metric-description pairs; §4 where it applies; §5 checklist.
- **References**: `agents/experiments/SUMMARY_RESULTS.md`, `experiments/results/README.md`, MD_CONVENTION.md.

---

## Table of Contents

- [1. The Rule](#1-the-rule)
- [2. Required 5W1H Block](#2-required-5w1h-block)
- [3. Metric Description Pairs](#3-metric-description-pairs)
- [4. Where It Applies](#4-where-it-applies)
- [5. Checklist](#5-checklist)

---

## 1. The Rule

Every result value reported to teammates or teachers MUST be accompanied by a
5W1H explanation. A number alone is not a result — it is a claim.

- **What** — which metric, computed on which data split, which model variant.
- **Why** — the question this metric answers / why it matters.
- **When** — when it was measured (date, checkpoint, epoch, seed).
- **Where** — where the artifact lives (file/notebook/plot path) and where it was run (device).
- **Who** — who produced it (author/team) and who it is for.
- **How** — how it was computed (evaluation protocol, hyperparameters, leakage notes).

## 2. Required 5W1H Block

Insert this block (or the equivalent table) before any results table in a report:

```markdown
> **5W1H — <Result headline>**
> - **What**: ...
> - **Why**: ...
> - **When**: measured <date> on checkpoint <name> (epoch <n>, seed <s>).
> - **Where**: artifacts in `experiments/results/<file>.json`; run on <device>.
> - **Who**: <author> — for <audience>.
> - **How**: <evaluation protocol, hyperparameters, leakage guarantees>.
```

For per-metric explanations use the table form in §3.

## 3. Metric Description Pairs

Every metric in a table must have a companion "what it means / how it is
computed" line. Standard LAB2 metrics:

| Metric | 5W1H description |
|---|---|
| `test_acc` | What: top-1 accuracy on the official 10k test split. How: argmax over softmax, no TTA unless stated. Why: headline generalization. |
| `val_acc` / `val_loss` | What: metrics on the fixed 5k validation split (seed 42). How: computed every epoch after training step. Why: model selection. |
| `isolated_acc` | What: accuracy restricted to cat+dog samples. How: subset of test where `y ∈ {cat, dog}`. Why: LAB2 focuses on the hardest class pair. |
| `cross` | What: number of cat↔dog cross-errors. How: `cat→dog + dog→cat` counts on test. Why: quantifies the confusion cluster. |
| `dAcc` | What: delta vs the reference baseline (fixed 0.5/0.5 soft-voting). How: `metric - baseline`. Why: isolates the added method's gain. |
| `F1 (macro)` | What: mean per-class F1. Why: class-balance-aware summary. |
| `latency (ms)` | What: mean±std wall-clock per batch of 64 at 224×224. How: `time.perf_counter`, warmup 10, 50 iters, batch 64. Why: deployment cost. |
| `img/s` | What: throughput = batch/latency. Why: efficiency at serving time. |
| `params (M)` | What: total parameter count. Why: model size / memory cost. |

**Removed**: `GFLOPs` — measured FLOPs were unstable (dataloader noise dominated
the 1–2% spread) and did not reflect real cost; params + latency + throughput
are the meaningful, reproducible efficiency metrics.

## 4. Where It Applies

- `agents/experiments/*.md` result tables (add the block above each table).
- `experiments/results/README.md` metric index (every file gets What/Why/How).
- Notebook analysis sections that present numbers to an audience.
- Any message to teammates/teachers quoting a metric.

## 5. Checklist

- [ ] Every table has a 5W1H block (or header row with context).
- [ ] Every metric has a companion description (§3 table or inline).
- [ ] Split, seed, checkpoint, and evaluation protocol are stated.
- [ ] Artifact file paths are given.
- [ ] No bare numbers without context.
