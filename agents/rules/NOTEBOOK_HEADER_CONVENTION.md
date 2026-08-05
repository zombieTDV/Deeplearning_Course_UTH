# NOTEBOOK_HEADER_CONVENTION.md
Rules for the first cell of every notebook. Condensed from
notebook_header_guide.md — see that file for the full worked example.

Every notebook's first cell = a single markdown cell with 3 sections, in order:

## 1. Title (H1)
`# <Scope> <N>: <Short Description>`

| Scope prefix | Use for |
|---|---|
| `Practice N:` | Standalone practice/learning notebooks |
| `Phase N:` | Foundational / diagnostic experiments |
| `Experiment N:` | Single-variable-change experiments |
| `Appendix N:` | Supplementary / follow-up analysis |

## 2. Subtitle + Purpose (H2 + short paragraph)
One-line H2 subtitle + 1–2 sentences on what the notebook does and why.

For single-variable experiments, use this structured form instead:
```
## Rationale
<why this experiment>

**Single variable changed**: <the one thing>
**Held constant**: <everything else — training config, loss, data pipeline, etc.>
```

## 3. Roadmap Table
Exactly these 4 columns, one row per notebook step:
```
| Step | Description | What it does | Import path |
```
- **Step**: sequential number
- **Description**: short action phrase
- **What it does**: 5–15 words
- **Import path**: `src/...` module or `—` if none
- Table must end with `---` immediately after

## Hard rule
This entire header is ONE markdown cell — the first cell in the notebook.
Section headings in the notebook body come after this block, not inside it.

## Output persistence & cell independence (hard rules)

### 1. Persist every output to its designated folder
When working in a notebook, ALL produced data must be written to its
designated folder — never left only in-memory or inside cell outputs:

| Output | Designated folder |
|---|---|
| Training results, per-epoch metrics, summary tables | `experiments/results/` |
| TensorBoard logs | `experiments/runs/` |
| Checkpoints | `experiments/checkpoints/` |
| Plots / figures | `experiments/plots/` |

Use `PROJECT_ROOT`-relative `Path` objects and explicit `write_text` /
`torch.save` / `SummaryWriter` calls so results survive kernel restarts and
are reusable by later cells, other notebooks, and scripts.

### 2. Each cell must be runnable independently
Structure the notebook so any specific cell can be executed on its own,
without requiring all preceding cells — especially resource- or time-intensive
ones (e.g. a long training loop):
- Recompute cheap prerequisites inline (data loaders, model builders, device),
  OR load persisted artifacts (checkpoints, saved metrics) instead of depending
  on an earlier cell having run.
- Never require re-running a long training cell just to re-evaluate: if a
  checkpoint exists, load it; otherwise train.
- Cells should be idempotent and safe to re-run in isolation once their cheap
  prerequisites are satisfied.
