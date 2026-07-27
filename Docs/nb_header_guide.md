# Notebook Header Convention

Every notebook must begin with a **single markdown cell** containing three sections in order.

---

## 1. Title (H1)

Format: `# <Scope> <N>: <Short Description>`

| Scope prefix | When to use |
|---|---|
| `Practice N:` | Standalone practice/learning notebooks |
| `Phase N:` | Foundational / diagnostic experiments |
| `Experiment N:` | Single-variable-change experiments |
| `Appendix N:` | Supplementary / follow-up analysis |

Examples:
- `# Practice 1: PyTorch FashionMNIST Classification`
- `# Phase 1: MLP Error Analysis — Baseline + Extended + Logit Bias Sweep`
- `# Experiment 2: Architecture Change — Wider MLP (512→256→10)`

---

## 2. Subtitle + Purpose (H2 + paragraph)

One-line H2 subtitle followed by 1–2 sentences explaining what the notebook does and why.

```
## MLP vs DiagnosticCNN — Comparison

Notebook comparing the performance of a simple MLP against a DiagnosticCNN
on the FashionMNIST dataset across 13 metrics and visualizations.
```

For single-variable experiments, use a structured list instead:

```
## Rationale

Doubling hidden layer capacity to check if Shirt confusion is a model-capacity bottleneck.

**Single variable changed**: Architecture (hidden layer sizes / depth)
**Held constant**: Training (30 epochs, Adam lr=0.001, CosineAnnealingLR), loss (CrossEntropy),
                 dropout (0.2), data pipeline, evaluation protocol
```

---

## 3. Roadmap Table

A table mapping each cell/section to its purpose and source module. Must have exactly these columns:

```
| Step | Description | What it does | Import path |
|------|-------------|--------------|-------------|
| 1 | Import Libraries | Load PyTorch, src modules, detect device | — |
| 2 | Load Dataset | Download FashionMNIST + apply transforms | `src/data_utils.py` |
| 3 | Explore Dataset | Shape, class distribution, pixel stats | `src/vis_utils.py` |
```

Rules:
- **Step**: sequential number (1, 2, 3, …)
- **Description**: short action phrase ("Train model", "Compare confusion matrices")
- **What it does**: 5–15 words explaining the objective
- **Import path**: the src module path or `—` (em dash) if none
- The table must end with a `---` horizontal rule immediately after the table

---

## Full Example

```markdown
# Practice 1: PyTorch FashionMNIST Classification

## MLP vs DiagnosticCNN — Comparison

| Step | Description | What it does | Import path |
|------|-------------|--------------|-------------|
| 1 | Import Libraries | Load PyTorch, src modules, detect device | — |
| 2 | Load Dataset | Download FashionMNIST + apply transforms | `src/data_utils.py` |
| 3 | Explore Dataset | Shape, class distribution, pixel stats | `src/vis_utils.py` |
| 4 | Create DataLoaders | Split into train/test batches | `src/data_utils.py` |
| 5 | Define Models | MLP (Linear) vs DiagnosticCNN (Conv, BN, GAP) | — |
| 6 | Train Both Models | MLP: 10 epochs Adam; CNN: 35 epochs AdamW | `src/train_utils.py` |
| 7 | Compare: Loss Curves | Loss curves of MLP + CNN overlaid | `src/vis_utils.py` |
| 8 | Compare: Confusion Matrices | Per-class TPR/FPR/Precision + heatmaps | `src/eval_utils.py` |
| 9 | Compare: ROC Curves | Per-class ROC curves + AUC scores | `src/eval_utils.py`, `src/vis_utils.py` |
| 10 | Compare: PR Curves | Per-class Precision-Recall curves + AP | `src/eval_utils.py`, `src/vis_utils.py` |
| 11 | Summary Table | Accuracy / ROC-AUC / PR-AUC / Params side-by-side | `src/eval_utils.py` |
| 12 | Save Models | Save .pth weights to `outputs/model/` | `src/model_utils.py` |
| 13 | Display Predictions | Predicted vs actual images for MLP + CNN | `src/vis_utils.py` |

---
```

This header must be **a single markdown cell** — the first cell in the notebook. Section headings within the notebook body (`## 1. Import Libraries`, etc.) follow after this block.
