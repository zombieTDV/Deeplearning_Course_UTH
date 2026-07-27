# Markdown Documentation Creation Guide

This guide documents the process for creating comprehensive Markdown documentation files — from initial analysis to final review.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Workflow Overview](#workflow-overview)
- [Step 1 — Understand the Source Material](#step-1--understand-the-source-material)
- [Step 2 — Craft the Header Structure](#step-2--craft-the-header-structure)
- [Step 3 — Add a Table of Contents](#step-3--add-a-table-of-contents)
- [Step 4 — Organize the Body Content](#step-4--organize-the-body-content)
- [Step 5 — Review and Refine](#step-5--review-and-refine)
- [Conventions and Best Practices](#conventions-and-best-practices)

---

## Prerequisites

- Familiarity with the project's source material (notebooks, experiments, code)
- Understanding of the target audience (AI agents, human collaborators, or both)
- Knowledge of Markdown syntax (headings, lists, tables, code blocks, links, anchors)

---

## Workflow Overview

```
Read source material ──> Identify key sections ──> Draft header ──> Build TOC ──> Write body ──> Review
```

The process is iterative: each step may reveal missing context that requires re-reading the source.

---

## Step 1 — Understand the Source Material

Before writing any documentation, thoroughly review the source:

1. **Read all source files** — notebooks, changelogs, configuration files, and related outputs.
2. **Identify the core narrative** — What problem does this document address? What was the process? What were the results?
3. **Note dependencies** — libraries, frameworks (e.g., PyTorch, scikit-learn), schedulers (e.g., CosineAnnealingLR), and datasets (e.g., Fashion-MNIST).
4. **Extract key data points** — metrics, comparisons, tables, and figures that must be preserved.

**Output of this step**: A mental or written inventory of sections to include and the relationships between them.

---

## Step 2 — Craft the Header Structure

Every documentation file must begin with a structured header containing five mandatory fields. These fields provide an at-a-glance summary of the document's purpose and scope.

### 2.1 Motivation/Background

Explain *why* this document exists and the context that led to its creation.

```
- **Motivation/Background**: <1–3 sentences describing the problem space, system role, and rationale.>
```

**Example**: *This document tracks MLP experiments for error analysis on Fashion-MNIST. The MLP serves as a non-convolutional baseline to understand how fully-connected architectures handle fine-grained class discrimination, and to identify architectural limitations compared to CNNs.*

### 2.2 Purpose

State the primary objective in a single, actionable sentence.

```
- **Purpose**: <One sentence defining what this document achieves.>
```

**Example**: *Document all MLP-specific experiments, results, and findings following a single-variable principle to systematically evaluate how width and depth affect classification performance.*

### 2.3 Overview Pipeline

Summarize the workflow, process, or methodology that produced the document's content.

```
- **Overview Pipeline**: <1–2 sentences describing the sequence of steps or experiments.>
```

**Example**: *The document was created through iterative experimentation: Phase 1 established a diagnostic baseline, followed by Experiment 2 (wider MLP) and Experiment 3 (deeper MLP), with each step varying one architectural factor while holding all others constant.*

### 2.4 Detailed Plan

Provide a roadmap of the document's sections and subsections in a compact, readable format.

```
- **Detailed Plan**: <Section A — description; Section B — description; ...>
```

**Example**: *Phase 1 — MLP Diagnostic Baseline (baseline 10-epoch training, extended 30-epoch training with CosineAnnealingLR, logit bias sweep); Experiment 2 — Wider MLP (512→256→10); Experiment 3 — Deeper MLP (512→256→128→10); Cross-Experiment Summary.*

### 2.5 References

List libraries, frameworks, tools, and key dependencies used in the underlying process.

```
- **References**: <Comma-separated list with parenthetical clarifications where helpful.>
```

**Example**: *PyTorch, torchvision (Fashion-MNIST), scikit-learn (accuracy_score, classification metrics), matplotlib, seaborn, NumPy, pandas, Adam optimizer, CosineAnnealingLR scheduler.*

### Header Example

```
# Document Title

- **Motivation/Background**: ...
- **Purpose**: ...
- **Overview Pipeline**: ...
- **Detailed Plan**: ...
- **References**: ...

---
```

---

## Step 3 — Add a Table of Contents

A Table of Contents (TOC) improves navigation for both human readers and AI agents.

### 3.1 Placement

Insert the TOC immediately after the header (after `---` if a separator is used) and before the first content section.

### 3.2 Anchor Links

Each `[text](#anchor)` link must match the target heading's auto-generated anchor:

- Convert headings to lowercase
- Remove punctuation except hyphens
- Replace spaces with hyphens
- Strip leading numbers and special characters

**Anchor mapping examples:**

| Heading | Anchor |
|---------|--------|
| `## Phase 1 — MLP Diagnostic Baseline` | `#phase-1--mlp-diagnostic-baseline` |
| `### Results — Baseline (10 epochs)` | `#results--baseline-10-epochs` |
| `## Experiment 2 — Architecture: Wider MLP` | `#experiment-2--architecture-wider-mlp` |

### 3.3 Scope

- Include all `##` (major) sections
- Include all `###` (subsection) headings under each major section
- Omit `####` or deeper unless the subsection is critical

### 3.4 TOC Example

```
## Table of Contents

- [Phase 1 — MLP Diagnostic Baseline](#phase-1--mlp-diagnostic-baseline)
  - [Changes](#changes)
  - [Results — Baseline (10 epochs)](#results--baseline-10-epochs)
- [Experiment 2 — Architecture: Wider MLP](#experiment-2--architecture-wider-mlp)
  - [Results](#results-1)
  - [Key findings](#key-findings)
- [Cross-Experiment Summary](#cross-experiment-summary)
```

---

## Step 4 — Organize the Body Content

### 4.1 Section Headings

- `##` for top-level sections (e.g., `## Phase 1 — MLP Diagnostic Baseline`)
- `###` for subsections (e.g., `### Results`, `### Changes`)
- `####` for sub-subsections only when necessary

Keep headings concise and descriptive. Avoid vague labels like "Details" or "Info".

### 4.2 Data Presentation

- **Tables** for structured comparison (metrics, parameters, configurations)
- **Bullet lists** for unordered items (key findings, changes, fixes)
- **Numbered lists** for sequential steps
- **Blockquotes** (`>`) for experimental design notes or variable-change descriptions
- **Bold** for emphasis on key metrics or values
- **Code blocks** for file paths, commands, or configuration snippets

### 4.3 Key Findings Sections

Each experiment section should include a `### Key findings` subsection that:

- Summarizes the most important results in plain language
- Explains what was learned (not just what happened)
- Connects findings to previous experiments (e.g., "Confirms Phase 1 hypothesis")
- Highlights unexpected regressions or limitations

### 4.4 Cross-Experiment Summary

When a document contains multiple experiments, include a final `## Cross-Experiment Summary` with:

- A comparison table consolidating key metrics
- A narrative paragraph synthesizing overarching conclusions
- A statement of limitations or architectural ceilings

---

## Step 5 — Review and Refine

### 5.1 Self-Review Checklist

- [ ] Does the header contain all five mandatory fields?
- [ ] Do all TOC anchor links resolve correctly?
- [ ] Are tables properly aligned and readable?
- [ ] Are metrics and numbers accurate (check against source)?
- [ ] Are file paths and notebook references correct?
- [ ] Is the language consistent in tense and style?
- [ ] Are experimental variables clearly documented (what changed, what stayed constant)?

### 5.2 Formatting Checks

- No trailing whitespace on lines
- Tables render correctly with aligned columns
- Separators (`---`) used between major experiments and after the header
- Consistent use of bold for schema and italics for emphasis

---

## Conventions and Best Practices

| Convention | Guideline |
|---|---|
| **Single-variable principle** | Each experiment changes exactly one factor; document the held-constant variables explicitly. |
| **File paths** | Use relative paths from project root. Prefer absolute references using `PROJ_ROOT` or `_find_root()` over fragile relative paths. |
| **Date format** | `YYYY-MM-DD` throughout. |
| **Notebook references** | `notebooks/<category>/<experiment>/<filename>.ipynb` format. |
| **Output directories** | `outputs/<category>/<experiment>/` format. |
| **Separators** | `---` between major experiment sections and after the header block. |
| **Bias/variable documentation** | Use a `> **Variable changed**: ...` / `> **Held constant**: ...` blockquote pair above each experiment's metadata. |
| **Metric deltas** | Show Δ columns in comparison tables. Highlight positive Δ in bold. |
| **TOC updates** | Regenerate the TOC whenever a new section or subsection is added. |

---

## Example: Header and TOC in Practice

```
# MLP Changelog

- **Motivation/Background**: This document tracks MLP experiments for error analysis on Fashion-MNIST...
- **Purpose**: Document all MLP-specific experiments, results, and findings...
- **Overview Pipeline**: The document was created through iterative experimentation: Phase 1...
- **Detailed Plan**: Phase 1 — MLP Diagnostic Baseline; Experiment 2 — Wider MLP; ...
- **References**: PyTorch, torchvision (Fashion-MNIST), scikit-learn, matplotlib, ...

## Table of Contents

- [Phase 1 — MLP Diagnostic Baseline](#phase-1--mlp-diagnostic-baseline)
  - [Changes](#changes)
  - [Results](#results)
- ...

---

## Phase 1 — MLP Diagnostic Baseline
```

Refer to `MLP_changelog.md` for a complete worked example of this guide.
