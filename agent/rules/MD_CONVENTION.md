# MD_CONVENTION.md
Rules for any .md file the agent creates in this project. Condensed from
MD_creation_guide.md — see that file for full rationale and examples.

## Required header (5 fields, in order)
```
# <Title>

- **Motivation/Background**: 1–3 sentences — why this doc exists.
- **Purpose**: one sentence — what this doc achieves.
- **Overview Pipeline**: 1–2 sentences — the process that produced this content.
- **Detailed Plan**: compact list of sections/subsections and what each covers.
- **References**: comma-separated libraries/tools/frameworks used.

---
```

## Table of Contents
- Place immediately after the header `---`, before first content section
- Include all `##` and `###` headings; omit `####`+ unless critical
- Anchor rule: lowercase, strip punctuation except hyphens, spaces→hyphens
- Regenerate TOC whenever a section is added/removed

## Body formatting
- `##` top-level sections, `###` subsections, `####` only if necessary
- No vague headings ("Details", "Info")
- Tables for structured comparisons; bullets for unordered items;
  numbered lists for sequential steps; blockquotes for experiment-design
  notes; bold for key metrics; code blocks for paths/commands/config
- Each experiment section needs a `### Key findings` subsection:
  plain-language summary, connects to prior experiments, flags regressions
- Multi-experiment docs end with `## Cross-Experiment Summary`
  (comparison table + narrative + limitations)

## Conventions
| Rule | Requirement |
|---|---|
| Single-variable principle | One changed factor per experiment; state what's held constant |
| File paths | Relative from project root |
| Dates | `YYYY-MM-DD` |
| Notebook refs | `notebooks/<category>/<experiment>/<filename>.ipynb` |
| Output dirs | `outputs/<category>/<experiment>/` |
| Separators | `---` after header and between major sections |
| Variable docs | `> **Variable changed**: ...` / `> **Held constant**: ...` pair above each experiment |
| Metric deltas | Show Δ columns in comparison tables; bold positive Δ |

## Self-review before finalizing
- [ ] All 5 header fields present
- [ ] TOC anchors resolve
- [ ] Metrics match source
- [ ] File/notebook paths correct
