# MD_CONVENTION.md — Markdown Formatting, Timestamp & Metadata Standard

- **Motivation/Background**: Project documentation across audits, phase specifications, bug reports, and experiment records requires consistent structure and strict temporal traceability to avoid confusion between current, stale, and superseded artifacts.
- **Purpose**: Define the mandatory 7-field header (including metadata and update timestamps), anchor/TOC rules, clickable cross-reference standards, and lifecycle conventions for all `.md` files in this repository.
- **Overview Pipeline**: Formulated during project consolidation and codified as an immutable constitutional governance rule.
- **Detailed Plan**: §1 Required Header Specification; §2 Markdown Update-Timestamp Standard; §3 Document Lifecycle Statuses; §4 Mandatory Cross-Reference Links; §5 Body Formatting Rules; §6 Conventions Table; §7 Self-Review Checklist.
- **References**: `agents/rules/FOLDER_STRUCTURE.md`, `agents/rules/CODEBASE_AUDIT.md`.
- **Created**: 2026-09-06T13:05:18+07:00
- **Last Updated**: 2026-09-06T21:25:00+07:00

---

## Table of Contents

- [1. Required Header Specification (7 Fields)](#1-required-header-specification-7-fields)
- [2. Markdown Update-Timestamp Standard](#2-markdown-update-timestamp-standard)
- [3. Document Lifecycle Statuses](#3-document-lifecycle-statuses)
- [4. Mandatory Cross-Reference Links](#4-mandatory-cross-reference-links)
- [5. Body Formatting Rules](#5-body-formatting-rules)
- [6. Conventions](#6-conventions)
- [7. Self-Review Before Finalizing](#7-self-review-before-finalizing)

---

## 1. Required Header Specification (7 Fields)

Every Markdown file created or materially modified in this repository MUST start with the standard 7-field header block in exact order:

```markdown
# <Title>

- **Motivation/Background**: 1–3 sentences — why this doc exists.
- **Purpose**: one sentence — what this doc achieves.
- **Overview Pipeline**: 1–2 sentences — the process that produced this content.
- **Detailed Plan**: compact list of sections/subsections and what each covers.
- **References**: comma-separated libraries/tools/frameworks/rules used.
- **Created**: YYYY-MM-DDTHH:MM:SS±HH:MM (ISO 8601 with explicit timezone offset)
- **Last Updated**: YYYY-MM-DDTHH:MM:SS±HH:MM (ISO 8601 with explicit timezone offset)

---
```

*(Alternatively, structured reference or audit documents may render these 7 fields as a Markdown key-value table.)*

---

## 2. Markdown Update-Timestamp Standard

### Exact Rules

1. **Format:** Standard ISO 8601 extended format with explicit timezone offset: `YYYY-MM-DDTHH:MM:SS±HH:MM` (e.g. `2026-09-06T21:25:00+07:00`).
2. **Timezone Convention:** Use the local project timezone offset (e.g. `+07:00` or UTC `Z`). Never omit the timezone designator.
3. **Creation Timestamp (`Created`):** Set once when the file is created. Never modified afterwards.
4. **Update Timestamp (`Last Updated`):**
   - Must be updated **every time** the file is edited or touched by an agent or human.
   - Any agent performing modifications to a `.md` file is strictly required to refresh `Last Updated` to the current execution time.
5. **Handling Existing Markdown Files Without Timestamps:**
   - When encountering an older `.md` file lacking timestamp fields, do not invent historical timestamps.
   - Use the best verifiable timestamp available from Git history (`git log --follow --reverse --format="%aI" -1 -- <file>` for `Created`, `git log -1 --format="%aI" -- <file>` for `Last Updated`) or explicit dates embedded in the document text, clearly preserving provenance.

---

## 3. Document Lifecycle Statuses

To eliminate ambiguity across multi-session work, documents tracking plans, roadmaps, or progress should display their lifecycle state:

- **`[STATUS: ACTIVE]`**: Canonical, currently maintained document.
- **`[STATUS: COMPLETED]`**: Finalized plan or completed milestone preserved for historical audit.
- **`[STATUS: SUPERSEDED]`**: Replaced by a newer document. Must include a prominent link:
  ```markdown
  > [!WARNING]
  > This document is SUPERSEDED by [NEW_DOC.md](path/to/NEW_DOC.md). Retained for historical provenance only.
  ```

---

## 4. Mandatory Cross-Reference Links

- **Whenever a file, module, notebook, or artifact path is mentioned in AI-generated Markdown, it MUST be a working cross-reference link** — never bare text.
- Link formats:
  - Same tree: relative link from document location (`src/lab2/training/train_model.py`, `notebooks/lab2/practice_2.ipynb`).
  - Jump to a heading: cross-file anchor link (`agents/rules/LOGGING_CHECKPOINT_RULES.md#5-resume-procedure`).
  - External resources: canonical URL.
- **External Model & Dataset Links:** Every reference to an external model ID (e.g. Hugging Face [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased), torchvision `resnet18`) or dataset ID (e.g. [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb), CIFAR-10) MUST be written as a direct, clickable markdown link to its canonical hub/documentation page.
- Notebook headers must include a `## References` block linking the rules, scripts, and artifact locations they consume per [NOTEBOOK_HEADER_CONVENTION.md](NOTEBOOK_HEADER_CONVENTION.md).

---

## 5. Body Formatting Rules

- `##` top-level sections, `###` subsections, `####` only if necessary.
- No vague headings ("Details", "Info").
- Tables for structured comparisons; bullets for unordered items; numbered lists for sequential steps; blockquotes for experiment notes; bold for key metrics; code blocks for commands/paths.
- Each experiment section needs a `### Key findings` subsection: plain-language summary connecting to prior experiments and flagging regressions.
- Multi-experiment docs end with `## Cross-Experiment Summary` (comparison table + narrative + limitations).

---

## 6. Conventions

| Rule | Requirement |
| :--- | :--- |
| **Header (7 fields)** | Motivation/Background, Purpose, Overview Pipeline, Detailed Plan, References, Created, Last Updated. |
| **Header Timestamps** | Mandatory `Created` and `Last Updated` in ISO 8601 (`YYYY-MM-DDTHH:MM:SS±HH:MM`). |
| **Table of Contents** | Required if document exceeds ~50 lines. Placed immediately after the header `---`. |
| **Links** | Mandatory clickable links for every mentioned path, module, notebook, model hub ID, or dataset. |
| **Code blocks** | Always declare syntax highlighting language (`bash`, `python`, `text`, `yaml`, `mermaid`). |
| **TOC Anchors** | Lowercase, strip punctuation except hyphens, spaces to hyphens. |

---

## 7. Self-Review Before Finalizing

Before closing any `.md` editing turn, verify:
- [ ] Header has all 7 required fields in exact order.
- [ ] `Last Updated` reflects the current time with explicit timezone offset.
- [ ] All mentioned paths, files, and external IDs are active clickable links.
- [ ] TOC matches all `##` and `###` headings.
