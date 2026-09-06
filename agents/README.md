# AI Agent Constitution (`/agents`)

- **Motivation/Background**: This directory serves as the immutable project-wide governance layer for AI coding agents across all deep learning coursework modules.
- **Purpose**: Define permanent behavior rules, quality gates, coding standards, and document templates that do NOT change between research stages.
- **Overview Pipeline**: Read on session start → enforce rules during code/doc generation → verify before long runs or pull requests.
- **Detailed Plan**: §1 Constitutional Architecture; §2 Distinction Between `/agents` and `/docs`; §3 Working Rules Summary.
- **References**: `agents/rules/`, `agents/templates/`, `docs/shared/`.
- **Created**: 2026-09-06T13:05:18+07:00
- **Last Updated**: 2026-09-06T21:25:00+07:00

---

## 1. Constitutional Architecture

Per course architecture standards, `/agents` is treated strictly as an **immutable constitution**, not an evolving knowledge base.

```text
agents/
├── README.md                          # This file (governance index)
├── rules/                             # What the AI agent MUST consistently do
│   ├── AGENT_AI.md                    # Core behavior layer, 6-stage workflow & prompting rules
│   ├── CODEBASE_AUDIT.md              # Drift audit procedure & finding resolution lifecycle
│   ├── FOLDER_STRUCTURE.md            # Canonical repository layout (Dual-Paradigm architecture)
│   ├── LOGGING_CHECKPOINT_RULES.md    # Script-only training & full-state checkpoint rules
│   ├── MD_CONVENTION.md               # Markdown formatting, timestamps & clickable link standards
│   ├── NAMING_CONVENTION.md           # File, code, and experiment naming rules
│   ├── NOTEBOOK_HEADER_CONVENTION.md  # Standardized notebook first-cell headers
│   ├── PYTORCH_FRAMEWORK_RULES.md     # PyTorch device/seed/VRAM/eval rules
│   └── RESULTS_REPORTING.md           # 5W1H empirical reporting protocol
└── templates/                         # Standardized document skeletons
    ├── BUG_TEMPLATE.md                # Bug report skeleton
    ├── CODEBASE_AUDIT_TEMPLATE.md     # Audit report skeleton (with Status lifecycle)
    ├── EXPERIMENT_TEMPLATE.md         # Experiment report skeleton
    ├── PHASE_DOC_TEMPLATE.md          # Pipeline phase specification skeleton
    ├── PROGRESS_STATUS_TEMPLATE.md    # Phase progress tracking skeleton
    ├── PROJECT_ROADMAP_TEMPLATE.md    # Milestone & execution roadmap skeleton
    ├── REFERENCE_TEMPLATE.md          # Reusable reference guide skeleton
    └── SMOKE_TEST_CHECKLIST.md        # Pre-execution verification checklist
```

---

## 2. Distinction Between `/agents` and `/docs`

To keep the repository clean and academically defensible:

* **`/agents` (The Constitution):** Contains only rules and templates that dictate what the agent **MUST** consistently do. It does not evolve between labs.
* **`/docs` (The Research Memory):** Contains evolving project knowledge, research briefs, stage phase specifications, experiment reports, and progress tracking organized by research stage (`docs/lab1`, `docs/lab2`, `docs/lab3`).
* **`/docs/shared/`:** Contains procedural SOPs and reference guides (e.g. `HOW_TO_SETUP_AI_AGENT.md`, `HANDOFF_TEMPLATE.md`, `ML_PIPELINE_REFERENCE_v3.md`, `OPTUNA_DB_GUIDE.md`, `GIT_AND_RELEASE_BEST_PRACTICES.md`, `MD_CREATION_GUIDE.md`).

---

## 3. Working Rules Summary

1. **Mandatory Codebase Audit**: Run [rules/CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md) before multi-file refactors to prevent drift, tracking individual finding resolution status (`RESOLVED`, `PARTIALLY RESOLVED`, `NOT RESOLVED`).
2. **Script-Only Training**: Training loops must run from `src/` scripts with full-state checkpointing; notebooks are strictly for testing, visualization, and analysis ([rules/LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md)).
3. **5W1H Results Reporting**: Every reported metric must include What, Why, When, Where, Who, and How context ([rules/RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md)).
4. **PyTorch Framework Hygiene**: Enforce `set_seed()`, device-agnostic execution, `weights_only=True`, and VRAM targets ([rules/PYTORCH_FRAMEWORK_RULES.md](rules/PYTORCH_FRAMEWORK_RULES.md)).
5. **Dual-Paradigm Layout & Colocation**: In this multi-track repository, colocate lab-specific code, tests, and documentation under `src/<lab>/`, `tests/<lab>/`, and `docs/<lab>/` ([rules/FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md)).
