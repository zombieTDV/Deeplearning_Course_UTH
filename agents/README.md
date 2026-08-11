# Agent AI Knowledge Base & Control Layer

Welcome to the **Agent AI Knowledge Base** for the `Deeplearning_Course_UTH` repository. This folder serves as the "second brain", memory, and behavioral control layer for AI agents working on this project.

---

## 🏗️ Architecture Overview (LAB2)

`agents/` governs a layered ML pipeline; each layer owns code, docs, and status:

```
Data (src/data) → Models (src/models) → Training scripts (src/training, script-only)
    → Evaluation (src/eval) → Analysis notebooks (notebooks/, test/demo/visualize only)
                ↕ artifacts (experiments/runs|results) ↕
Governance (this folder): rules → phases → progress → experiments → bugs
```

- **rules/** — binding conventions the agent must follow: naming, folder
  structure, MD format, notebook policy, **logging & checkpoints** (naming,
  format, resume procedure), **5W1H result reporting**, codebase audit.
- **phases/** — how each pipeline stage works (data prep → model → training → eval).
- **progress/** — live status of each phase (`*_STATUS.md`).
- **experiments/** — experiment plans, SOTA benchmarks, and results with 5W1H.
- **bugs/** — resolved runtime issues (BUG-01, BUG-02).
- **templates/** — reusable skeletons (phase doc, progress status, audit, smoke test).
- **references/** — external technical guides.

Training runs **only from scripts** (`src/training/train_lab2_models.py`,
`src/experiments/*_train.py`); notebooks analyze artifacts. Logging/checkpoint
rules: `rules/LOGGING_CHECKPOINT_RULES.md`. Result reporting (5W1H):
`rules/RESULTS_REPORTING.md`.

---

## 📁 Directory Architecture

```
agents/
├── README.md                      # Entry point & navigation guide (this file)
├── OVERVIEW.md                    # Core project overview, goals & phase roadmap
├── PURPOSE.md                     # Original exercise brief & requirements
├── codebase-audit-report.md       # LAB2 audit baseline & prioritized action plan
│
├── experiments/                   # Experiment reports & technical comparisons
│   ├── SUMMARY_RESULTS.md         # Final comparison report (5W1H formatted)
│   ├── PRACTICE2_SOTA_BENCHMARK_DOCUMENTATION.md
│   ├── CODE_DIFFERENCE_PRACTICE2_VS_EXP07.md
│   ├── MOE_EXPERIMENT.md, LOGIT_BIAS_SWEEP_STATUS.md, EXP_01..07, ...
├── rules/                         # Guidelines, standards & philosophy for Agent AI
│   ├── AGENT_AI.md                # Agent AI philosophy & behavior layer rules
│   ├── CODEBASE_AUDIT.md          # Mandatory pre-task codebase audit checklist
│   ├── FOLDER_STRUCTURE.md        # Source of truth for repository directory layout
│   ├── MD_CONVENTION.md           # Documentation format standards
│   ├── NAMING_CONVENTION.md       # Naming rules for files, code & experiments
│   ├── NOTEBOOK_HEADER_CONVENTION.md # Standardized headers for Jupyter notebooks
│   ├── LOGGING_CHECKPOINT_RULES.md  # Logging, checkpoint format, resume procedure
│   └── RESULTS_REPORTING.md         # 5W1H rules for every reported result
│
├── phases/                        # Step-by-step pipeline & phase specifications
│   ├── DATA_PREP.md, DATASET.md, DATALOADER.md, INSPECTION.md, STATISTICS.md,
│   ├── TRANSFORMS.md, MODEL.md, TRAINING_INFO.md, EVAL.md, ERROR_ANALYSIS.md
│   └── PRACTICE2_EXP07_UPGRADE_PLAN.md, PRACTICE2_EARLYSTOPPING_ERROR_ANALYSIS_PLAN.md
│
├── templates/                     # Standard templates & checklists for agents
│   ├── PHASE_DOC_TEMPLATE.md      # Template for writing new phase documentation
│   ├── PROGRESS_STATUS_TEMPLATE.md# Template for tracking task status
│   ├── CODEBASE_AUDIT_TEMPLATE.md # Reusable audit report skeleton
│   └── SMOKE_TEST_CHECKLIST.md    # Pre-run smoke test verification checklist
│
├── progress/                      # Live progress tracking per phase
│   ├── DATA_PREP_STATUS.md, MODEL_STATUS.md, TRAINING_STATUS.md, EVAL_STATUS.md
│   └── MOE_STATUS.md, ERROR_ANALYSIS.md
│
├── bugs/                          # Documented bug reports
│   ├── BUG_01_DATALOADER_BROKEN_PIPE_PYTHON314.md
│   └── BUG_02_PRACTICE2_TRAIN_MODEL_TEST_LOSSES_KEYERROR.md
│
└── references/                    # External guides & technical reference notes
    ├── OPTUNA_DB_GUIDE.md         # Guide for Optuna hyperparameter tracking DB
    └── GIT_AND_RELEASE_BEST_PRACTICES.md  # Git commits, CI, releases & approval gate
```

---

## 🚀 Guidelines for AI Agents

1. **Before starting a multi-file task**: Run the procedure in [rules/CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md) to check for drift between documentation and real code.
2. **Before creating or modifying code**: Verify directory locations in [rules/FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) and naming rules in [rules/NAMING_CONVENTION.md](rules/NAMING_CONVENTION.md).
3. **Executing a Phase**: Read the matching phase doc in `phases/<phase>.md` and maintain status in `progress/<phase>_STATUS.md`.
4. **Before long runs**: Perform the verification steps in [templates/SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md).
