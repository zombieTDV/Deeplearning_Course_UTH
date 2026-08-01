# FOLDER_STRUCTURE.md

Source of truth for where things live. The agent should check this
before creating any new file, and update it (with human approval)
if the structure changes.

```
project_root/
├── agent/                 # Agent AI Knowledge Base (behavior layer, rules, plans, progress)
│   ├── README.md          # Index & guide for Agent AI
│   ├── OVERVIEW.md        # Core project overview & roadmap
│   ├── PURPOSE.md         # Original exercise brief & requirements
│   ├── rules/             # Guidelines & standards for Agent AI
│   │   ├── AGENT_AI.md    # Agent AI philosophy & behavior rules
│   │   ├── CODEBASE_AUDIT.md # Codebase audit procedure
│   │   ├── FOLDER_STRUCTURE.md
│   │   ├── MD_CONVENTION.md
│   │   ├── NAMING_CONVENTION.md
│   │   └── NOTEBOOK_HEADER_CONVENTION.md
│   ├── phases/            # Phase & pipeline documentation
│   │   ├── DATA_PREP.md
│   │   ├── DATASET.md
│   │   ├── DATALOADER.md
│   │   ├── INSPECTION.md
│   │   ├── STATISTICS.md
│   │   ├── TRANSFORMS.md
│   │   ├── MODEL.md
│   │   ├── TRAINING_INFO.md
│   │   └── EVAL.md
│   ├── templates/         # Document & checklist templates
│   │   ├── PHASE_DOC_TEMPLATE.md
│   │   ├── PROGRESS_STATUS_TEMPLATE.md
│   │   └── SMOKE_TEST_CHECKLIST.md
│   ├── references/        # External guides & reference docs
│   │   └── OPTUNA_DB_GUIDE.md
│   └── progress/          # One status file per task/phase
│       ├── DATA_PREP_STATUS.md
│       ├── MODEL_STATUS.md
│       ├── TRAINING_STATUS.md
│       └── EVAL_STATUS.md
├── data/
│   ├── raw/               # never edited by the agent
│   ├── processed/
│   └── external/
├── src/
│   ├── data/              # loading, cleaning, feature eng
│   ├── models/            # model definitions
│   ├── training/          # training loops
│   ├── eval/              # metrics, evaluation scripts
│   └── utils/
├── notebooks/             # exploratory only, not production code
├── configs/
├── experiments/           # run outputs, checkpoints, logs (gitignored; created at first run)
└── tests/                 # smoke tests + unit tests
```

## Rules

- Agent must not create top-level folders without flagging it first
- Anything in `data/raw/` is read-only — never written to by any script
- Exploratory/throwaway code stays in `notebooks/`, not `src/`
- If actual folder structure diverges from this file, that's a
  CODEBASE_AUDIT.md finding, not something to silently "fix"
- Before working on a phase, read the matching agent/phases/<phase>.md and
  agent/progress/<phase>_STATUS.md yourself before asking the human for context.
