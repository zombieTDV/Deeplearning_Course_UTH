# folder_structure.md

Source of truth for where things live. The agent should check this
before creating any new file, and update it (with human approval)
if the structure changes.

```
project_root/
├── docs/                  # all .md files (behavior layer, plans, progress)
│   ├── overview.md
│   ├── data_prep.md
│   ├── training_info.md
│   ├── model.md
│   ├── eval.md
│   ├── naming_convention.md
│   ├── folder_structure.md
│   ├── codebase_audit.md
│   └── progress/          # one status file per task/phase
│       ├── data_prep_status.md
│       └── training_status.md
├── data/
│   ├── raw/                # never edited by the agent
│   ├── processed/
│   └── external/
├── src/
│   ├── data/                # loading, cleaning, feature eng
│   ├── models/               # model definitions
│   ├── training/              # training loops
│   ├── eval/                  # metrics, evaluation scripts
│   └── utils/
├── notebooks/                 # exploratory only, not production code
├── configs/
├── experiments/                # run outputs, checkpoints, logs (gitignored)
└── tests/                      # smoke tests + unit tests
```

## Rules

- Agent must not create top-level folders without flagging it first
- Anything in `data/raw/` is read-only — never written to by any script
- Exploratory/throwaway code stays in `notebooks/`, not `src/`
- If actual folder structure diverges from this file, that's a
  codebase_audit.md finding, not something to silently "fix"
- Before working on a phase, read the matching docs/<phase></phase>.md and
  docs/progress/<phase></phase>_status.md yourself before asking the human for context.
