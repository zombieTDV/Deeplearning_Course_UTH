# Agent AI Knowledge Base & Control Layer

Welcome to the **Agent AI Knowledge Base** for the `Deeplearning_Course_UTH` repository. This folder serves as the "second brain", memory, and behavioral control layer for AI agents working on this project.

---

## 📁 Directory Architecture

```
agent/
├── README.md                      # Entry point & navigation guide (this file)
├── OVERVIEW.md                    # Core project overview, goals & phase roadmap
├── PURPOSE.md                     # Original exercise brief & requirements
│
├── rules/                         # Guidelines, standards & philosophy for Agent AI
│   ├── AGENT_AI.md                # Agent AI philosophy & behavior layer rules
│   ├── CODEBASE_AUDIT.md          # Mandatory pre-task codebase audit checklist
│   ├── FOLDER_STRUCTURE.md        # Source of truth for repository directory layout
│   ├── MD_CONVENTION.md           # Documentation format standards
│   ├── NAMING_CONVENTION.md       # Naming rules for files, code & experiments
│   └── NOTEBOOK_HEADER_CONVENTION.md # Standardized headers for Jupyter notebooks
│
├── phases/                        # Step-by-step pipeline & phase specifications
│   ├── DATA_PREP.md               # Data Preparation (CIFAR-10 loading & split)
│   ├── DATASET.md                 # Dataset specifications
│   ├── DATALOADER.md              # DataLoader setup & performance tuning
│   ├── INSPECTION.md              # Inspection & data validation guide
│   ├── STATISTICS.md              # Dataset statistics computation
│   ├── TRANSFORMS.md              # Transform pipelines & augmentation
│   ├── MODEL.md                   # Pretrained ResNet/DenseNet adaptation
│   ├── TRAINING_INFO.md           # Training loop & hyperparameter sweeps
│   └── EVAL.md                    # Test evaluation & comparison metrics
│
├── templates/                     # Standard templates & checklists for agents
│   ├── PHASE_DOC_TEMPLATE.md      # Template for writing new phase documentation
│   ├── PROGRESS_STATUS_TEMPLATE.md# Template for tracking task status
│   └── SMOKE_TEST_CHECKLIST.md    # Pre-run smoke test verification checklist
│
├── progress/                      # Live progress tracking per phase
│   ├── DATA_PREP_STATUS.md        # Data prep phase status [Done]
│   ├── MODEL_STATUS.md            # Model setup phase status [Done]
│   ├── TRAINING_STATUS.md         # Training phase status [Done]
│   └── EVAL_STATUS.md             # Evaluation phase status [Done]
│
└── references/                    # External guides & technical reference notes
    └── OPTUNA_DB_GUIDE.md         # Guide for Optuna hyperparameter tracking DB
```

---

## 🚀 Guidelines for AI Agents

1. **Before starting a multi-file task**: Run the procedure in [rules/CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md) to check for drift between documentation and real code.
2. **Before creating or modifying code**: Verify directory locations in [rules/FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) and naming rules in [rules/NAMING_CONVENTION.md](rules/NAMING_CONVENTION.md).
3. **Executing a Phase**: Read the matching phase doc in `phases/<phase>.md` and maintain status in `progress/<phase>_STATUS.md`.
4. **Before long runs**: Perform the verification steps in [templates/SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md).
