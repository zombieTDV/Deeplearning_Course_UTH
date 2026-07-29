# Complete Project Skeleton & Directory Structure (Template)

*Updated: 2026-07-29*

This project skeleton implements the **10-Step Vibe Coding Workflow** specified in [`agents/WORKFLOW.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/WORKFLOW.md). All specifications are generic templates ready for any Data Science / Deep Learning dataset.

```text
Deeplearning_Course_UTH/
│
├── agents/                             # ★ Section 1 & 2: AI Knowledge Store & Prompt Specs
│   ├── WORKFLOW.md                     # Master 10-step Vibe Coding Workflow (Source of Truth)
│   ├── AI_REFERENCE.md                 # Section 6: Naming conventions & folder standards (`models/`)
│   ├── PROGRESS_TRACKING.md            # Section 4: Live Kanban task board (To Do / In Progress / Done)
│   ├── PROJECT_STRUCTURE.md            # Complete directory structure reference (Template)
│   │
│   ├── knowledge_store/                # Section 1.1: Context Management & Knowledge Store
│   │   ├── README.md                   # Context lookup guide
│   │   ├── data_schema_spec.md         # Template: Data shapes, normalization parameters, labels
│   │   └── model_architecture_spec.md  # Template: Network topology, dimensions, loss functions
│   │
│   ├── prompts/                        # Section 2 & 5: Behavioral Layer Prompting Specs
│   │   ├── prompt_spec_template.md     # Base prompt spec template
│   │   ├── data_prep/                  # Data preparation pipeline prompt spec template
│   │   ├── model/                     # Model architecture pipeline prompt spec template
│   │   ├── training/                  # Training pipeline prompt spec template
│   │   └── evaluation/                # Evaluation pipeline prompt spec template
│   │
│   └── templates/                      # Section 3 & 7: Task Description & Audit Templates
│       ├── task_description_template.md # Task specification template
│       └── codebase_audit_checklist.md  # Architecture, plan alignment & style audit
│
├── data/                               # Section 5: Pipeline 1 - Data Preparation
│   ├── raw/                            # Original raw dataset files
│   └── processed/                      # Preprocessed DataLoaders / tensors
│
├── models/                             # Section 5: Pipeline 2 - Model Architectures & Checkpoints
│   ├── __init__.py                     # PyTorch nn.Module architecture classes
│   ├── backup/                         # Model weight backups & checkpoints
│   └── baseline/                       # Baseline model checkpoints
│
├── training/                           # Section 5: Pipeline 3 - Training Pipeline
│   └── __init__.py                     # Training loops, loss functions, optimizers
│
├── evaluation/                         # Section 5: Pipeline 4 - Evaluation Pipeline
│   └── __init__.py                     # Metrics calculation, plot generation, report export
│
├── configs/                            # Section 6: Configurations
│   └── default_config.yaml             # YAML/JSON hyperparameters & experiment configs
│
├── scripts/                            # Section 6: Entrypoints & Execution Scripts
│   ├── train.py                        # Main training entrypoint
│   └── evaluate.py                     # Main evaluation entrypoint
│
├── tests/                              # Section 9: Two Testing Modes
│   ├── smoke_test.py                   # Tier-1: Fast Smoke Test (< 10s instant check)
│   ├── normal_test.py                  # Tier-2: Full Normal Test suite (Integration & Metrics)
│   └── unit/                           # Module unit tests
│       └── __init__.py
│
├── Cource_materials/                   # Course reference materials
├── .github/                            # CI/CD & GitHub templates
├── README.MD
├── requirements.txt
├── CHANGELOG.md
└── CONTRIBUTING.md
```

## Quick Start (Vibe Coding Loop)
1. **Define Task**: Copy `agents/templates/task_description_template.md` → update `agents/PROGRESS_TRACKING.md`.
2. **Configure Knowledge Store**: Fill in dataset & model specs in `agents/knowledge_store/`.
3. **Select Prompt Spec**: Pick matching prompt template from `agents/prompts/<pipeline>/`.
4. **Generate Code**: LLM generates modular code adhering to `agents/AI_REFERENCE.md`.
5. **Audit Code**: Run `agents/templates/codebase_audit_checklist.md`.
6. **Test Code**:
   - **Tier-1**: `python3 tests/smoke_test.py` (Fast < 10s check)
   - **Tier-2**: `python3 tests/normal_test.py` (Full verification)
