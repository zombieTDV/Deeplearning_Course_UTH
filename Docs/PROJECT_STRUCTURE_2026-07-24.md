# Project Structure — Deep Learning Course

*Generated: 2026-07-24*

## Overview

A scaffolded educational project for deep learning focused on a **PyTorch FashionMNIST Classification lab** (Lab 1). The project uses a **Git worktree** team workflow with 6 member branches under `LAB1-FashionMNIST-Classification/` for parallel development. Most content directories are scaffolded placeholders awaiting course exercises.

## Directory Tree

```
Deeplearning_Course/
│
├── .git/                               # Git repository (active, version-controlled)
├── .gitignore                          # Ignored: venv, __pycache__, data/, model/backup, lab results, logs
├── README.MD                           # Setup & installation instructions
├── requirements.txt                    # Python dependencies
│
├── agents/                             # AI agent reference docs & task definitions
│   ├── GIT-WORKING-GUIDE.md            # Git workflow guide for AI agent collaboration
│   ├── easay/
│   │   └── DEFIND_PROBLEMS_EASAY.md    # (empty) — essay assignment placeholder
│   └── labs1/
│       ├── DEFIND_PROBLEMS_LAB1.md     # Lab 1 assignment — FashionMNIST Classification (1056 lines)
│       └── Excercise.png               # Lab 1 exercise image (~478 KB)
│
├── co-work/                            # (empty) — collaborative work files
├── config/                             # (empty) — configuration files
├── Cource_materials/                   # Course materials
│   └── vibe_code_rules.png             # Coding rules / guidelines image (~323 KB)
├── data/                               # (empty) — datasets
├── Docs/                               # Documentation
│   ├── LAB1_TEAM_CONTRIBUTION.md       # Lab 1 team contribution tracker
│   └── PROJECT_STRUCTURE_2026-07-24.md # This file
├── labs/                               # Lab work
│   └── lab1/
│       ├── results/
│       │   ├── logs/                   # (empty) — training logs
│       │   ├── metrics/                # (empty) — evaluation metrics
│       │   ├── models/                 # (empty) — trained model checkpoints
│       │   └── plots/                  # (empty) — result visualizations
│       └── scripts/                    # (empty) — lab 1 scripts
├── logs/                               # (empty) — log files
├── model/
│   ├── backup/                         # (empty) — model weight backups / checkpoints
│   └── baseline/                       # (empty) — baseline model files
└── src/                                # Source modules (reusable imports)
    └── __init__.py                     # (empty) — package init placeholder
```

## Git Worktree — Lab 1 Team Branches

The project uses a **Git worktree** for team collaboration on Lab 1. The worktree `LAB1-FashionMNIST-Classification` contains 6 feature branches for parallel development:

| Branch | Role |
|--------|------|
| `main` | Shared integration / base branch |
| `member1-data-eda` | Data loading & exploratory analysis |
| `member2-baseline` | Baseline model implementation |
| `member3-mlp-arch` | MLP architecture development |
| `member4-experiments` | Hyperparameter experiments |
| `member5-evaluation` | Model evaluation & metrics |
| `member6-report` | Final report & documentation |

## Key Files

| File | Purpose |
|------|---------|
| `README.MD` | Virtual environment setup, PyTorch (CUDA 13.0) install, dependency install |
| `requirements.txt` | Dependencies: `pandas`, `numpy`, `tabulate`, `matplotlib` |
| `.gitignore` | 52 rules excluding venv, cache, data, model weights, lab results |
| `agents/GIT-WORKING-GUIDE.md` | 78-line Git workflow guide for AI agent collaboration |
| `agents/labs1/DEFIND_PROBLEMS_LAB1.md` | Lab 1 full assignment — FashionMNIST classification (1056 lines) |
| `Docs/LAB1_TEAM_CONTRIBUTION.md` | Lab 1 team role tracking & submission checklist |

## Technology Stack

- **Python** (3.8+) — primary language
- **PyTorch** (CUDA 13.0) — deep learning framework
- **pandas, numpy** — data manipulation
- **matplotlib, tabulate** — visualization & display
- **scikit-learn, imbalanced-learn** — referenced in pipeline guide (not yet in `requirements.txt`)

## Naming Convention

| Directory | Convention | Example |
|-----------|------------|---------|
| `agents/` | Descriptive PascalCase | `GIT-WORKING-GUIDE.md` |
| `labs/lab1/scripts/` | Descriptive snake_case | `train.py`, `evaluate.py` |
| `src/` | Standard Python modules | `data_loader.py`, `model.py` |
| `model/backup/` | `YYYY-MM-DD_model-name.pt` | `2026-07-24_resnet18.pt` |
| `Docs/` | Descriptive PascalCase | `PROJECT_STRUCTURE.md` |
| `config/` | Descriptive lowercase | `training_params.yaml` |

## State Notes

- **Active Git repository** with `.gitignore` (52 rules)
- **Git worktree** `LAB1-FashionMNIST-Classification` with 6 team branches
- **Lab 1 in progress** — assignment defined, team roles assigned, no code committed yet
- **No source code yet** — `src/` contains only an empty `__init__.py`
- **10 files** contain content across the project (6 markdown, 2 images, 1 requirements, 1 gitignore)
- Most directories are empty scaffolded placeholders for future work
