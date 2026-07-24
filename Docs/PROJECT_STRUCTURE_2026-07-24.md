# Project Structure — Deep Learning Course

*Generated: 2026-07-24*

## Overview

A scaffolded educational project for learning deep learning. Currently in an early template stage — most directories are empty, awaiting course exercises and source code.

## Directory Tree

```
Deeplearning_Course/
│
├── README.MD                           # Setup & installation instructions
├── requirements.txt                    # Python dependencies
├── scripts/                            # Executable scripts, named YYYY-MM-DD_desc.py
│
├── agents/                             # AI agent reference docs & task definitions
│   ├── ML_PIPELINE_REFERENCE_v3.md     # Comprehensive ML pipeline guide (2500+ lines, 21 steps)
│   ├── easay/
│   │   └── DEFIND_PROBLEMS_EASAY.md    # (empty) — essay assignment placeholder
│   └── labs/
│       └── DEFIND_PROBLEMS_LAB1.md     # (empty) — Lab 1 assignment placeholder
│
├── co-work/                            # (empty) — collaborative work files
├── config/                             # (empty) — configuration files
├── Cource_materials/                   # Course materials
│   └── vibe_code_rules.png             # Coding rules / guidelines image
├── data/                               # (empty) — datasets
├── Docs/                               # (empty) — documentation
├── logs/                               # (empty) — log files
├── model/
│   ├── backup/                         # (empty) — model weight backups / checkpoints
│   └── baseline/                       # (empty) — baseline model files
└── src/                                # Source modules (reusable imports)
```

## Key Files

| File | Purpose |
|------|---------|
| `README.MD` | Instructions: create `.venv`, install PyTorch (CUDA 13.0), install `requirements.txt` |
| `requirements.txt` | Dependencies: `pandas`, `numpy`, `tabulate`, `matplotlib` |
| `agents/ML_PIPELINE_REFERENCE_v3.md` | 21-step ML/DL pipeline reference guide with code snippets, decision trees, and EDA coverage |

## Technology Stack

- **Python** (3.8+) — primary language
- **PyTorch** (CUDA 13.0) — deep learning framework
- **pandas, numpy** — data manipulation
- **matplotlib, tabulate** — visualization & display
- **scikit-learn, imbalanced-learn** — referenced in pipeline guide (not yet in `requirements.txt`)

## Naming Convention

| Directory | Convention | Example |
|-----------|------------|---------|
| `scripts/` | `YYYY-MM-DD_desc.py` | `2026-07-24_exploratory_analysis.py` |
| `src/` | Standard Python modules (reusable) | `data_loader.py`, `model.py` |
| `model/backup/` | `YYYY-MM-DD_model-name.pt` | `2026-07-24_resnet18.pt` |
| `docs/` | Descriptive PascalCase | `PROJECT_STRUCTURE.md` |
| `config/` | Descriptive lowercase | `training_params.yaml` |

## State Notes

- **No source code yet** (all code directories are empty)
- **No version control** (no `.git/` directory or `.gitignore`)
- **6 files** contain content across the entire project (4 text files, 1 image, 2 empty placeholders)
- Most directories are empty scaffolded placeholders
