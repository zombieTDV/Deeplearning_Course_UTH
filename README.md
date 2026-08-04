# Deep Learning Course - LAB2: Computer Vision & Transfer Learning

## Project Overview

This project implements a modular, production-ready data pipeline and deep learning framework for computer vision tasks, specifically tailored for CIFAR-10 classification using pretrained PyTorch architectures (ResNet, DenseNet). The architecture strictly adheres to the Separation of Concerns (SoC) principle, dividing data processing, model building, training, evaluation, and AI Agent governance into decoupled layers.

---

## 🏗️ System Architecture

The overall pipeline flow is structured as follows:

```
Dataset → Inspection → Statistics → Transforms → DataLoader → Models → Training → Evaluation
                                                                             │
                                                                 Agent AI Knowledge Base (agents/)
```

---

## 📁 Repository Structure

```
Deeplearning_Course_UTH/
│
├── agents/                   # Agent AI Knowledge Base & Behavioral Control Layer
│   ├── README.md             # Navigation guide & entry point for AI Agent
│   ├── OVERVIEW.md           # Project roadmap, goals & phase overview
│   ├── PURPOSE.md            # Exercise brief & technical objectives
│   │
│   ├── rules/                # System conventions, rules & AI philosophy
│   │   ├── AGENT_AI.md       # Agent AI behavior, guidelines & prompt layer
│   │   ├── CODEBASE_AUDIT.md # Pre-task codebase audit checklist
│   │   ├── FOLDER_STRUCTURE.md # Repository directory source of truth
│   │   ├── MD_CONVENTION.md  # Documentation formatting standards
│   │   ├── NAMING_CONVENTION.md # Naming conventions for code & files
│   │   └── NOTEBOOK_HEADER_CONVENTION.md # Jupyter notebook header standard
│   │
│   ├── phases/               # Stage-specific pipeline documentation
│   │   ├── DATA_PREP.md      # Data preparation & split policy
│   │   ├── DATASET.md        # Dataset specifications & loaders
│   │   ├── DATALOADER.md     # DataLoader creation & optimization
│   │   ├── INSPECTION.md     # Data validation & quality checks
│   │   ├── STATISTICS.md     # Dataset statistics computation
│   │   ├── TRANSFORMS.md     # Augmentation & transform pipelines
│   │   ├── MODEL.md          # Pretrained ResNet/DenseNet adaptation
│   │   ├── TRAINING_INFO.md  # Training loop & hyperparameter sweeps
│   │   └── EVAL.md           # Test evaluation & metrics reporting
│   │
│   ├── templates/            # Standard templates & checklists
│   │   ├── PHASE_DOC_TEMPLATE.md
│   │   ├── PROGRESS_STATUS_TEMPLATE.md
│   │   └── SMOKE_TEST_CHECKLIST.md
│   │
│   ├── progress/             # Live progress status per phase
│   │   ├── DATA_PREP_STATUS.md
│   │   ├── MODEL_STATUS.md
│   │   ├── TRAINING_STATUS.md
│   │   └── EVAL_STATUS.md
│   │
│   ├── experiments/          # Experiment reports, status & SOTA benchmarks
│   │   ├── SUMMARY_RESULTS.md # Final experiment comparison report
│   │   └── LOGIT_BIAS_SWEEP_STATUS.md # Decision threshold optimization status
│   │
│   └── references/           # External guides & technical documentation
│       └── OPTUNA_DB_GUIDE.md
│
├── configs/
│   └── data.yaml             # Centralized configuration for data pipeline
│
├── data/                     # Primary Data Pipeline (Single Source of Truth)
│   ├── external/             # Raw dataset storage (CIFAR-10)
│   ├── processed/            # Processed data, statistics & persisted split
│   ├── dataset.py            # Dataset loading & split handling
│   ├── inspection.py         # Quality checks & label verification
│   ├── statistics.py         # Mean & Std computation
│   ├── transforms.py         # Training & evaluation transform pipelines
│   ├── dataloader.py         # DataLoader builder functions
│   └── config.py             # Configuration loader
│
├── src/                      # Core Deep Learning Modules
│   ├── models/               # Pretrained model setup (ResNet18, DenseNet121)
│   ├── training/             # Training loop, optimizers & TensorBoard logging
│   ├── eval/                 # Test metrics, confusion matrix & reporting
│   ├── scratch/              # Notebook builders & temporary test scripts
│   ├── experiments/          # Python experiment execution scripts
│   └── utils/                # Helper utilities
│
├── notebooks/                # Exploratory & Deliverable Jupyter Notebooks
│   ├── practice_2.ipynb      # Main deliverable notebook
│   └── practice_2_logit_bias_sweep.ipynb # Logit bias & threshold sweep notebook
│
├── experiments/              # Run outputs, checkpoints, plots & results JSON
│
├── tests/                    # Unit tests & smoke tests
├── requirements.txt          # Dependency requirements
└── README.md                 # Project root documentation
```

---

## 🤖 Agent AI Control & Knowledge Base

The repository includes a dedicated AI Agent system in `agents/`:

- **System Conventions**: All configuration files for the AI Agent use uppercase names with lowercase `.md` extension (`README.md`, `AGENT_AI.md`, `CODEBASE_AUDIT.md`) for instant recognition and visual highlighting.
- **Behavior Layer**: [agents/rules/AGENT_AI.md](agents/rules/AGENT_AI.md) acts as the memory and second brain for AI coding assistants.
- **Audit Procedure**: Before multi-file tasks, the AI agent executes [agents/rules/CODEBASE_AUDIT.md](agents/rules/CODEBASE_AUDIT.md) to prevent documentation drift.
- **Phase Execution**: Pipeline stages are documented in [agents/phases/](agents/phases/) and tracked live in [agents/progress/](agents/progress/).

---

## ⚙️ Data Pipeline Workflow

1. **Dataset** (`data/dataset.py`): Download CIFAR-10 & load dataset splits.
2. **Inspection** (`data/inspection.py`): Validate data integrity & class balance.
3. **Statistics** (`data/statistics.py`): Compute & persist dataset mean/std.
4. **Transforms** (`data/transforms.py`): Augmentation pipeline (224x224 resize for pretrained models).
5. **DataLoader** (`data/dataloader.py`): Batching with persistent single-split policy.

---

## 🚀 Quick Start

### Installation

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate

# 2. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. Install project dependencies
pip install -r requirements.txt
```

### Running the Data Pipeline

```python
from src.data.dataset import download_cifar10
from src.data.dataloader import get_cifar10_loaders

# Download dataset (auto-downloads if missing)
download_cifar10()

# Create DataLoaders (returns train, val, test)
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)
```

### Running Tests

```bash
# Run unit tests
pytest tests/ -v
```

---

## 📜 License

This repository is developed as part of the Deep Learning Course at UTH.
