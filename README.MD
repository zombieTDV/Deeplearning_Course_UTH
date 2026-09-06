# Deep Learning Coursework — Consolidated Repository

[![CI](https://github.com/zombieTDV/Deeplearning_Course_UTH/actions/workflows/ci.yml/badge.svg)](https://github.com/zombieTDV/Deeplearning_Course_UTH/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A unified deep learning coursework and research portfolio covering computer vision, transfer learning, ensemble modeling, and natural language processing.

---

## 📚 Curriculum & Coursework Stages

| Stage | Domain / Task | Core Methodology & Highlights | Code & Notebooks | Docs |
| :--- | :--- | :--- | :--- | :--- |
| **LAB 1** | **Fashion-MNIST Classification** | Baseline MLP vs. CNN comparison, data augmentation, systematic Optuna hyperparameter optimization, and error analysis. | [`src/lab1/`](src/lab1/)<br>[`notebooks/lab1/`](notebooks/lab1/) | [`docs/lab1/`](docs/lab1/) |
| **LAB 2** | **CIFAR-10 Transfer Learning** | Fine-tuning torchvision backbones (ResNet-18, DenseNet-121, ConvNeXt-Tiny), Layer-wise Learning Rate Decay (LLRD), RandAugment, Class-Logit Bias Sweeping, Soft-Voting Ensembles, and UMAP deep feature inspection (95.42% SOTA). | [`src/lab2/`](src/lab2/)<br>[`notebooks/lab2/`](notebooks/lab2/) | [`docs/lab2/`](docs/lab2/) |
| **LAB 3** | **Hugging Face Sentiment Analysis** | Transformer sentiment classification using `distilbert-base-uncased`, data-centric AI with 5-Fold OOF Cleanlab label noise auditing, Head+Tail sequence truncation, Stochastic Weight Averaging (SWA), and PEFT LoRA adaptation. | [`src/lab3/`](src/lab3/)<br>[`notebooks/lab3/`](notebooks/lab3/) | [`docs/lab3/`](docs/lab3/) |

---

## 📂 Project Architecture

The repository enforces strict domain namespacing across source code, notebooks, experiments, tests, and documentation:

```text
Deeplearning_Course/
├── .github/workflows/ci.yml  # GitHub Actions CI matrix (multi-lab CPU testing & linting)
├── agents/                   # Constitutional governance (universal rules & templates)
├── configs/                  # Lab configuration files (YAML)
├── data/                     # Namespaced data directories (raw/processed/splits)
├── docs/                     # Research specifications, experiment reports & SOPs
│   ├── shared/               # Universal SOPs and consolidation audit records
│   ├── lab1/                 # LAB1 documentation & reports
│   ├── lab2/                 # LAB2 documentation & experiment logs
│   └── lab3/                 # LAB3 documentation & phase reports
├── experiments/              # Experiment artifacts, checkpoints, and metrics
│   ├── lab1/                 # LAB1 experiment outputs
│   ├── lab2/runs/            # LAB2 namespaced run directories
│   └── lab3/runs/            # LAB3 namespaced run directories
├── notebooks/                # Interactive analysis & pedagogical walkthroughs
│   ├── lab1/                 # Fashion-MNIST notebooks
│   ├── lab2/                 # CIFAR-10 transfer learning notebooks
│   └── lab3/                 # Hugging Face NLP sentiment notebooks
├── pyproject.toml            # Canonical PEP 517/518 build & tool configuration
├── requirements/             # Multi-tier dependency specifications
│   ├── base.txt              # Core scientific & deep learning stack
│   ├── lab1.txt              # LAB1 dependencies
│   ├── lab2.txt              # LAB2 dependencies
│   ├── lab3.txt              # LAB3 NLP, LoRA & Cleanlab dependencies
│   └── dev.txt               # Complete developer environment (pytest, ruff)
├── requirements.txt          # Root developer proxy (-r requirements/dev.txt)
├── src/                      # Namespaced source packages
│   ├── lab1/                 # src.lab1.*
│   ├── lab2/                 # src.lab2.*
│   └── lab3/                 # src.lab3.*
└── tests/                    # Comprehensive unit & integration test suites
    ├── lab1/                 # tests/lab1/
    ├── lab2/                 # tests/lab2/
    └── lab3/                 # tests/lab3/
```

---

## ⚙️ Installation & Setup

### 1. Environment Setup

Create and activate a virtual environment using Python 3.12 (recommended) or 3.11:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Dependency Installation

#### Standard / CPU Workstations & CI:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### NVIDIA GPU Workstations (CUDA 13.0 / PyTorch wheels):
```bash
pip install --upgrade pip
pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
```

---

## 🧪 Verification & Testing

Execute the complete test battery across all three laboratories:

```bash
# Run all tests (CPU compatible)
pytest tests/lab1 tests/lab2 tests/lab3 -v

# Run tests with GPU markers excluded (ideal for headless CPU environments)
pytest -m "not gpu"

# Run code linting on maintained modules
ruff check --select E,F,W --ignore E501,E402 --exclude "src/lab2/experiments,src/lab3/experiments" src/lab2 src/lab3
```

---

## 📜 Governance & References

- [Constitutional Rules & Templates](agents/README.md)
- [Folder Structure Specification](agents/rules/FOLDER_STRUCTURE.md)
- [Master Research Documentation](docs/README.md)
- [Consolidation Handoff Specification](CONSOLIDATION_HANDOFF.md)
