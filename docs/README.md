# Coursework Research Documentation (`/docs`)

- **Motivation/Background**: This directory consolidates all evolving research knowledge, stage specifications, experiment plans, and reference manuals across all three coursework laboratories.
- **Purpose**: Provide a structured, stage-by-stage memory of research evolution that is clean, reproducible, and verifiable.
- **Overview Pipeline**: Organized by research stage boundaries (LAB1, LAB2, LAB3), with cross-project SOPs under `shared/` and early drafts preserved under `archive/`.
- **Detailed Plan**: §1 Architecture; §2 Stage Navigation; §3 Shared Manuals.
- **References**: `agents/rules/FOLDER_STRUCTURE.md`, `docs/shared/`.
- **Created**: 2026-09-06T13:05:18+07:00
- **Last Updated**: 2026-09-06T21:25:00+07:00

---

## 1. Documentation Structure

This repository implements the **Dual-Paradigm Architecture (Archetype B: Multi-Track Coursework)**.

- **Global / Shared (`docs/`, `docs/shared/`):** Master research index, inter-agent handoff templates, universal engineering SOPs, and cross-lab consolidation audit reports.
- **Colocated Track Documentation (`docs/lab1/`, `docs/lab2/`, `docs/lab3/`):** Each laboratory maintains its own phase specifications (`phases/`), live progress trackers (`progress/`), empirical experiment writeups (`experiments/`), and defect analyses (`bugs/`), preventing cross-domain clutter.

```text
docs/
├── README.md                          # This file (master research index)
├── shared/                            # Cross-project reference guides, SOPs, and manuals
│   ├── HOW_TO_SETUP_AI_AGENT.md       # 10-step agent workflow setup SOP
│   ├── HANDOFF_TEMPLATE.md            # Inter-agent task handoff specification
│   ├── ML_PIPELINE_REFERENCE_v3.md    # 18-step ML engineering reference guide
│   ├── MD_CREATION_GUIDE.md           # 5-step pedagogical documentation guide
│   ├── OPTUNA_DB_GUIDE.md             # Optuna SQLite analysis and export guide
│   ├── GIT_AND_RELEASE_BEST_PRACTICES.md # Git commits, CI, and release management SOP
│   ├── FINAL_CONSOLIDATION_AUDIT_REPORT.md  # Comprehensive consolidation readiness audit
│   ├── LAB2_LAB3_EXPERIMENT_AUDIT_REPORT.md # Experiment lineage & artifact integrity audit
│   └── RUNS_PATH_AUDIT_REPORT.md      # Runtime path verification report

├── archive/                           # Preserved historical drafts
│   └── lab1/                          # LAB1 early rulebase and notes
├── lab1/                              # Fashion-MNIST Classification Stage
│   ├── README.md                      # Overview of LAB1 research & deliverables
│   ├── PURPOSE.md                     # Original brief (from purposre.md)
│   ├── GIT_WORKING_GUIDE.md           # Sub-branch collaboration protocol
│   └── experiments/README.md          # Index pointing to practice_1 outputs
├── lab2/                              # CIFAR-10 Transfer Learning Stage
│   ├── README.md                      # Overview of CIFAR-10 benchmarks & SOTA
│   ├── OVERVIEW.md                    # LAB2 execution plan
│   ├── PURPOSE.md                     # LAB2 exercise brief & requirements
│   ├── CODEBASE_AUDIT_REPORT.md       # LAB2 historical audit baseline
│   ├── experiments/                   # EXP-01 to EXP-07, logit bias sweep, SOTA docs
│   ├── phases/                        # 13 CIFAR-10 pipeline phase specifications
│   ├── progress/                      # CIFAR-10 live phase status tracking
│   └── bugs/                          # Bugs 01-02
└── lab3/                              # Hugging Face IMDB Sentiment Stage
    ├── README.md                      # Overview of NLP, LoRA, and Cleanlab breakthroughs
    ├── OVERVIEW.md                    # LAB3 execution plan
    ├── PURPOSE.md                     # LAB3 exercise brief & requirements
    ├── PROJECT_ROADMAP.md             # LAB3 milestone execution roadmap
    ├── CODEBASE_AUDIT_REPORT.md       # LAB3 historical audit baseline
    ├── experiments/                   # EX1 to EX14 reports (baseline -> LoRA peak)
    ├── phases/                        # 8 NLP pipeline phase specifications
    ├── plans/                         # Active and completed anti-overfitting plans
    ├── progress/                      # LAB3 live phase status tracking
    └── bugs/                          # Bugs 01-05
```

---

## 2. Research Stages Overview

* **LAB1 (Fashion-MNIST)**: Comparison of Multi-Layer Perceptrons (MLPs) vs. Convolutional Neural Networks (CNNs), systematic Optuna hyperparameter optimization, data augmentation, and error analysis.
* **LAB2 (CIFAR-10 Transfer Learning)**: Fine-tuning torchvision backbones (ResNet18, DenseNet121, ConvNeXt-Tiny), Layer-wise Learning Rate Decay (LLRD), RandAugment, Class-Logit Bias Sweeping, Soft-Voting Ensembles, and UMAP deep feature inspection.
* **LAB3 (Hugging Face NLP Sentiment Analysis)**: Transformer sentiment classification using `distilbert-base-uncased`, data-centric AI with 5-Fold OOF Cleanlab label auditing, Head+Tail sequence truncation, and PEFT LoRA adaptation.
