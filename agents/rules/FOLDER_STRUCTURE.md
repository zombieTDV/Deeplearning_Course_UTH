# FOLDER_STRUCTURE.md

Source of truth for where things live. The agent should check this before creating any new file, and update it (with human approval) if the structure changes.

```text
project_root/
├── agents/                                # CONSTITUTIONAL AI GOVERNANCE (Immutable rules & templates only)
│   ├── README.md                          # Guide to repository agent constitution
│   ├── rules/                             # What the AI agent MUST consistently do
│   │   ├── AGENT_AI.md                    # Core behavior layer & prompting rules
│   │   ├── CODEBASE_AUDIT.md              # Drift audit procedure & gate
│   │   ├── FOLDER_STRUCTURE.md            # Canonical repository directory layout (this file)
│   │   ├── LOGGING_CHECKPOINT_RULES.md    # Script-only training & full-state checkpoint rules
│   │   ├── MD_CONVENTION.md               # Markdown formatting & clickable link standards
│   │   ├── NAMING_CONVENTION.md           # File, code, and experiment naming rules
│   │   ├── NOTEBOOK_HEADER_CONVENTION.md  # Standardized notebook first-cell headers
│   │   ├── PYTORCH_FRAMEWORK_RULES.md     # PyTorch device/seed/VRAM/eval rules
│   │   └── RESULTS_REPORTING.md           # 5W1H empirical reporting protocol
│   └── templates/                         # Standardized document skeletons
│       ├── BUG_TEMPLATE.md                # Bug report skeleton
│       ├── CODEBASE_AUDIT_TEMPLATE.md     # Audit report skeleton
│       ├── EXPERIMENT_TEMPLATE.md         # Experiment report skeleton
│       ├── PHASE_DOC_TEMPLATE.md          # Pipeline phase specification skeleton
│       ├── PROGRESS_STATUS_TEMPLATE.md    # Phase progress tracking skeleton
│       ├── PROJECT_ROADMAP_TEMPLATE.md    # Milestone & execution roadmap skeleton
│       └── SMOKE_TEST_CHECKLIST.md        # Pre-execution verification checklist
│
├── docs/                                  # EVOLVING RESEARCH & STAGE DOCUMENTATION
│   ├── README.md                          # Master index of coursework research stages
│   ├── shared/                            # Cross-project reference guides, SOPs, and manuals
│   │   ├── HOW_TO_SETUP_AI_AGENT.md       # 9-step agent workflow setup SOP
│   │   ├── ML_PIPELINE_REFERENCE_v3.md    # 18-step ML engineering reference guide
│   │   ├── MD_CREATION_GUIDE.md           # 5-step pedagogical documentation guide
│   │   ├── OPTUNA_DB_GUIDE.md             # Optuna SQLite analysis and export guide
│   │   └── GIT_AND_RELEASE_BEST_PRACTICES.md # Git commits, CI, and release management SOP
│   ├── archive/                           # Preserved historical drafts
│   │   └── lab1/                          # LAB1 early rulebase and notes
│   ├── lab1/                              # Fashion-MNIST Classification Stage
│   │   ├── README.md                      # Overview of LAB1 research & deliverables
│   │   ├── PURPOSE.md                     # Original brief (renamed from purposre.md)
│   │   ├── GIT_WORKING_GUIDE.md           # Sub-branch collaboration protocol
│   │   └── experiments/README.md          # Index pointing to practice_1 outputs
│   ├── lab2/                              # CIFAR-10 Transfer Learning Stage
│   │   ├── README.md                      # Overview of CIFAR-10 benchmarks & SOTA
│   │   ├── OVERVIEW.md                    # LAB2 execution plan
│   │   ├── PURPOSE.md                     # LAB2 exercise brief & requirements
│   │   ├── CODEBASE_AUDIT_REPORT.md       # LAB2 historical audit baseline
│   │   ├── experiments/                   # EXP-01 to EXP-07, logit bias sweep, SOTA docs
│   │   ├── phases/                        # 13 CIFAR-10 pipeline phase specifications
│   │   ├── progress/                      # CIFAR-10 live phase status tracking
│   │   └── bugs/                          # Bugs 01-02
│   └── lab3/                              # Hugging Face IMDB Sentiment Stage
│       ├── README.md                      # Overview of NLP, LoRA, and Cleanlab breakthroughs
│       ├── OVERVIEW.md                    # LAB3 execution plan
│       ├── PURPOSE.md                     # LAB3 exercise brief & requirements
│       ├── PROJECT_ROADMAP.md             # LAB3 milestone execution roadmap
│       ├── CODEBASE_AUDIT_REPORT.md       # LAB3 historical audit baseline
│       ├── experiments/                   # EX1 to EX14 reports (baseline -> LoRA peak)
│       ├── phases/                        # 8 NLP pipeline phase specifications
│       ├── plans/                         # Active and completed anti-overfitting plans
│       ├── progress/                      # LAB3 live phase status tracking
│       └── bugs/                          # Bugs 01-05
│
├── src/                                   # DOMAIN-SEGREGATED SOURCE CODE
│   ├── lab1/                              # Fashion-MNIST modules (flat utilities)
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── model_utils.py
│   │   ├── train_utils.py
│   │   ├── eval_utils.py
│   │   └── vis_utils.py
│   ├── lab2/                              # CIFAR-10 Computer Vision Package
│   │   ├── __init__.py
│   │   ├── data/                          # Dataset, dataloader, transforms, inspection, statistics
│   │   ├── models/                        # build_model (ResNet, DenseNet, ConvNeXt)
│   │   ├── training/                      # train_model, train_lab2_models
│   │   ├── eval/                          # evaluate_model (CIFAR-10 metrics)
│   │   ├── experiments/                   # exp_01 to exp_08, plotters, benchmark_sota
│   │   ├── eda/                           # UMAP feature extraction and visualization
│   │   └── utils/                         # run_logger, checkpoint_utils, optuna_db_report
│   └── lab3/                              # Hugging Face NLP Package
│       ├── __init__.py                    # Lazy exports
│       ├── data/                          # prepare_imdb, eda_imdb, cleanlab_denoiser
│       ├── models/                        # model_builder, predictor
│       ├── training/                      # imdb_sentiment_train, trainer
│       ├── eval/                          # evaluate_model, evaluator, plotter, error_auditor
│       ├── experiments/                   # baseline_imdb_sentiment
│       └── utils/                         # run_logger, checkpoint_utils, resource_monitor, truncation_viz
│
├── notebooks/                             # INTERACTIVE ANALYSIS NOTEBOOKS
│   ├── lab1/                              # Fashion-MNIST notebooks
│   │   ├── practice_1.ipynb
│   │   └── error_analysis/
│   ├── lab2/                              # CIFAR-10 Transfer Learning notebooks
│   │   ├── practice_2.ipynb
│   │   ├── practice_2_calibration_verification.ipynb
│   │   ├── practice_2_error_analysis.ipynb
│   │   ├── practice_2_logit_bias_sweep.ipynb
│   │   ├── practice_2_moe.ipynb
│   │   ├── practice_2_stacking_mlp.ipynb
│   │   └── practice_2_tta.ipynb
│   └── lab3/                              # Hugging Face IMDB Sentiment notebooks
│       ├── 01_ex1_sentiment_baseline.ipynb
│       └── 02_ex2_finetune.ipynb
│
├── experiments/                           # PERSISTED RUN ARTIFACTS BY STAGE
│   ├── lab1/                              # Fashion-MNIST outputs (from outputs/)
│   │   ├── practice_1/                    # Metrics, plots, model weights
│   │   └── error_analysis/
│   ├── lab2/                              # CIFAR-10 outputs
│   │   ├── plots/                         # ~60 visualization PNGs
│   │   ├── results/                       # benchmark_metrics.json, UMAP embeddings
│   │   └── runs/                          # Gitignored run states (.gitkeep)
│   └── lab3/                              # Hugging Face outputs
│       ├── plots/                         # ~15 visualization PNGs
│       ├── results/                       # benchmark_metrics.json, Cleanlab issues JSON
│       └── runs/                          # Gitignored run states (.gitkeep)
│
├── data/                                  # DATASETS & ARTIFACT CACHES
│   ├── raw/                               # Read-only external datasets (.gitkeep)
│   ├── lab1/
│   │   └── splits/                        # JSON split indices
│   ├── lab2/
│   │   └── processed/                     # cifar10_split_seed42.json
│   └── lab3/
│       └── processed/                     # Cached denoised & tokenized arrow splits
│
├── configs/                               # EXPERIMENT CONFIGURATIONS
│   ├── lab2_data.yaml                     # CIFAR-10 data configuration
│   └── config_imdb_sentiment*.yaml        # 9 IMDB experiment configurations
│
├── tests/                                 # UNIT & INTEGRATION TESTS
│   ├── lab1/                              # Fashion-MNIST smoke tests
│   ├── lab2/                              # CIFAR-10 test suite (14 files + conftest.py)
│   └── lab3/                              # Hugging Face test suite (6 files + conftest.py)
│
├── requirements/                          # MULTI-TIER ENVIRONMENT DEPENDENCIES
│   ├── base.txt                           # Shared scientific core (numpy, pandas, torch, optuna)
│   ├── lab1.txt                           # Base + tabulate, jupyter
│   ├── lab2.txt                           # Base + tensorboard, pillow
│   ├── lab3.txt                           # Base + transformers, datasets, cleanlab, peft
│   └── dev.txt                            # Full dev environment + pytest, ruff, mypy
│
├── pyproject.toml                         # Unified build system, ruff, mypy, pytest config
├── pytest.ini                             # Root pytest configuration
├── requirements.txt                       # Convenience proxy file
└── README.MD                              # Unified project presentation
```

## Rules

- No random files in project_root (keep root lean: README, configs, tests, source dirs).
- Never write code directly in `data/raw/` or edit raw datasets.
- Always use `src/` modules for logic and call high-level class abstractions in Jupyter notebooks.
- **Constitutional Separation**: `/agents` contains only rules and templates that govern AI behavior and never change between labs. Evolving project knowledge, phase documentation, and experiment reports live in `/docs`.
- **Zero Premature Abstraction**: Keep lab source implementations isolated under `src/labN` until a module meets the strict 3-point test for shared code (used by ≥2 labs, identical semantics, zero change to research behavior).
