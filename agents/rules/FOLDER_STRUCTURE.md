# FOLDER_STRUCTURE.md

Source of truth for where things live. The agent should check this before creating any new file, and update it (with human approval) if the structure changes.

```
project_root/
├── agents/                # Agent AI Knowledge Base (behavior layer, rules, plans, progress)
│   ├── README.md          # Index & guide for Agent AI
│   ├── OVERVIEW.md        # Core project overview & roadmap
│   ├── PURPOSE.md         # Original brief & requirements
│   ├── PROJECT_ROADMAP.md # Execution plan: milestones, phases, tasks, timeline
│   ├── CODEBASE_AUDIT_REPORT.md # Running audit log (findings + phase completion gates)
│   ├── HOW_TO_SETUP_AI_AGENT.md # Agent workflow setup guide
│   ├── rules/             # Guidelines & standards for Agent AI
│   │   ├── AGENT_AI.md    # Agent AI philosophy & behavior rules
│   │   ├── CODEBASE_AUDIT.md # Codebase audit procedure
│   │   ├── FOLDER_STRUCTURE.md
│   │   ├── MD_CONVENTION.md
│   │   ├── NAMING_CONVENTION.md
│   │   ├── NOTEBOOK_HEADER_CONVENTION.md
│   │   └── PYTORCH_FRAMEWORK_RULES.md
│   ├── phases/            # Phase & pipeline documentation
│   │   ├── PHASE_TEMPLATE.md
│   │   └── <PHASE>.md     # SETUP, DATA_PREP, FEATURE_SPLIT, BASELINE, MODEL, TRAINING_INFO, EVAL, REPORT
│   ├── plans/             # Experiment/implementation plans
│   │   ├── PLAN_EX2_*.md            # active plans
│   │   └── completed/               # archived/resolved plans (BUGFIX_*, PLAN_EX2_*)
│   ├── templates/         # Document & checklist templates
│   │   ├── PROJECT_ROADMAP_TEMPLATE.md
│   │   ├── PROGRESS_STATUS_TEMPLATE.md
│   │   ├── CODEBASE_AUDIT_TEMPLATE.md
│   │   └── SMOKE_TEST_CHECKLIST.md
│   ├── references/        # External guides & reference docs
│   │   ├── REFERENCE_TEMPLATE.md
│   │   └── <GUIDE>.md     # e.g. OPTUNA_DB_GUIDE.md, GIT_AND_RELEASE_BEST_PRACTICES.md
│   ├── experiments/       # Experiment reports, plans & technical comparisons
│   │   ├── README.md
│   │   └── <EXP>.md       # EX1…EX14 reports (indexed in README.md)
│   ├── progress/          # One status file per task/phase
│   │   └── <PHASE>_STATUS.md  # from PROGRESS_TEMPLATE.md
│   └── bugs/              # Documented bug reports
│       ├── README.md
│       └── BUG_<NN>_<SHORT>.md
├── data/
│   ├── raw/               # Never edited by the agent
│   ├── processed/         # Cached tokenized IMDB datasets (imdb_tokenized_512, imdb_denoised_512)
│   └── external/
├── src/                   # Core modular package (top-level exports in src/__init__.py)
│   ├── data/              # prepare_imdb.py (clean_text, head_tail_tokenize), eda_imdb.py (IMDBDatasetEDA), cleanlab_denoiser.py (IMDBCleanlabAuditor)
│   ├── models/            # model_builder.py (build_model, LLRD), predictor.py (SentimentPredictor)
│   ├── training/          # imdb_sentiment_train.py (CLI), trainer.py (IMDBTrainer)
│   ├── eval/              # evaluate_model.py, evaluator.py (IMDBEvaluator), plotter.py (IMDBPlotter), error_auditor.py (ErrorAuditor)
│   └── utils/             # run_logger.py, checkpoint_utils.py, resource_monitor.py, truncation_viz.py (visualize_head_tail_truncation)
├── notebooks/             # Deliverable interactive notebooks (analysis only — no training loops)
│   ├── 01_ex1_sentiment_baseline.ipynb  # Zero-Shot Baseline Exercise 1 (89.07% Acc)
│   └── 02_ex2_finetune.ipynb            # Exercise 2 end-to-end: EDA, HTML cleaning, Head+Tail truncation, 5-Fold OOF Cleanlab audit, LoRA training & eval (peak 93.14% Acc)
├── scratch/               # Generated utility & audit scripts (build_notebook.py)
├── configs/               # YAML experiment configuration files (config_imdb_sentiment_*.yaml incl. denoised + fullft presets)
├── experiments/           # Run outputs, checkpoints, plots, evaluation results
└── tests/                 # Unit tests & smoke test suite
```

## Rules

- No random files in project_root (keep root lean: README, configs, tests, source dirs).
- Never write code directly in `data/raw/` or edit raw datasets.
- Always use `src/` modules for logic and call high-level class abstractions in Jupyter notebooks.
