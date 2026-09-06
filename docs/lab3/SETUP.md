# SETUP.md — How to Bootstrap a New Project from This Template

- **Motivation/Background**: Every new deep-learning project repeats the same
  scaffolding work: folder layout, agent knowledge base, lint/test config, CI.
  This template packages all of it so a new project can start with the agent
  workflow already wired.
- **Purpose**: One-time instructions for turning this template into a working
  new project — paste the template, fill in placeholders, and follow the agent
  setup guide.
- **Overview Pipeline**: copy template → rename/fill placeholders → create
  environment → git init → follow the agent setup guide → start phase 1.
- **Detailed Plan**: §1 prerequisites; §2 instantiate; §3 placeholders to fill;
  §4 environment; §5 git; §6 agent workflow; §7 verification.
- **References**: `agents/HOW_TO_SETUP_AI_AGENT.md`, `agents/rules/*`,
  `requirements.txt`, `.github/workflows/ci.yml`.

- **Created**: 2026-09-06T14:38:06+07:00
- **Last Updated**: 2026-09-06T14:38:06+07:00

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Instantiate a New Project](#2-instantiate-a-new-project)
3. [Fill In the Placeholders](#3-fill-in-the-placeholders)
4. [Create the Environment](#4-create-the-environment)
5. [Initialize Git](#5-initialize-git)
6. [Wire Up the AI Agent Workflow](#6-wire-up-the-ai-agent-workflow)
7. [Verification Checklist](#7-verification-checklist)

---

## 1. Prerequisites

- Python 3.11+ installed.
- PyTorch/torchvision installable (see
  [requirements.txt](requirements.txt) — CUDA index URL may be needed).
- A git remote (GitHub) if CI is desired.

## 2. Instantiate a New Project

1. Copy the whole `Deep_learning_template` folder to the new project location.
2. Rename the folder to the project name (e.g. `My_Project`).
3. Delete this file from the copy if you prefer, or keep it as a reference.

## 3. Fill In the Placeholders

| File                                                                                | What to replace                                                           |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [README.md](README.md)                                                               | `[PROJECT_NAME]`, project description, architecture details             |
| [agents/PURPOSE.md](agents/PURPOSE.md)                                               | The project brief / objective — Step 2 of the agent setup guide          |
| [agents/OVERVIEW.md](agents/OVERVIEW.md)                                             | Project plan, phases, datasets, success criteria                          |
| [configs/config.yaml.example](configs/config.yaml.example)                           | Copy to`configs/config.yaml` and set dataset/dataloader/training values |
| [pyproject.toml](pyproject.toml)                                                     | `name`, `description`, ruff excludes for your scripts                 |
| [.github/workflows/ci.yml](.github/workflows/ci.yml)                                 | Branch triggers if you use feature branches                               |
| [agents/rules/FOLDER_STRUCTURE.md](agents/rules/FOLDER_STRUCTURE.md)                 | Revisit after every phase — it drifts with implementation                |
| [agents/rules/RESULTS_REPORTING.md](agents/rules/RESULTS_REPORTING.md)               | The metric-description table (§3) — define your project's metrics       |
| [agents/rules/LOGGING_CHECKPOINT_RULES.md](agents/rules/LOGGING_CHECKPOINT_RULES.md) | Appendix A — your script entry points, run names, result dirs            |

## 4. Create the Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux)
pip install -r requirements.txt
# If the project needs a CUDA build of PyTorch, install it FIRST:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify:

```bash
pytest tests/ -v        # smoke tests pass out of the box
ruff check src tests    # lint passes out of the box
```

## 5. Initialize Git

```bash
git init
git add .
git commit -m "chore: bootstrap project from Deep_learning_template"
git branch -M main
git remote add origin <url>
```

Do not commit `data/`, `experiments/runs/`, `.venv/`, or large binaries —
the provided [.gitignore](.gitignore) already excludes them.

## 6. Wire Up the AI Agent Workflow

Follow [agents/HOW_TO_SETUP_AI_AGENT.md](agents/HOW_TO_SETUP_AI_AGENT.md) step
by step. The short version:

1. **rules/** are already in place (always-on conventions).
2. Write your objective into [agents/PURPOSE.md](agents/PURPOSE.md).
3. Have the agent run the clarifying interview, then finalize PURPOSE.md.
4. Generate the roadmap from
   [agents/templates/PROJECT_ROADMAP_TEMPLATE.md](agents/templates/PROJECT_ROADMAP_TEMPLATE.md).
5. Create phase docs from
   [agents/phases/PHASE_TEMPLATE.md](agents/phases/PHASE_TEMPLATE.md).
6. Implement phase by phase; track status in `agents/progress/`; audit before
   marking a phase done.

## 7. Verification Checklist

- [ ] `pytest` passes (smoke tests included)
- [ ] `ruff check src tests` passes
- [ ] `configs/config.yaml` exists with real values
- [ ] `agents/PURPOSE.md` and `agents/OVERVIEW.md` describe the real project
- [ ] README `[PROJECT_NAME]` placeholders replaced
- [ ] First phase doc exists in `agents/phases/`
- [ ] Agent setup guide (Step 6) read and followed

---

## Appendix — What Was Generalized From the Source Project

The template was extracted from the `Deeplearning_Course` repository (LAB2,
CIFAR-10 transfer learning). Project-specific content was removed or
templated; the reusable core was kept:

- **Kept verbatim**: agent philosophy/rules that are tool-agnostic
  (`AGENT_AI.md`, `CODEBASE_AUDIT.md`, `NAMING_CONVENTION.md`), all document
  templates, the ML pipeline reference, the git/release guide, the Optuna DB
  guide, `pytest.ini`, `requirements.txt`.
- **Generalized**: `FOLDER_STRUCTURE.md`, `RESULTS_REPORTING.md` (metric table
  made project-definable), `LOGGING_CHECKPOINT_RULES.md` (LAB2 appendix
  replaced with a blank example), `MD_CONVENTION.md` /
  `NOTEBOOK_HEADER_CONVENTION.md` (LAB2 example paths), `README.md`,
  `agents/README.md`, CI workflow (branch list), `.gitignore`, `pyproject.toml`.
- **Templated (fill per project)**: `agents/PURPOSE.md`, `agents/OVERVIEW.md`,
  `configs/config.yaml.example`, `experiments/results/README.md`,
  `agents/experiments/README.md`, `agents/bugs/README.md`.
