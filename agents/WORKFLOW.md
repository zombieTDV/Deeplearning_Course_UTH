# Vibe Coding Workflow with LLM

## 1. Building the LLM System

### 1.1. Context Management

The goal is to ensure the LLM always works within the correct project context.

Key factors to optimize:

* **Context Relevance**: Only load information relevant to the current task into the active LLM context window.
* **Faithfulness**: Responses must strictly follow existing documentation and codebase loaded into context.
* **Hallucination Control**: Prevent AI from guessing or fabricating information by requiring context lookup.
* **Knowledge Store**: Build a documentation store in `agents/` and `agents/knowledge_store/` so the LLM loads specs into the active LLM context window during code generation.

---

## 2. Behavioral Layer (Prompting)

All AI behaviors are controlled through Markdown (`.md`) files.

Each file describes a specific function or workflow.

Standard structure:

```text
Name
Description
Purpose
Input
Output
Workflow / How to do
Examples (if needed)
```

Example:

```text
Name:
Data Preparation

Description:
Standardize and preprocess data.

Purpose:
Prepare data before training.

Input:
Raw Dataset

Output:
Processed Dataset

How to do:
1. Load data
2. Clean data
3. Feature Engineering
4. Validation
```

---

## 3. Task Description

Each task must be fully described before AI starts writing code.

Must include:

* Goal
* Plan
* Pipeline
* Detailed execution steps
* Completion criteria

Flow:

```text
Plan
↓
Pipeline
↓
Detailed Execution Plan
```

---

## 4. Progress Tracking

Each task must have a clear status.

Kanban states:

* To Do
* In Progress
* Done
* On Hold
* Cancelled

This helps AI know which parts are completed and which are not yet implemented.

Track in: `agents/PROGRESS_TRACKING.md`

---

## 5. Pipeline Separation

Do not do everything in one file or one prompt.

Split into separate pipelines:

```text
Data Preparation
Training
Model
Evaluation
Deployment (if applicable)
```

Each pipeline has:

* Its own prompt spec
* Its own documentation
* Its own checklist

---

## 6. AI Reference

Build a reference standard so AI always follows project conventions.

### Naming Convention

Rules:

* **File names**: `snake_case` (e.g., `data_loader.py`, `train_pipeline.py`)
* **Class names**: `PascalCase` (e.g., `ResNetBackbone`, `Trainer`)
* **Function / Variable names**: `snake_case` (e.g., `compute_loss()`, `learning_rate`)
* **Constants**: `UPPER_SNAKE_CASE` (e.g., `BATCH_SIZE`, `NUM_CLASSES`)

### Folder Structure

```text
project/
├── agents/         # LLM workflow, prompts, rules, references
├── data/           # Raw & Processed datasets
├── models/         # Model architectures & saved checkpoints
├── labs/           # Lab exercise implementations
├── scripts/        # Entrypoint scripts & utilities
├── logs/           # Training & experiment logs
├── tests/          # Smoke, unit, integration tests
└── Cource_materials/
```

AI must always follow the agreed folder structure.

---

## 7. Codebase Audit

After AI generates code, it must be reviewed.

Process:

```text
Generated Code
↓
Codebase Audit
↓
Check if it matches the architecture
↓
Check if it matches the Detailed Plan
↓
Check coding convention compliance
```

Not only check if the code runs, but also check if it matches the original design.

---

## 8. Unit Test

After completing a module:

* Write Unit Test
* Run Unit Test
* Compare results against the Detailed Plan

If tests do not reflect the requirements in the plan, fix them.

---

## 9. Two Testing Modes

Each time AI generates code, run two levels of verification.

### Smoke Test

Quick check:

* Does it run?
* Any syntax errors?
* Are imports correct?
* Are dependencies available?

### Normal Test

Full check:

* Unit Test
* Integration Test
* Evaluation
* Result verification
* Performance check

Do not run full tests from the start — it wastes time.

---

## 10. AI-First Workflow

Recommended implementation order:

```text
Problem
↓
AI Solution Design
↓
Data
↓
Model
↓
Training
↓
Evaluation
↓
Full Stack Integration
```

The idea is to let AI assist from the problem analysis phase first, then sequentially implement data, model, and integrate the entire system.

---

# Overall Workflow

```text
Requirement
        │
        ▼
LLM Context Management
(Context / Faithfulness / Hallucination / Knowledge Store)
        │
        ▼
Behavior Layer (Prompting)
        │
        ▼
Task Description
(Plan → Pipeline → Detailed Plan)
        │
        ▼
Progress Tracking
(To Do / In Progress / Done / On Hold / Cancelled)
        │
        ▼
Pipeline Separation
(Data Prep → Training → Model → Evaluation)
        │
        ▼
AI Reference
(Naming Convention / Folder Structure)
        │
        ▼
AI Generate Code
        │
        ▼
Codebase Audit
        │
        ▼
Unit Test
        │
        ▼
Smoke Test
        │
        ▼
Normal Test
        │
        ▼
Complete
```
