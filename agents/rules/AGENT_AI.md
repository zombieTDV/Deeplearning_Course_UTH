# AGENT_AI.md — AI Agent Behavior, Engineering Philosophy & Workflow

- **Motivation/Background**: AI agent interactions often suffer from ungrounded assumptions, silent hallucinated path changes, drift between documentation and code, and lack of structured verification before code execution.
- **Purpose**: Define the binding engineering philosophy, communication rules, six-stage workflow lifecycle, and operational constraints for AI agents working in this repository.
- **Overview Pipeline**: Formulated from the multi-lab consolidation and codified as an immutable constitutional governance rule.
- **Detailed Plan**: §1 Core Agent Philosophy; §2 Six-Stage Engineering Workflow (AUDIT → PLAN → IMPLEMENT → VERIFY → COMMIT → MERGE); §3 Inter-Agent Handoff Protocol; §4 Multi-Track Colocation Principle; §5 Hard Operational Constraints.
- **References**: `agents/rules/FOLDER_STRUCTURE.md`, `agents/rules/CODEBASE_AUDIT.md`, `agents/rules/MD_CONVENTION.md`.
- **Created**: 2026-09-06T13:05:18+07:00
- **Last Updated**: 2026-09-06T21:25:00+07:00

---

## Table of Contents

- [1. Core Agent Philosophy](#1-core-agent-philosophy)
- [2. The Six-Stage Workflow Lifecycle](#2-the-six-stage-workflow-lifecycle)
- [3. Inter-Agent Handoff Protocol](#3-inter-agent-handoff-protocol)
- [4. Multi-Track Colocation Principle](#4-multi-track-colocation-principle)
- [5. Hard Operational Constraints](#5-hard-operational-constraints)

---

## 1. Core Agent Philosophy

1. **Second Brain, Not Second-Guesser:** The agent maintains project memory through structured documentation so human engineers and subsequent agents do not suffer context burnout.
2. **Read First, Act Second:** Before executing code changes, the agent must read the constitutional rules under `agents/rules/` and current project state in `docs/`.
3. **No Silent Drift:** Any divergence between documentation and code is a blocker to be reported, not something to silently "fix" without human alignment.
4. **Verified Evidence Over Claims:** Never state that tests passed or a model works without citing exact commands, execution logs, and 5W1H empirical context.

---

## 2. The Six-Stage Workflow Lifecycle

All non-trivial engineering tasks MUST progress through these six distinct stages:

```text
AUDIT ──► PLAN ──► IMPLEMENT ──► VERIFY ──► COMMIT ──► MERGE
```

### Stage 1: AUDIT (Strictly Read-Only)
- Inspect the current working tree, Git status, and active dependencies.
- Catch drift between documentation claims and real source code per [agents/rules/CODEBASE_AUDIT.md](CODEBASE_AUDIT.md).
- Identify risks, constraints, and dependencies before modifying any file.

### Stage 2: PLAN (Proposal & Design)
- Formulate a minimal, concrete implementation plan.
- Map out files to be created, modified, or moved.
- State verification steps in advance.
- Present plan to human engineer for approval when design choices exist.

### Stage 3: IMPLEMENT (Minimal Targeted Execution)
- Modify only authorized files.
- Adhere strictly to [agents/rules/FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) and [agents/rules/NAMING_CONVENTION.md](NAMING_CONVENTION.md).
- In multi-track projects, colocate new code, tests, and experiment specs within that lab's folder rather than scattering them across unrelated global folders.
- Keep implementation modular, preserving existing working interfaces.
- Update timestamps on all touched `.md` files per [agents/rules/MD_CONVENTION.md](MD_CONVENTION.md).

### Stage 4: VERIFY (Local Non-Destructive Testing)
- Run smoke tests on modified scripts before full execution.
- Run the test suite: `pytest tests/<target_lab>/ -q`.
- Check dependency integrity: `python -m pip check`.
- Verify package imports cleanly from root: `python -c "import src.<target_lab>"`.

### Stage 5: COMMIT (Human Gate & Clean Working Tree)
- Verify `git status` and `git diff`.
- Present clear summary and commit message to human engineer.
- **Obtain explicit human approval before committing.**
- Commit with conventional commit format (`feat:`, `fix:`, `refactor:`, `docs:`).

### Stage 6: MERGE (Isolated Feature Integration)
- Merge only fully verified, clean branches into `main`.
- Resolve merge conflicts with extreme care; never overwrite working code.

---

## 3. Inter-Agent Handoff Protocol

When transferring context to a subsequent agent or turn:
1. State the **Current Objective** and completed deliverables.
2. Record **Files Modified** and git working-tree state.
3. List **Active Blockers** or open questions.
4. Detail **Concrete Next Steps** with exact files and commands.
5. Reference the handoff template at [`docs/shared/HANDOFF_TEMPLATE.md`](../../docs/shared/HANDOFF_TEMPLATE.md).

---

## 4. Multi-Track Colocation Principle

In this repository, work is structured around domain-segregated tracks (`lab1`, `lab2`, `lab3`).
- Code belongs in `src/<lab>/`.
- Unit and integration tests belong in `tests/<lab>/`.
- Phase specs, experiment writeups, and bug reports belong in `docs/<lab>/`.
- Configs belong in `configs/`.
- Do not create global grab-bag folders that mix artifacts from different labs.

---

## 5. Hard Operational Constraints

1. **No Training in Notebooks:** Notebooks are for exploration, verification, demo, and visualization only. All training must run via CLI Python scripts under `src/<lab>/training/` or `src/<lab>/experiments/`.
2. **Human Approval Gate for Commits and Pushes:** An AI agent must NEVER commit or push code without explicit human review and approval.
3. **Mandatory Bug Documentation Rule:** Every time a bug, regression, or environment failure is encountered and fixed, the agent MUST document it under the active lab directory (`docs/<lab>/bugs/`) using [agents/templates/BUG_TEMPLATE.md](../templates/BUG_TEMPLATE.md) and register it in that lab's bug index.
4. **Data Immutability:** Never modify raw data files in `data/raw/` or edit raw datasets.
5. **EDA Baseline:** Baseline exploratory data analysis must at minimum examine `df.info()`, `df.describe()`, and `df.shape`.
6. **LLM Evaluation Dimensions:** When evaluating NLP/LLM outputs, assess across three pillars: Hallucination, Context Relevance, and Faithfulness.
