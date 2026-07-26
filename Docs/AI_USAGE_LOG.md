# AI Usage Log

> **Project:** Deep Learning Course — UTH  
> **Team:** 6 Members  
> **Repository:** [GitHub Link]  
> **Last Updated:** 2026-07-26  

---

## AI Tools Registry

| Tool | Version/Model | Purpose | Risk Level | Data Policy |
|---|---|---|---|---|
| ChatGPT | GPT-4o / GPT-5 | Analysis, debugging, documentation | Medium | No sensitive data |
| GitHub Copilot | Latest | Code completion, boilerplate | Low | Opt-out training |
| Google Gemini | 2.5 Pro / 3.6 Flash | Research, code review | Medium | No sensitive data |
| Claude | Sonnet 4 / Opus 4.6 | Refactoring, architecture | Medium | No sensitive data |
| Antigravity AI | Latest | Architectural design, SOP, setup | Medium | Workspace scoped |

---

## Usage Records

### Record: AIL-2026-07-26-001

| Field | Details |
|---|---|
| **Date** | 2026-07-26 |
| **Member** | Project Lead / Team |
| **AI Tool** | Antigravity AI — Claude Opus 4.6 / Pro Tier |
| **SDLC Phase** | Requirements & System Design |
| **Objective** | Create a master SOP and workflow diagram/analysis suite for AI-Assisted Development (Vibe Coding) adhering to UTH lecturer requirements and ISO/OWASP standards. |
| **Context** | Needed an industry-grade SOP covering 7 SDLC phases, governance charter, RACI matrix, compliance evaluation (ISO 12207, ISO 25010, OWASP, GitHub Flow, Conventional Commits), and 12 Mermaid diagrams. |
| **Related Files** | `Docs/SOP_AI_Assisted_Development.md`, `Docs/SOP_Workflow_Diagrams_and_Analysis.md` |

**Original Prompt:**
> You are a Senior Software Engineering Architect... I am building a standardized AI-Assisted Development Workflow (AI + Vibe Coding Workflow) for my university software engineering team. The workflow must satisfy my lecturer's 9 requirements while following software engineering best practices...

**Refined Prompt:**
> Don't simplify anything. Instead make it more professional, more complete, more maintainable, and more scalable while keeping it practical for university students. @[Docs/AI_usage_process.md]

**AI Response Summary:**
> Generated 2 comprehensive core SOP documents containing 7 SDLC phases, RACI matrix for 6-10 members, governance principles, 12 Mermaid diagrams, 3 sequence diagrams, BPMN workflow, ISO 12207 / ISO 25010 / OWASP compliance mapping, and gap analysis against original draft.

**Decision:** Modified

| What was... | Details |
|---|---|
| ✅ Accepted | 7 SDLC phases, ISO/OWASP compliance evaluations, Mermaid diagram architecture, 11-step vibe coding sub-workflow |
| ✏️ Modified | Tailored team RACI matrix specifically for 6 members, integrated Vietnamese technical terms in parentheses |
| ❌ Rejected | Monolithic single-file structure (split into Master SOP + Diagrams/Analysis for readability) |

**Modification Details:**
> Split output into two dedicated files (`SOP_AI_Assisted_Development.md` and `SOP_Workflow_Diagrams_and_Analysis.md`) to maintain document modularity and prevent single-file clutter.

**Reason for Decision:**
> Enhances document maintainability and allows team members to reference diagrams independently during project defense.

**Verification:**
- Method: Manual verification & Markdown diagram rendering check
- Result: PASS ✅
- Security Review: Verified inclusion of OWASP Top 10 security review checklist in Phase 3 & Phase 4

**Related Git Commit(s):**
- `docs(governance): establish AI-assisted workflow SOP and project structure`

**Traceability:**
- Requirement: REQ-GOV-01 (Lecturer Requirement 1-9)
- Prompt File: `Docs/prompts/requirements/bush-ai-assisted-workflow-sop.md`

---

### Record: AIL-2026-07-26-002

| Field | Details |
|---|---|
| **Date** | 2026-07-26 |
| **Member** | Project Lead / Tech Lead |
| **AI Tool** | Antigravity AI — Pro Tier |
| **SDLC Phase** | Design & Governance |
| **Objective** | Generate 10 standardized markdown templates for AI governance, prompt library, ADRs, code review, testing, PRs, commit conventions, and branching strategy. |
| **Context** | Team needed ready-to-use template files to enforce standard operating procedures across all 6 team members. |
| **Related Files** | `Docs/templates/AI_USAGE_LOG.md`, `Docs/templates/PROMPT_LIBRARY.md`, `Docs/templates/AI_DECISION_RECORD.md`, `Docs/templates/CODE_REVIEW_CHECKLIST.md`, `Docs/templates/TESTING_TEMPLATE.md`, `Docs/templates/PULL_REQUEST_TEMPLATE.md`, `Docs/templates/COMMIT_CONVENTION.md`, `Docs/templates/BRANCHING_STRATEGY.md`, `Docs/templates/REPOSITORY_STRUCTURE.md`, `Docs/templates/AI_GOVERNANCE_CHARTER.md` |

**Original Prompt:**
> Create the template files for an AI-Assisted Development Workflow SOP covering AI Usage Log, Prompt Library, ADR, Code Review Checklist, Testing Template, PR Template, Commit Convention, Branching Strategy, Repository Structure, and AI Governance Charter.

**Refined Prompt:**
> Ensure all templates contain complete filled-in examples, 4-level AI usage classifications, OWASP checklists, 15 commit convention examples, and `experiment/*` branch rules.

**AI Response Summary:**
> Created 10 industry-grade templates with real-world examples, Conventional Commits metadata fields, GitHub Flow extension with AI experiment branches, and 4-level AI risk classification.

**Decision:** Accepted

| What was... | Details |
|---|---|
| ✅ Accepted | All 10 template files with filled-in examples and standard metadata fields |
| ✏️ Modified | Updated PR template to copy directly to `.github/PULL_REQUEST_TEMPLATE.md` |
| ❌ Rejected | None |

**Modification Details:**
> Placed primary template copies in `Docs/templates/` and copied the active GitHub PR template to `.github/`.

**Reason for Decision:**
> Allows GitHub to automatically populate the PR template when team members open Pull Requests.

**Verification:**
- Method: File integrity & format validation
- Result: PASS ✅
- Security Review: Verified presence of OWASP checklist items in Code Review and Governance Charter

**Related Git Commit(s):**
- `docs(governance): establish AI-assisted workflow SOP and project structure`

**Traceability:**
- Requirement: REQ-GOV-02 (SOP Templates)
- Test Case: TC-GOV-01

---

### Record: AIL-2026-07-26-003

| Field | Details |
|---|---|
| **Date** | 2026-07-26 |
| **Member** | Tech Lead / Developers |
| **AI Tool** | Antigravity AI — Flash Tier |
| **SDLC Phase** | Implementation & Repository Setup |
| **Objective** | Set up complete repository folder structure, GitHub issue/PR templates, contributing guidelines, editor config, and `.gitkeep` placeholders for empty folders. |
| **Context** | Needed a clean, production-ready directory structure for GitHub collaboration without modifying existing `src/` codebase files. |
| **Related Files** | `.github/`, `src/`, `tests/`, `scripts/`, `Docs/`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.editorconfig` |

**Original Prompt:**
> Create folder structure with simple details what it folder use for and my team have 6 member. Add .gitkeep to all empty folders except src/.

**Refined Prompt:**
> Keep src/ untouched except original `__init__.py`. Add 18 `.gitkeep` files to all other empty subdirectories across Docs/, tests/, scripts/, and .github/.

**AI Response Summary:**
> Created directory layout, 14 folder READMEs, `.editorconfig`, issue templates (`bug_report.md`, `feature_request.md`, `ai_experiment.md`), `CONTRIBUTING.md`, `CHANGELOG.md`, and placed 18 `.gitkeep` files while leaving `src/` untouched.

**Decision:** Modified

| What was... | Details |
|---|---|
| ✅ Accepted | Directory layout, issue templates, `.gitkeep` placement in 18 empty folders |
| ✏️ Modified | Cleaned up READMEs from `src/` to respect "src do not touch" requirement |
| ❌ Rejected | Direct README placement inside `src/backend/`, `src/frontend/`, `src/shared/` |

**Modification Details:**
> Removed generated README files inside `src/` subfolders per developer directive while retaining folder structure and `src/__init__.py`.

**Reason for Decision:**
> Respect explicit developer constraint regarding `src/` codebase isolation.

**Verification:**
- Method: `find` shell verification & git status check
- Result: PASS ✅
- Security Review: Verified `.gitignore` covers `.env` and virtual environments

**Related Git Commit(s):**
- `docs(governance): establish AI-assisted workflow SOP and project structure`

**Traceability:**
- Requirement: REQ-DEV-01 (Repository Structure)

---

### Record: AIL-2026-07-26-004

| Field | Details |
|---|---|
| **Date** | 2026-07-26 |
| **Member** | Documentation Manager / All Members |
| **AI Tool** | Antigravity AI — Gemini 3.6 Flash |
| **SDLC Phase** | Workflow & Documentation |
| **Objective** | Create a step-by-step 10-stage Vibe Coding Pipeline guide (`Docs/VIBE_CODING_PIPELINE.md`) and establish topic + author prompt file naming rules for the 6-member team. |
| **Context** | Team needed a practical reference card and workflow diagram to follow during daily coding sessions to guarantee lecturer traceability. |
| **Related Files** | `Docs/VIBE_CODING_PIPELINE.md`, `Docs/prompts/README.md` |

**Original Prompt:**
> Create a full pipeline for use AI assistance vibecoding to support teammate use. Organize prompts by topic + member name in filename.

**Refined Prompt:**
> Include 10 steps, security red flags, python test examples, quick log template, commit message examples, printable checklist card, and Mermaid flowchart. Set prompt format to `<member-name>-<description>.md`.

**AI Response Summary:**
> Generated `Docs/VIBE_CODING_PIPELINE.md` with 10 steps, security red flags, code modification examples, python testing snippets, printable 11-item checklist, Mermaid flowchart, and updated `Docs/prompts/README.md` with member-name naming rules.

**Decision:** Accepted

| What was... | Details |
|---|---|
| ✅ Accepted | 10-step pipeline, security red flags table, code modification examples, prompt file naming convention (`<member-name>-<description>.md`) |
| ✏️ Modified | None |
| ❌ Rejected | None |

**Modification Details:**
> Applied directly as created.

**Reason for Decision:**
> Perfectly balances academic traceability (who used which prompt) with clear topic organization (`requirements/`, `design/`, `coding/`, `testing/`, `documentation/`).

**Verification:**
- Method: Manual verification
- Result: PASS ✅
- Security Review: Includes 8 security red flags (SQL injection, hardcoded secrets, exposed stack traces, input validation)

**Related Git Commit(s):**
- `docs(governance): establish AI-assisted workflow SOP and project structure`

**Traceability:**
- Requirement: REQ-GOV-03 (Vibe Coding Pipeline)
- Prompt File: `Docs/prompts/coding/bush-vibe-coding-pipeline.md`

---

<!-- ADD NEW RECORDS ABOVE THIS LINE -->

## Monthly Summary

| Month | Total Uses | Accepted | Modified | Rejected | Top Tool |
|---|---|---|---|---|---|
| July 2026 | 4 | 2 | 2 | 0 | Antigravity AI |
