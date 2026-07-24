# AI-Assisted Development Rule Base
## Software Development & Documentation Standards

---

> **Scope:** This rule base governs the use of AI tools throughout the software development lifecycle. All contributors are expected to follow these guidelines to ensure quality, accountability, and intellectual integrity.

---

## Table of Contents

1. [Role Definition](#1-role-definition)
2. [Transparency & AI Usage Logging](#2-transparency--ai-usage-logging)
3. [Development Workflow](#3-development-workflow)
4. [Software Architecture Rules](#4-software-architecture-rules)
5. [Coding Rules](#5-coding-rules)
6. [AI-Generated Code Rules](#6-ai-generated-code-rules)
7. [Documentation Standards](#7-documentation-standards)
8. [Report Writing Rules](#8-report-writing-rules)
9. [Testing Rules](#9-testing-rules)
10. [Version Control Rules](#10-version-control-rules)
11. [Technical Decision Records](#11-technical-decision-records)
12. [Security Rules](#12-security-rules)
13. [Debugging Rules](#13-debugging-rules)
14. [Learning & Mentor Mode](#14-learning--mentor-mode)
15. [Project Completion Checklist](#15-project-completion-checklist)

---

## 1. Role Definition

The AI assistant serves as a **development support tool**, not a replacement for developer judgment or responsibility.

### AI Responsibilities
- Assist with analysis, design, implementation, testing, debugging, and documentation.
- Provide explanations, suggestions, and structured output upon request.
- Flag potential risks, ambiguities, or missing information proactively.

### Developer Responsibilities

The developer retains full ownership and accountability for:

| Area | Responsibility |
|---|---|
| Source Code | Final review and approval of all generated code |
| Architecture | System design decisions |
| Security | Identifying and mitigating vulnerabilities |
| Correctness | Verifying functional accuracy |
| Deployment | Ensuring the system is production-ready |
| Intellectual Ownership | All submitted work |

> **Principle:** AI-generated content must always be reviewed, understood, and approved by the developer before use.

---

## 2. Transparency & AI Usage Logging

Every AI interaction that meaningfully contributes to the project **must be documented** to maintain transparency and traceability.

### 2.1 AI Development Log

Create an entry in `docs/AI_Development_Log.md` for each significant AI session.

```markdown
## AI Development Log Entry

**Date:** YYYY-MM-DD
**AI Tool:** (e.g., Antigravity, GitHub Copilot)
**Model:** (e.g., Gemini 2.5 Pro)

### Context
- **Problem:** Description of the problem being solved
- **Module Affected:** Name of the project module

### Prompt
> Original prompt or request submitted to the AI

### AI Output Summary
Brief description of the idea, code, or document generated.

### Decision
- [ ] Accepted
- [ ] Modified
- [ ] Rejected

### Modifications Made
Description of changes applied after the AI suggestion.

### Rationale
Justification for the decision made.

### Related Commit
`git commit hash`
```

### 2.2 Prompt Record

Maintain a structured log of all significant prompts in `docs/prompts/`.

```markdown
## Prompt Record

**ID:** PR-YYYY-MM-DD-###
**Date:** YYYY-MM-DD

### Objective
What problem needs to be solved?

### Context
Project background and constraints provided to the AI.

### Prompt Input
Full prompt text submitted.

### Expected Output
Description of the desired result.

### Actual Result
Summary of the AI's response.

### Evaluation
Assessment of accuracy and usefulness (1–5 scale or descriptive).

### Status
- [ ] Accepted
- [ ] Modified
- [ ] Rejected
```

---

## 3. Development Workflow

All development activities — whether AI-assisted or manual — must follow this sequential workflow:

```
Requirement Analysis
        │
        ▼
  System Design
        │
        ▼
  Implementation
        │
        ▼
     Testing
        │
        ▼
  Documentation
        │
        ▼
     Review
        │
        ▼
      Commit
```

Do not proceed to the next phase without completing and reviewing the current one.

---

## 4. Software Architecture Rules

Before implementing any large feature, the following must be defined:

- **Requirement:** What the feature must accomplish.
- **Component Responsibility:** What each module handles.
- **Data Flow:** How data moves through the system.
- **Interface Design:** API contracts and boundaries.
- **Dependencies:** External libraries, services, or modules.

### Recommended Project Structure

```
Project/
├── src/
│   ├── controllers/       # Request handling and routing logic
│   ├── services/          # Business logic layer
│   ├── models/            # Data models and schemas
│   ├── utils/             # Shared utility functions
│   └── config/            # Configuration and environment settings
│
├── tests/                 # Unit, integration, and end-to-end tests
│
├── docs/                  # Project documentation
│
├── outputs/               # Generated artifacts and reports
│
└── README.md
```

### Layer Architecture Pattern

```
Feature Request
      │
      ▼
  Controller       ← Handles input/output
      │
      ▼
   Service         ← Implements business logic
      │
      ▼
  Data Layer       ← Manages data access and persistence
      │
      ▼
 Documentation     ← Updated with every change
```

> Each module must have a **single, clearly defined responsibility**.

---

## 5. Coding Rules

### 5.1 Before Writing Code

Always explain the following **before** generating or writing any code:

- **Purpose:** What the component does and why it is needed.
- **Input / Output:** Expected data types and structures.
- **Dependencies:** Libraries, services, or modules required.
- **Design Decisions:** Chosen approach and rationale.
- **Potential Risks:** Edge cases, failure modes, or performance concerns.

### 5.2 Code Quality Requirements

All code must be:

- ✅ **Clean and readable** — self-documenting with clear intent
- ✅ **Modular** — each unit has a single responsibility
- ✅ **Maintainable** — easy to modify without breaking other components
- ✅ **Consistently named** — follows project naming conventions
- ✅ **Standard-compliant** — adheres to the project's coding standards

### 5.3 What to Avoid

- ❌ Unnecessary dependencies or libraries
- ❌ Over-engineered solutions for simple problems
- ❌ Large, uncontrolled code generation
- ❌ Replacing existing code without prior analysis
- ❌ Copy-paste solutions without explanation

---

## 6. AI-Generated Code Rules

### 6.1 New Code

When generating new code, always provide:

1. **Explanation** — What the code does and why this approach was chosen.
2. **Implementation** — The actual code, well-commented.
3. **Testing Method** — How to verify correctness.
4. **Potential Issues** — Known limitations or failure points.

### 6.2 Modifying Existing Code

Before modifying any existing code in an established project:

**Analyze first:**
- Current structure and design patterns in use
- Existing implementation and business logic
- Dependencies that may be affected
- Possible side effects of the change

> **Rule:** Do **not** rewrite large portions of the project unless explicitly requested and approved by the developer.

---

## 7. Documentation Standards

AI must actively support the creation and maintenance of project documentation.

### 7.1 Required Documentation

```
docs/
├── README.md                  # Project overview and setup guide
├── System_Design.md           # Architecture and technical design
├── Testing_Report.md          # Test cases and results
├── AI_Development_Log.md      # AI usage history
├── Decision_Record.md         # Architectural decision records
└── prompts/                   # Stored prompt records
```

### 7.2 README Structure

Every project README must contain the following sections:

```markdown
# Project Name

## Introduction
Brief project overview and purpose.

## Features
List of main capabilities.

## Technology Stack
- **Languages:**
- **Frameworks:**
- **Libraries:**
- **Database:**

## Installation
Step-by-step setup instructions.

## Usage
How to run and interact with the project.

## Project Structure
Explanation of the folder layout.

## Configuration
Required environment variables and settings.

## Future Development
Planned improvements and known limitations.
```

### 7.3 Technical Documentation Structure

Technical documentation must include the following:

| Section | Contents |
|---|---|
| **System Overview** | Purpose, scope, and main components |
| **Architecture** | Architecture pattern, data flow, component communication |
| **Database Design** | ERD, table definitions, relationships |
| **API Documentation** | Endpoints, request/response schemas |
| **Algorithm Explanation** | Purpose, input/output, complexity, limitations |
| **Workflow Diagrams** | Visual representations of key processes |

---

## 8. Report Writing Rules

All project reports must follow **academic writing standards**.

### 8.1 Required Report Structure

1. Introduction
2. Problem Definition
3. Requirement Analysis
4. System Design
5. Implementation
6. Testing
7. Results
8. Discussion
9. Limitations
10. Future Development

### 8.2 Report Integrity Rules

- ✅ Use only actual project data and verified information.
- ✅ Clearly distinguish between assumptions and confirmed facts.
- ❌ Do **not** fabricate results, statistics, or experiments.
- ❌ Do **not** invent test outcomes or performance metrics.

If information is unavailable, mark it explicitly as:
> `[Requires verification from developer]`

---

## 9. Testing Rules

Every implemented feature **must** have a corresponding test record.

### 9.1 Test Case Template

```markdown
## Test Case

**Feature:**
**Test ID:** TC-###
**Date:** YYYY-MM-DD

### Input
Description of test input.

### Expected Result
What the system should produce.

### Actual Result
What the system actually produced.

### Status
- [ ] PASS
- [ ] FAIL

### Notes
Additional observations or follow-up actions.
```

### 9.2 Required Test Coverage

Every feature must be tested against:

| Test Type | Description |
|---|---|
| **Normal Cases** | Standard, expected input scenarios |
| **Edge Cases** | Boundary values and extreme inputs |
| **Invalid Input** | Malformed, null, or out-of-range data |
| **Error Handling** | System behavior on failure or exception |

---

## 10. Version Control Rules

All significant changes must be tracked using Git with a consistent commit format.

### 10.1 Commit Message Format

```
<type>(<scope>): <short description>
```

### 10.2 Commit Types

| Type | Usage |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `test` | Adding or updating tests |
| `refactor` | Code restructuring without behavior change |
| `chore` | Build process, tooling, or dependency updates |
| `perf` | Performance improvements |

### 10.3 Examples

```bash
feat(auth): add JWT-based authentication module
fix(db): resolve connection timeout on high load
docs(api): update endpoint documentation for v2
test(model): add unit tests for prediction pipeline
```

### 10.4 What to Avoid

Vague, non-descriptive commit messages such as:
- ❌ `update`
- ❌ `fix`
- ❌ `change`
- ❌ `test`

---

## 11. Technical Decision Records

All significant architectural or technical decisions must be documented in `docs/Decision_Record.md`.

### Decision Record Template

```markdown
# Technical Decision Record

**ID:** ADR-###
**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Deprecated | Superseded

## Problem Statement
Clear description of the problem or question that requires a decision.

## Context
Background information, constraints, and relevant factors.

## Considered Options

### Option A: [Name]
Description, pros, and cons.

### Option B: [Name]
Description, pros, and cons.

## Decision
**Chosen Option:** [Option Name]

## Rationale
Explanation of why this option was selected over alternatives.

## Trade-offs

| Advantage | Disadvantage |
|---|---|
| ... | ... |

## Consequences
Impact of this decision on the system and future development.
```

---

## 12. Security Rules

Security must be considered at every stage of development, not as an afterthought.

### 12.1 Always Consider

- **Authentication** — Verify the identity of users and services.
- **Authorization** — Enforce access control and permissions.
- **Input Validation** — Sanitize and validate all external input.
- **Data Protection** — Encrypt sensitive data at rest and in transit.
- **Secure Configuration** — Use environment variables and secrets management.
- **Dependency Security** — Audit third-party libraries for known vulnerabilities.

### 12.2 Never Generate

- ❌ Hard-coded passwords or credentials
- ❌ Exposed API keys or tokens in source code
- ❌ Unparameterized database queries (SQL injection risk)
- ❌ Insecure default settings or configurations
- ❌ Disabled security headers or CORS wildcards without justification

---

## 13. Debugging Rules

When diagnosing and resolving issues, follow this structured process:

| Step | Action |
|---|---|
| **1. Read** | Read the full error message and stack trace carefully. |
| **2. Identify** | Identify the root cause, not just the symptom. |
| **3. Explain** | Explain the problem clearly before suggesting a fix. |
| **4. Fix** | Apply the **minimal** change necessary to resolve the issue. |
| **5. Verify** | Confirm the fix resolves the issue without introducing new ones. |

> **Rule:** Do **not** randomly rewrite or refactor unrelated code while debugging.

---

## 14. Learning & Mentor Mode

When the developer is in a learning context, the AI must prioritize **understanding over output**.

### Guiding Principles

- Prioritize **explanation** over code generation.
- Walk through **reasoning**, not just results.
- Present **alternative approaches** and their trade-offs.
- Discuss **core concepts** relevant to the task.
- Highlight **best practices** and common pitfalls.

> **Rule:** Do **not** provide only a final answer. Always explain **why** a solution works and what the developer should understand from it.

---

## 15. Project Completion Checklist

Before considering the project complete, verify all items below.

### 📁 Source Code

- [ ] Code runs successfully without errors
- [ ] Project structure is organized and logical
- [ ] All dependencies are documented (`requirements.txt`, `package.json`, etc.)
- [ ] Source code is committed and version-controlled

### 🤖 AI Usage

- [ ] All AI tools used are recorded
- [ ] Important prompts are saved to `docs/prompts/`
- [ ] AI Development Log is completed and up to date
- [ ] Developer understands and can explain all AI-generated code

### 📄 Documentation

- [ ] `README.md` is complete and accurate
- [ ] Technical documentation is complete
- [ ] Testing report is complete
- [ ] Final project report is complete
- [ ] Decision records are up to date

### 🧪 Testing

- [ ] All test cases are documented
- [ ] Test results are recorded and verified
- [ ] Edge cases and error handling are tested
- [ ] No critical bugs remain unresolved

### 🚀 Delivery

- [ ] Source code repository is accessible
- [ ] All documentation is included
- [ ] Demo video is recorded (if required)
- [ ] Deployment instructions are documented
- [ ] AI usage report is prepared

---

*This rule base is a living document. Update it as project standards evolve.*