# Repository Structure

This document outlines the standard folder structure for our capstone project repository. Maintaining a clean and predictable structure is crucial for team collaboration and code maintainability.

---

## Directory Tree

```text
Project/
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── ai_experiment.md
│   └── workflows/
│       ├── ci.yml
│       └── lint.yml
│
├── src/
│   ├── backend/
│   │   ├── controllers/       # Request handling and routing logic
│   │   ├── services/          # Business logic layer
│   │   ├── models/            # Data models and schemas
│   │   ├── middleware/        # Authentication, logging, error handling
│   │   ├── utils/             # Shared utility functions
│   │   └── config/            # Configuration and environment settings
│   ├── frontend/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Page-level components / routes
│   │   ├── assets/            # Images, fonts, static files
│   │   ├── styles/            # Global CSS / SCSS files
│   │   └── utils/             # Client-side utility functions
│   └── shared/
│       ├── types/             # Shared type definitions / interfaces
│       └── constants/         # Shared constants and enums
│
├── tests/
│   ├── unit/                  # Unit tests (isolated component tests)
│   ├── integration/           # Integration tests (cross-module tests)
│   ├── e2e/                   # End-to-end tests (full workflow tests)
│   └── fixtures/              # Test data, mocks, and stubs
│
├── docs/
│   ├── SRS.md                              # Software Requirements Specification
│   ├── System_Design.md                    # System design and architecture
│   ├── Architecture.md                     # Architecture diagrams and decisions
│   ├── Database_Design.md                  # ERD, table definitions, relationships
│   ├── API_Documentation.md                # API endpoints, request/response schemas
│   ├── AI_GOVERNANCE_CHARTER.md            # AI governance policies and principles
│   ├── AI_USAGE_LOG.md                     # AI usage records (lecturer requirement)
│   ├── AI_TOOLS_REGISTRY.md                # Approved AI tools with versions
│   │
│   ├── DECISION_RECORDS/                   # Architecture & AI Decision Records
│   │   ├── ADR-001-database-choice.md
│   │   ├── ADR-002-auth-strategy.md
│   │   └── ...
│   │
│   ├── prompts/                            # AI Prompt Library
│   │   ├── PROMPT_LIBRARY.md               # Master prompt library index
│   │   ├── requirements/                   # Requirements analysis prompts
│   │   ├── design/                         # System design prompts
│   │   ├── coding/                         # Code generation prompts
│   │   ├── testing/                        # Testing prompts
│   │   └── documentation/                  # Documentation prompts
│   │
│   ├── ai_outputs/                         # Raw AI outputs (for transparency)
│   │   ├── requirements/                   # AI outputs for requirements phase
│   │   ├── design/                         # AI outputs for design phase
│   │   ├── code/                           # AI-generated code samples
│   │   └── testing/                        # AI-generated test cases
│   │
│   ├── reviews/                            # Review documentation
│   │   ├── code_reviews/                   # Code review checklists
│   │   └── ai_reviews/                     # AI output review records
│   │
│   ├── testing/                            # Testing documentation
│   │   ├── test_plan.md                    # Overall test strategy
│   │   ├── test_cases/                     # Individual test case documents
│   │   └── test_reports/                   # Test execution reports
│   │
│   ├── templates/                          # Document templates
│   │   ├── AI_USAGE_LOG.md
│   │   ├── PROMPT_LIBRARY.md
│   │   ├── AI_DECISION_RECORD.md
│   │   ├── CODE_REVIEW_CHECKLIST.md
│   │   ├── TESTING_TEMPLATE.md
│   │   ├── PULL_REQUEST_TEMPLATE.md
│   │   ├── COMMIT_CONVENTION.md
│   │   ├── BRANCHING_STRATEGY.md
│   │   ├── AI_GOVERNANCE_CHARTER.md
│   │   └── REPOSITORY_STRUCTURE.md
│   │
│   ├── meeting_notes/                      # Sprint/team meeting notes
│   └── lessons_learned/                    # Retrospective and lessons learned
│
├── scripts/
│   ├── setup.sh                            # Project setup automation
│   └── validate_ai_logs.py                 # Validate AI log completeness
│
├── .gitignore                              # Git ignore rules
├── .editorconfig                           # Editor configuration
├── README.md                               # Project overview and setup
├── CONTRIBUTING.md                         # Contribution guidelines
├── CHANGELOG.md                            # Version changelog
├── LICENSE                                 # Project license
└── requirements.txt / package.json         # Dependencies
```

---

## Directory Details

### `.github/`

| Attribute | Details |
|---|---|
| **Purpose** | GitHub-specific configuration for CI/CD, issue templates, and PR templates |
| **Contents** | Issue templates (bug report, feature request, AI experiment), PR template, GitHub Actions workflows |
| **Naming** | Lowercase files, `.yml` for workflows |
| **Responsibility** | DevOps Lead / Project Lead |

**Files:**
- `PULL_REQUEST_TEMPLATE.md` — Standardized PR form with AI assistance tracking
- `ISSUE_TEMPLATE/bug_report.md` — Bug report template with severity levels
- `ISSUE_TEMPLATE/feature_request.md` — Feature request template with acceptance criteria
- `ISSUE_TEMPLATE/ai_experiment.md` — Template for proposing AI experiments
- `workflows/ci.yml` — Continuous Integration pipeline (build, test, lint)
- `workflows/lint.yml` — Code linting and style enforcement

---

### `src/`

| Attribute | Details |
|---|---|
| **Purpose** | All production source code |
| **Contents** | Divided into `backend/`, `frontend/`, and `shared/` to enforce separation of concerns |
| **Naming** | `camelCase` for variables/functions, `PascalCase` for classes and components |
| **Responsibility** | Development Team (Backend and Frontend Developers) |

#### `src/backend/`
- `controllers/` — Request handlers and routing logic. One file per resource (e.g., `userController.py`)
- `services/` — Business logic layer. Keeps controllers thin
- `models/` — Data models, ORM definitions, and schemas
- `middleware/` — Cross-cutting concerns: auth, logging, error handling, rate limiting
- `utils/` — Shared utility functions (e.g., date formatting, validation helpers)
- `config/` — Environment configuration, database settings, app constants

#### `src/frontend/`
- `components/` — Reusable UI components (e.g., `Button.jsx`, `Modal.jsx`)
- `pages/` — Page-level components corresponding to routes
- `assets/` — Static files: images, fonts, icons
- `styles/` — Global stylesheets, theme files
- `utils/` — Client-side helpers (e.g., API client, formatters)

#### `src/shared/`
- `types/` — TypeScript interfaces or Python type hints shared between frontend and backend
- `constants/` — Application-wide constants and enums

---

### `tests/`

| Attribute | Details |
|---|---|
| **Purpose** | All automated tests |
| **Contents** | Separated by testing level |
| **Naming** | Files should follow `<module>.test.<ext>` or `test_<module>.<ext>` convention |
| **Responsibility** | QA Lead / All Developers |

- `unit/` — Isolated tests for individual functions/classes. No external dependencies
- `integration/` — Tests that verify interaction between modules (e.g., API + Database)
- `e2e/` — End-to-end tests simulating real user workflows
- `fixtures/` — Test data, mock objects, factory functions, and stubs

---

### `docs/`

| Attribute | Details |
|---|---|
| **Purpose** | All project documentation, AI governance artifacts, and academic deliverables |
| **Contents** | Technical docs, AI logs, prompts, reviews, testing reports, templates |
| **Naming** | `Pascal_Snake_Case.md` or `UPPER_SNAKE_CASE.md` for major documents |
| **Responsibility** | Documentation Manager / Entire Team |

#### Core Documents (Root Level)
- `SRS.md` — Software Requirements Specification
- `System_Design.md` — System architecture and design document
- `Architecture.md` — Architecture diagrams, component relationships
- `Database_Design.md` — ERD, table definitions, normalization notes
- `API_Documentation.md` — REST API endpoints, request/response schemas

#### AI Governance Documents
- `AI_GOVERNANCE_CHARTER.md` — Team's AI usage policies and principles
- `AI_USAGE_LOG.md` — **Critical:** All AI interactions documented per lecturer requirements
- `AI_TOOLS_REGISTRY.md` — Approved tools with versions and risk levels

#### `DECISION_RECORDS/`
- Architecture Decision Records documenting significant technical choices
- Format: `ADR-<NNN>-<short-description>.md`
- Includes AI-specific decision records

#### `prompts/`
- `PROMPT_LIBRARY.md` — Master index of reusable team prompts
- Subdirectories organize prompts by SDLC phase
- Each prompt file includes: template, usage notes, effectiveness rating

#### `ai_outputs/`
- Raw or lightly edited AI outputs kept for academic transparency
- Organized by SDLC phase
- Referenced from AI Usage Log entries

#### `reviews/`
- `code_reviews/` — Completed code review checklists
- `ai_reviews/` — AI output review records

#### `testing/`
- `test_plan.md` — Overall testing strategy
- `test_cases/` — Individual test case documents
- `test_reports/` — Test execution reports with results

#### `templates/`
- All document templates used by the team
- Reference copies — actual documents are created from these

#### `meeting_notes/`
- Sprint planning, daily standup, and retrospective notes
- Format: `YYYY-MM-DD-<meeting-type>.md`

#### `lessons_learned/`
- Retrospective documents capturing what worked and what didn't
- Especially valuable for AI-related learnings

---

### `scripts/`

| Attribute | Details |
|---|---|
| **Purpose** | Automation scripts for development, setup, and validation |
| **Contents** | Shell scripts, Python utilities, database seeders |
| **Naming** | `snake_case.<ext>` |
| **Responsibility** | DevOps Lead / Tech Lead |

- `setup.sh` — One-command project setup (install deps, create `.env`, run migrations)
- `validate_ai_logs.py` — Script to verify AI log completeness and format compliance

---

### Root Level Files

| File | Purpose | Responsibility |
|---|---|---|
| `README.md` | Project overview, setup instructions, tech stack | Project Lead |
| `CONTRIBUTING.md` | Contribution guidelines, links to conventions | Project Lead |
| `CHANGELOG.md` | Version history and release notes | Release Manager |
| `LICENSE` | Project license (e.g., MIT, Apache 2.0) | Project Lead |
| `.gitignore` | Files/directories excluded from version control | DevOps Lead |
| `.editorconfig` | Consistent editor settings across IDEs | DevOps Lead |
| `requirements.txt` / `package.json` | Project dependencies | Tech Lead |

---

## Quick Setup Checklist

When initializing a new project, ensure these directories and files exist:

- [ ] `.github/PULL_REQUEST_TEMPLATE.md` copied from `docs/templates/`
- [ ] `src/backend/` and `src/frontend/` directories created
- [ ] `tests/unit/`, `tests/integration/`, `tests/e2e/` directories created
- [ ] `docs/AI_GOVERNANCE_CHARTER.md` completed and signed by team
- [ ] `docs/AI_USAGE_LOG.md` initialized from template
- [ ] `docs/AI_TOOLS_REGISTRY.md` populated with approved tools
- [ ] `docs/DECISION_RECORDS/` directory created
- [ ] `docs/prompts/` directory structure created
- [ ] `docs/ai_outputs/` directory structure created
- [ ] `README.md` populated with project information
- [ ] `.gitignore` configured for the project's tech stack
- [ ] `CONTRIBUTING.md` links to commit convention and branching strategy

---

*This structure is a living document. Adjust it as the project's needs evolve, but maintain the core organizational principles.*
