# Contributing Guidelines

Welcome to the project! Please follow these guidelines when contributing.

---

## 🔀 Branching Strategy

We use a modified **GitHub Flow**. See [Branching Strategy](Docs/templates/BRANCHING_STRATEGY.md) for full details.

| Branch | Purpose | Merge Into |
|---|---|---|
| `main` | Stable, production-ready code | — |
| `develop` | Integration branch | `main` |
| `feature/*` | New features | `develop` |
| `bugfix/*` | Bug fixes | `develop` |
| `hotfix/*` | Emergency fixes | `main` |
| `docs/*` | Documentation only | `develop` |
| `experiment/*` | AI experiments (do NOT merge directly) | — |

**Branch naming:** `<type>/<issue-number>-<short-description>`

---

## 📝 Commit Messages

We follow **Conventional Commits** with AI metadata. See [Commit Convention](Docs/templates/COMMIT_CONVENTION.md).

```
<type>(<scope>): <subject>

<body>

AI-Assisted: Yes | No
AI-Tool: <tool-name> <model-version>
AI-Contribution: <what AI helped with>
Human-Modification: <what you changed>
Verified-By: <how you tested it>
```

---

## 🔄 Pull Request Process

1. Create a feature branch from `develop`
2. Make your changes and commit following the convention
3. Push and open a PR using the [PR Template](.github/PULL_REQUEST_TEMPLATE.md)
4. Fill in the AI Assistance section if AI was used
5. Request a review from at least 1 team member
6. Wait for CI to pass and approval before merging

---

## 🤖 AI Usage Rules

- **Log every AI interaction** in `Docs/AI_USAGE_LOG.md`
- **Save prompts** in `Docs/prompts/`
- **Never commit AI code without reviewing it first**
- **You must understand every line** you commit
- See [AI Governance Charter](Docs/templates/AI_GOVERNANCE_CHARTER.md) for full policy

---

## 👥 Team (6 Members)

| # | Role | Responsibilities |
|---|---|---|
| 1 | Project Lead / AI Governance Officer | Process compliance, AI policy, sprint planning |
| 2 | Tech Lead / Backend Developer | Architecture decisions, backend development |
| 3 | Backend Developer | API development, database, business logic |
| 4 | Frontend Developer | UI components, pages, client-side logic |
| 5 | Frontend Developer / QA Tester | UI development, testing |
| 6 | Documentation Manager / QA Tester | Docs, AI logs, testing |
