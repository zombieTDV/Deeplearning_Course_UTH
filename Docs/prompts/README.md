# Docs/prompts

## Purpose
This directory serves as the team's AI Prompt Library — storing effective and reusable prompt templates organized by SDLC phase. Each file includes the member's name for traceability.

## Structure
```
prompts/
├── PROMPT_LIBRARY.md          ← Master template & examples
├── requirements/              ← Prompts for requirements analysis
├── design/                    ← Prompts for architecture & database design
├── coding/                    ← Prompts for code generation & refactoring
├── testing/                   ← Prompts for test case generation
└── documentation/             ← Prompts for docs, README, API specs
```

## Naming Convention

**Format:** `<member-name>-<description>.md`

| Member | Role | Example Filename |
|---|---|---|
| Member 1 | Project Lead / AI Governance | `member1-system-requirements.md` |
| Member 2 | Tech Lead / Backend Dev | `member2-database-schema.md` |
| Member 3 | Backend Developer | `member3-crud-api-generation.md` |
| Member 4 | Frontend Developer | `member4-login-page-component.md` |
| Member 5 | Frontend Dev / QA | `member5-unit-test-generation.md` |
| Member 6 | Doc Manager / QA | `member6-api-documentation.md` |

> ⚠️ **Replace `member1`, `member2`...** with actual names (e.g., `nam-database-schema.md`, `hoa-login-page.md`)

## Prompt File Template

Each prompt file should contain:

```markdown
# Prompt: [Short Title]

- **Author:** [Your Name]
- **Date:** YYYY-MM-DD
- **AI Tool:** ChatGPT / Copilot / Gemini / Claude
- **Category:** Requirements / Design / Coding / Testing / Documentation

## Objective
What you wanted to achieve.

## Original Prompt
> Paste your first prompt here

## Refined Prompt (if improved)
> Paste improved version here

## Why Refined?
What was missing in the original prompt.

## Result Quality (1-5): ⭐⭐⭐⭐
## Reusable: Yes / No
```

## Who Saves Prompts?
**All 6 team members** — every time you use AI, save your prompt in the matching folder with your name in the filename.

## Responsible
- All team members (each saves their own prompts)
- Documentation Manager reviews and organizes monthly
