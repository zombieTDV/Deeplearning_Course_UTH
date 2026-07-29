# Contributing Guidelines

Welcome to the project! Please follow these guidelines when contributing to the Deep Learning project.

---

## 🔀 Branching & Commit Conventions

- **Branching**: Use feature branches named `feature/<short-description>` or `bugfix/<short-description>`.
- **Commits**: Follow conventional commits formatted as `<type>(<scope>): <subject>`. Include AI metadata if AI assisted with the code:

```text
<type>(<scope>): <subject>

AI-Assisted: Yes | No
AI-Tool: <tool-name>
Verified-By: <how you tested it (e.g., Smoke Test)>
```

---

## 🤖 AI Usage Rules

- Follow the 10-step Vibe Coding Workflow in [`agents/WORKFLOW.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/WORKFLOW.md).
- Follow project naming conventions & folder standards in [`agents/AI_REFERENCE.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/AI_REFERENCE.md).
- Track task status in [`agents/PROGRESS_TRACKING.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/PROGRESS_TRACKING.md).
- Run the fast smoke test (`python3 tests/smoke_test.py`) after code generation.
- **Never commit AI code without reviewing it first**.
