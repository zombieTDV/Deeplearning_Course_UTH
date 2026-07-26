# Commit Message Convention

## Purpose and Importance
Consistent and well-structured commit messages are essential for our team. They help us:
- Automatically generate changelogs.
- Understand the history and intent of changes without looking at the code.
- Track AI assistance and hold developers accountable for AI-generated code.
- Streamline the review and debugging process.

## Format Specification
We use a format based on Conventional Commits, extended with AI metadata in the body.

```text
<type>(<scope>): <subject>

<body>

AI-Assisted: Yes | No
AI-Tool: <tool-name> <model-version>
AI-Contribution: <what AI helped with>
Human-Modification: <what developer changed>
Verified-By: <verification method>

Refs: #<issue-number>
Reviewed-by: <reviewer>
```

## Type Definitions

| Type | Description |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Changes that do not affect the meaning of the code (white-space, formatting, etc.) |
| `refactor` | A code change that neither fixes a bug nor adds a feature |
| `perf` | A code change that improves performance |
| `test` | Adding missing tests or correcting existing tests |
| `build` | Changes that affect the build system or external dependencies |
| `ci` | Changes to our CI configuration files and scripts |
| `chore` | Other changes that don't modify src or test files |
| `revert` | Reverts a previous commit |

## Scope Examples
Scopes should be specific to the project's architecture. Examples:
- `auth`: Authentication module
- `api`: API endpoints
- `db`: Database models/migrations
- `ui`: Frontend user interface components
- `docs`: Documentation files

## Subject Line Rules
- Limit to 50 characters.
- Use imperative mood ("add feature" not "added feature").
- Do not capitalize the first letter.
- Do not end with a period.

## Body Writing Guidelines
- Wrap at 72 characters.
- Explain *what* and *why* instead of *how*.
- Include the AI metadata fields exactly as specified if AI was used. If AI was not used, simply omit the AI fields or set `AI-Assisted: No`.

## AI Metadata Fields Explanation
- **AI-Assisted**: Must be `Yes` or `No`.
- **AI-Tool**: E.g., `ChatGPT (GPT-4)`, `GitHub Copilot`.
- **AI-Contribution**: Specifically what the AI did (e.g., "Generated regex for email validation").
- **Human-Modification**: How the AI output was changed (e.g., "Adjusted regex to support subdomains").
- **Verified-By**: How you ensured the code works (e.g., "Unit tests, manual review").

## Breaking Changes
If a commit introduces a breaking change, include a `BREAKING CHANGE:` block at the very beginning of the body or footer, explaining the change and migration path. The commit type/scope should be followed by a `!`.
Example: `feat(api)!: remove v1 endpoints`

---

## Examples

### 1. Feature with AI assistance
```text
feat(auth): implement JWT token refresh mechanism

Added an endpoint to refresh expired JWTs securely using http-only cookies.

AI-Assisted: Yes
AI-Tool: GitHub Copilot
AI-Contribution: Generated boilerplate for the Express route and JWT verification logic.
Human-Modification: Moved secret key to environment variables and added error handling for token mismatch.
Verified-By: Added integration tests for token refresh flow.

Refs: #42
```

### 2. Feature without AI assistance
```text
feat(ui): add loading spinner to dashboard

Added a CSS-only loading spinner component to improve UX during data fetch.

AI-Assisted: No

Refs: #45
```

### 3. Bug fix found by AI
```text
fix(api): prevent race condition in user creation

Resolved a race condition where double-clicking the submit button created duplicate users.

AI-Assisted: Yes
AI-Tool: Claude 3.5 Sonnet
AI-Contribution: Identified the race condition from the log file and suggested using a database constraint.
Human-Modification: Implemented a unique constraint in the DB migration rather than application-level locking.
Verified-By: Load testing script with concurrent requests.

Refs: #51
```

### 4. Bug fix found manually
```text
fix(ui): correct modal alignment on mobile

Adjusted flexbox properties on the login modal to prevent cutoff on screens < 400px.

AI-Assisted: No

Refs: #55
```

### 5. Documentation update
```text
docs(readme): add setup instructions for local dev

AI-Assisted: Yes
AI-Tool: ChatGPT (GPT-4)
AI-Contribution: Drafted the markdown instructions based on the setup shell script.
Human-Modification: Corrected the Node version requirement.
Verified-By: Manual review.
```

### 6. Refactoring with AI suggestion
```text
refactor(db): extract query logic to repository pattern

AI-Assisted: Yes
AI-Tool: ChatGPT (GPT-4)
AI-Contribution: Suggested moving direct Prisma calls from the controller to a dedicated repository class.
Human-Modification: Implemented the suggestion and updated all controller injections.
Verified-By: Existing unit tests passed.
```

### 7. Test addition
```text
test(auth): add edge case tests for password reset

AI-Assisted: Yes
AI-Tool: GitHub Copilot
AI-Contribution: Generated test cases for expired tokens and invalid email formats.
Human-Modification: Updated mock assertions to match our testing framework conventions.
Verified-By: Jest test suite run.
```

### 8. Security fix
```text
fix(api): sanitize user input in search query

Parameterized the search query to prevent SQL injection via the search bar.

AI-Assisted: No

Refs: #60
```

### 9. Database migration
```text
feat(db): add profiles table

AI-Assisted: Yes
AI-Tool: GitHub Copilot
AI-Contribution: Generated SQL schema based on the Profile model class.
Human-Modification: Added indexing to the user_id column.
Verified-By: Ran migration locally.
```

### 10. API endpoint creation
```text
feat(api): create GET /users endpoint

AI-Assisted: Yes
AI-Tool: GitHub Copilot
AI-Contribution: Generated route and basic pagination logic.
Human-Modification: Added validation middleware for query parameters.
Verified-By: Postman manual testing.
```

### 11. Frontend component
```text
feat(ui): implement DataTable component

AI-Assisted: Yes
AI-Tool: Claude 3.5 Sonnet
AI-Contribution: Provided a React component structure for sortable tables.
Human-Modification: Replaced generic styling with Tailwind CSS classes to match our design system.
Verified-By: Storybook visual verification.
```

### 12. CI/CD configuration
```text
ci(actions): add python linting workflow

AI-Assisted: Yes
AI-Tool: ChatGPT (GPT-4)
AI-Contribution: Wrote the GitHub Actions YAML file for running flake8.
Human-Modification: Changed the python version to 3.11.
Verified-By: Pushed to a test branch and verified action success.
```

### 13. Dependency update
```text
build(deps): bump react from 18.2 to 18.3

AI-Assisted: No
```

### 14. AI usage log update
```text
docs(governance): update AI usage log for sprint 2

AI-Assisted: No
```

### 15. Rejected AI suggestion (documented why)
```text
fix(db): optimize slow analytical query

Rewrote the aggregate query to use window functions instead of subqueries.

AI-Assisted: Yes
AI-Tool: ChatGPT (GPT-4)
AI-Contribution: Suggested adding a materialized view.
Human-Modification: Rejected the suggestion because data freshness is critical. Implemented window functions instead.
Verified-By: EXPLAIN ANALYZE shows 40% performance improvement.
```

---

## Anti-Patterns

**Bad:** `Fixed bug`
**Correction:** `fix(ui): resolve null pointer exception in navigation bar`

**Bad:** `Update login` (No context, starts with capital letter)
**Correction:** `feat(auth): add social login providers`

**Bad:** `chore: added stuff from chatgpt`
**Correction:** Requires the full AI metadata block in the body to document exactly what was added and how it was verified.

**Bad:** `update` / `fix` / `change` / `test`
**Correction:** Always include type, scope, and a descriptive imperative subject.

---

## Git Hooks for Validation

To enforce this convention, we recommend using `commitlint` and `husky` to run a `commit-msg` hook that validates the format before allowing the commit.

### Setup with Husky (Node.js projects)

```bash
# Install commitlint and husky
npm install --save-dev @commitlint/cli @commitlint/config-conventional husky

# Setup husky
npx husky init

# Create commitlint config
echo "module.exports = { extends: ['@commitlint/config-conventional'] };" > commitlint.config.js

# Add commit-msg hook
echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg
```

### Setup with pre-commit (Python projects)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```
