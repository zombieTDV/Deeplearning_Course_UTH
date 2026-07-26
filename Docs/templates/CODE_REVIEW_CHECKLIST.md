# Code Review Checklist

This checklist includes standard software engineering practices alongside specific checks for AI-assisted development.

**Review ID:** CR-YYYY-MM-DD-NNN  
**Reviewer:** [Name]  
**Author:** [Name]  
**Date:** [Date]  
**PR/Branch Reference:** [Link or branch name]  
**Files Reviewed:** [List of critical files]  

---

## 1. AI Traceability
*Ensure all AI usage is transparent and documented.*

- [ ] AI usage is documented in `AI_USAGE_LOG.md` (if applicable for significant logic).
- [ ] Original and refined prompts are saved in the log.
- [ ] AI tool and version are recorded.
- [ ] AI Decision Record (ADR) exists for significant AI-driven architectural or design outputs.
- [ ] Commit messages include AI assistance metadata (e.g., "Co-authored-by: AI").
- [ ] Developer can clearly and fully explain the AI-generated code.
- [ ] Traceability chain is complete (Prompt → Output → Modification → Commit).

## 2. Code Quality
*Standard software engineering best practices.*

- [ ] Code is readable, well-structured, and easy to understand.
- [ ] Naming conventions (variables, functions, classes) are followed consistently.
- [ ] No unnecessary complexity or "clever" code.
- [ ] Single Responsibility Principle (SRP) is maintained.
- [ ] Error handling is appropriate and fails gracefully.
- [ ] No dead code, unused variables, or commented-out code blocks.
- [ ] DRY (Don't Repeat Yourself) principle is followed; code is reusable.
- [ ] Code is properly documented/commented, especially complex logic.
- [ ] No magic numbers or hardcoded configuration values.
- [ ] Consistent coding style (matches linter/formatter rules).

## 3. Security Review
*Based on OWASP Top 10 and general security practices.*

- [ ] No SQL injection vulnerabilities (use ORMs or parameterized queries).
- [ ] Input validation is present on all client and server boundaries.
- [ ] No hardcoded credentials, API keys, or secrets in the code.
- [ ] Proper authentication and authorization checks exist on endpoints/functions.
- [ ] CSRF protection is implemented for state-changing requests.
- [ ] XSS prevention (output encoding, escaping) is applied.
- [ ] Secure session management (e.g., HTTP-only, secure flags on cookies).
- [ ] Proper error handling (no sensitive stack traces exposed to users).
- [ ] Dependencies added are up to date and scanned for known vulnerabilities.
- [ ] Sensitive data (PII, passwords) is properly handled and encrypted/hashed.

## 4. Testing
*Ensuring reliability and correctness.*

- [ ] Unit tests exist for new code and core logic.
- [ ] All tests pass successfully locally and in CI pipeline.
- [ ] Edge cases, boundary values, and negative scenarios are covered.
- [ ] Test coverage meets the project's minimum threshold (e.g., 80%).
- [ ] Integration tests are updated or added if module interactions changed.
- [ ] Test code quality matches production code quality (DRY, maintainable).

## 5. AI-Specific Concerns
*Validating the specific risks associated with LLM-generated code.*

- [ ] AI-generated code does NOT contain hallucinated (non-existent) APIs, libraries, or methods.
- [ ] AI-generated code uses the project's established technology stack and architectural patterns.
- [ ] No license-violating code patterns (ensure no exact copies of proprietary code).
- [ ] AI-generated comments and documentation are accurate and actually match the code logic.
- [ ] Code does not contain leftover AI artifacts (e.g., `Here is the code you requested:`, ````python`, conversation markers).
- [ ] Performance and algorithmic complexity of AI suggestions are acceptable for the use case.
- [ ] Code integrates seamlessly with the existing codebase without breaking encapsulation.
- [ ] No over-engineering (AI sometimes suggests complex design patterns for simple problems).

---

## Summary and Sign-off

**Overall Assessment:**
[ ] APPROVE - Code is ready to merge.
[ ] REQUEST CHANGES - Minor issues to address.
[ ] REJECT - Significant rework required.

**Reviewer Comments:**
> [Add detailed comments, identifying specific lines or files that need attention, or praising good work.]

**Sign-off:** ___________________________ (Reviewer) Date: ____________
