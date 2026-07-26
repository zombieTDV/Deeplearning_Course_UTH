# Test Plan

> **Project:** Deep Learning Course — UTH  
> **Version:** 1.0  
> **Date:** 2026-07-26  
> **QA Lead:** [Name]  

---

## Test Strategy

| Test Type | Scope | Tools | Responsible |
|---|---|---|---|
| Unit Tests | Individual functions/classes | pytest / jest | All Developers |
| Integration Tests | Cross-module interactions | pytest / supertest | Backend Developers |
| E2E Tests | Full user workflows | Selenium / Cypress | QA Testers |
| Security Tests | OWASP Top 10 checks | Manual + bandit | Tech Lead |
| AI Output Tests | Verify AI-generated code | Manual review + tests | Code Author |

---

## Test Environment

| Component | Details |
|---|---|
| OS | Linux / macOS / Windows |
| Runtime | Python 3.x / Node.js 18+ |
| Database | MySQL / PostgreSQL |
| CI/CD | GitHub Actions |

---

## Coverage Goals

| Area | Target |
|---|---|
| Overall code coverage | ≥ 70% |
| AI-generated code coverage | ≥ 80% |
| Critical paths (auth, data) | 100% |

---

## Test Cases

See `test_cases/` directory for individual test case documents.

Use the [Testing Template](templates/TESTING_TEMPLATE.md) for creating new test cases.
