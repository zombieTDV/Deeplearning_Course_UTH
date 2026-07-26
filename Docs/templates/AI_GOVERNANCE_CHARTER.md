# AI Governance Charter

**Project Name:** [Insert Project Name]  
**Team Name:** [Insert Team Name]  
**Course:** [Insert Course Name]  
**Institution:** University of Transport Ho Chi Minh City (UTH)  
**Date:** [YYYY-MM-DD]  
**Version:** 1.0  
**Status:** Active  

---

## 1. Purpose and Scope

This charter defines the rules, principles, and procedures for utilizing Artificial Intelligence (AI) tools during the development of our capstone project. It ensures:

- **Academic integrity** — AI use is transparent and properly documented
- **Code quality** — AI-generated content meets project standards
- **Security** — No sensitive data is exposed to AI tools
- **Accountability** — Every team member takes responsibility for AI-generated content they use
- **Traceability** — Complete audit trail from prompt to final commit

This charter applies to all team members and covers all phases of the Software Development Life Cycle (SDLC).

---

## 2. AI Governance Principles

### 2.1 Transparency
All use of AI must be openly declared and documented in:
- Commit messages (AI metadata fields)
- Pull Request descriptions
- AI Usage Log (`docs/AI_USAGE_LOG.md`)
- AI Decision Records (for significant decisions)

### 2.2 Accountability
The human developer is **always 100% responsible** for the code they commit, regardless of whether AI generated it. AI is a tool; the developer is the author.

### 2.3 Human Oversight
No AI-generated code will be merged into `main` or `develop` without:
- Manual review and comprehension
- Testing (unit tests at minimum)
- Peer review via Pull Request
- Documentation in AI Usage Log

### 2.4 Data Privacy
No sensitive information will be shared with external AI services, including:
- Passwords, API keys, JWT secrets
- Database credentials
- Personally Identifiable Information (PII)
- Proprietary university data

### 2.5 Intellectual Integrity
- AI is a tool for assistance, not a replacement for understanding
- Developers must be able to explain every line of code they commit
- Claiming AI-generated code as original work without documentation is an academic integrity violation
- During project defense, every team member must demonstrate understanding of their code

### 2.6 Quality Assurance
AI-generated code is subject to the same (or stricter) standards as human-written code:
- Must pass all linting rules
- Must have associated tests
- Must follow project coding conventions
- Must be reviewed for security vulnerabilities

### 2.7 Ethical Use
AI will not be used to:
- Generate malicious code
- Bypass security controls
- Plagiarize or violate licenses
- Create deceptive or harmful content

---

## 3. Approved AI Tools Registry

### 3.1 Approved Tools

| Tool | Version/Model | License/Tier | Approved Use Cases | Data Policy | Risk Level |
|---|---|---|---|---|---|
| ChatGPT | GPT-4o / GPT-5 | Free/Plus | Brainstorming, debugging, documentation, design | Do NOT paste sensitive data | Medium |
| GitHub Copilot | Latest | Student/Pro | Code autocomplete, boilerplate generation | Opt-out of training recommended | Low |
| Google Gemini | 2.x / 2.5 Pro | Free/Pro | Research, documentation, code review | Do NOT paste sensitive data | Medium |
| Claude | Sonnet 4 / Opus 4 | Free/Pro | Complex reasoning, refactoring, architecture | Do NOT paste sensitive data | Medium |
| Cursor | Latest | Free/Pro | AI-assisted IDE coding | Check data retention policy | Medium |
| [Add New Tool] | | | | | |

### 3.2 Requesting New Tools

To request approval for a new AI tool:

1. Submit a proposal to the Project Lead including:
   - Tool name and version
   - License/pricing information
   - Intended use cases
   - Data privacy policy review
   - Risk assessment
2. Project Lead reviews and decides within 48 hours.
3. If approved, add to the registry table above.
4. If rejected, document the reason.

---

## 4. AI Usage Classifications

We classify AI usage into four levels to determine required documentation and review depth:

### Level 1 — Reference Only
- **Description:** Using AI like a search engine to understand concepts, syntax, or best practices.
- **Examples:** "What is the difference between REST and GraphQL?", "Explain Python decorators"
- **Documentation Required:** None (unless the insight directly shapes a decision)
- **Review Required:** None

### Level 2 — Assisted
- **Description:** AI suggests snippets, autocompletes lines, or generates basic boilerplate. Human implements the core logic.
- **Examples:** Copilot autocomplete, simple function scaffolding
- **Documentation Required:** Commit message AI metadata
- **Review Required:** Standard code review

### Level 3 — Augmented
- **Description:** AI generates significant blocks of logic, algorithms, or whole components. Human reviews, modifies, and tests.
- **Examples:** AI generates a complete CRUD API, database schema, or React component
- **Documentation Required:** Commit message AI metadata + AI Usage Log entry + Prompt saved
- **Review Required:** Detailed code review + AI-specific review checklist

### Level 4 — Automated
- **Description:** AI generates complete solutions with minimal to no human modification.
- **Documentation Required:** All Level 3 requirements + AI Decision Record (ADR)
- **Review Required:** **REQUIRES ADDITIONAL PEER REVIEW** + explicit approval from Project Lead
- **⚠️ Warning:** Level 4 usage should be rare. Excessive Level 4 usage indicates over-reliance on AI.

---

## 5. Data Sensitivity Classification

### 5.1 Data You CAN Share with AI Tools

| Category | Examples |
|---|---|
| Public knowledge | Programming concepts, language syntax, design patterns |
| Generic code patterns | Algorithm implementations, UI component structures |
| Open-source code | Publicly available library usage examples |
| Sanitized code | Code with all secrets, PII, and proprietary info removed |
| Project requirements | General feature descriptions (non-confidential) |

### 5.2 Data You MUST NOT Share with AI Tools

| Category | Examples | Risk |
|---|---|---|
| Credentials | Passwords, API keys, tokens, secrets | Security breach |
| PII | Student names, IDs, emails, phone numbers | Privacy violation |
| Database content | Production data, user records | Data leak |
| Proprietary algorithms | University-owned IP, exam content | IP violation |
| Configuration secrets | `.env` files, server configurations | Infrastructure risk |

### 5.3 Data Handling Procedures

1. **Before pasting code into AI tools:**
   - Remove all hardcoded credentials and secrets
   - Replace PII with placeholder data
   - Remove any comments referencing internal systems
   
2. **Use `.env` files** for all secrets and ensure they are in `.gitignore`

3. **Use environment variables** in code — never hardcode sensitive values

4. **When in doubt, don't share** — ask the Project Lead first

---

## 6. Accountability Matrix (RACI)

*R = Responsible, A = Accountable, C = Consulted, I = Informed*

| Activity | Developer | Reviewer | Project Lead | QA Lead | Doc Manager |
|---|---|---|---|---|---|
| Craft AI prompts | R, A | I | I | I | I |
| Review AI output before use | R, A | C | I | I | I |
| Write/modify code from AI output | R, A | C | I | I | I |
| Test AI-generated code | R | C | I | A | I |
| Document AI usage (commit msg) | R, A | C | I | I | I |
| Document AI usage (AI log) | R | I | C | I | A |
| Create AI Decision Records | R | C | A | I | I |
| Peer review AI-assisted code | I | R, A | C | C | I |
| Audit AI governance compliance | I | C | R, A | C | C |
| Approve Level 4 AI usage | I | C | R, A | C | I |
| Maintain Prompt Library | R | I | I | I | A |
| Final AI usage report | C | I | A | C | R |

---

## 7. AI Output Risk Assessment

### 7.1 Risk Levels

| Risk Level | Criteria | Examples |
|---|---|---|
| **Low** | Cosmetic, non-functional, documentation | CSS styling, README formatting, comments |
| **Medium** | Functional but non-critical logic | UI components, utility functions, data formatting |
| **High** | Core business logic, data processing | API endpoints, database queries, algorithms |
| **Critical** | Security, authentication, data handling | Auth flows, encryption, input validation, payments |

### 7.2 Required Review Depth by Risk Level

| Risk Level | Review Requirement | Testing Requirement | Documentation |
|---|---|---|---|
| **Low** | Standard code review | Manual verification | Commit message |
| **Medium** | Detailed code review | Unit tests required | Commit message + AI log |
| **High** | Line-by-line review + peer review | Unit + integration tests | Full traceability chain |
| **Critical** | Security audit + Project Lead approval | Unit + integration + security tests | Full traceability + ADR |

### 7.3 Escalation Procedures

1. **If unsure about risk level:** Treat it as the next higher level.
2. **If security concern is identified:** Stop, revert if already committed, and notify Project Lead.
3. **If AI output seems too good to be true:** Verify independently — AI can hallucinate APIs, libraries, and functions.
4. **If licensing concern arises:** Do not use the code. Research the original source.

---

## 8. Compliance Requirements

### 8.1 Lecturer Requirements Mapping

| Lecturer Requirement | How We Comply | Document/Location |
|---|---|---|
| 1. AI tool/model/version used | AI Tools Registry + per-record logging | `AI_TOOLS_REGISTRY.md`, `AI_USAGE_LOG.md` |
| 2. Date, objective, context | Every log entry includes date, objective, context | `AI_USAGE_LOG.md` |
| 3. Original and refined prompts | Prompt evolution documented | `prompts/`, `AI_USAGE_LOG.md` |
| 4. Related source files | Affected components listed per entry | `AI_USAGE_LOG.md` |
| 5. AI responses | Response summaries saved | `AI_USAGE_LOG.md`, `ai_outputs/` |
| 6. Accepted/modified/rejected | Decision field in every log entry | `AI_USAGE_LOG.md` |
| 7. Reasons for modifications | Rationale documented | `AI_USAGE_LOG.md`, ADRs |
| 8. Verification methods | V&V section in every log entry | `AI_USAGE_LOG.md` |
| 9. Related Git commits | Commit hash linked in log entries | `AI_USAGE_LOG.md`, Git history |

### 8.2 Academic Integrity Statement

> This team uses AI tools as assistive instruments in our software development process. All AI-generated content is reviewed, understood, modified where necessary, and tested by team members before integration. Team members take full responsibility for the accuracy, security, quality, and operability of the final product. No AI output is used without human comprehension and approval.

### 8.3 License Compliance

- We will not knowingly use AI-generated code that infringes on copyrights or licenses.
- When AI suggests code that closely mirrors open-source implementations, we will verify the license and provide attribution if required.
- We use AI tools in compliance with their respective Terms of Service.

---

## 9. Incident Response

### 9.1 What Constitutes an Incident

- AI-generated code introduces a **security vulnerability** (e.g., SQL injection, XSS)
- AI-generated code causes a **production bug** or system failure
- AI-generated code **violates a license** or copyright
- A team member **commits Level 4 code without proper review**
- Sensitive data is **accidentally shared** with an AI tool

### 9.2 Response Procedures

1. **Contain:** Immediately revert the offending commit/PR.
   ```bash
   git revert <commit-hash>
   ```

2. **Report:** Notify the Project Lead within 1 hour. Document the incident:
   - What happened
   - Which AI tool was involved
   - What code was affected
   - Potential impact

3. **Analyze:** Conduct a root cause analysis:
   - Why did the review process fail to catch the issue?
   - Was the AI output risk level correctly assessed?
   - Was the AI Usage Log properly maintained?

4. **Remediate:**
   - Fix the immediate issue
   - Update prompts/approach to prevent recurrence
   - Update this charter if process gaps are found
   - Retrain team members if needed

5. **Document:** Create an ADR documenting the incident and lessons learned.

### 9.3 Post-Incident Checklist

- [ ] Offending code has been reverted
- [ ] Project Lead has been notified
- [ ] Root cause has been identified
- [ ] Fix has been implemented and tested
- [ ] ADR has been created
- [ ] Team has been briefed on lessons learned
- [ ] Charter has been updated if necessary

---

## 10. Team Acknowledgment

By signing below, team members acknowledge they have:
- ✅ Read and understood this AI Governance Charter
- ✅ Agreed to follow all principles and procedures outlined herein
- ✅ Understood their accountability for AI-generated content
- ✅ Committed to maintaining transparency in AI usage
- ✅ Acknowledged that violations may result in academic consequences

| # | Full Name | Student ID | Role | Date | Signature / Initial |
|---|---|---|---|---|---|
| 1 | | | Project Lead | | |
| 2 | | | Tech Lead | | |
| 3 | | | Backend Developer | | |
| 4 | | | Backend Developer | | |
| 5 | | | Frontend Developer | | |
| 6 | | | Frontend Developer | | |
| 7 | | | QA Lead | | |
| 8 | | | Documentation Manager | | |
| 9 | | | | | |
| 10 | | | | | |

---

*This charter is a living document. Any amendments must be approved by the Project Lead and acknowledged by all team members.*

*Last Updated: [YYYY-MM-DD]*  
*Next Review Date: [YYYY-MM-DD]*
