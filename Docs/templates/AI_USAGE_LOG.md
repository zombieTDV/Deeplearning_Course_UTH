# AI Usage Log

**Project Name:** [Insert Project Name]  
**Team:** [Insert Team Name]  
**Repository URL:** [Insert Repository URL]  

## AI Tools Registry

| Tool | Version/Model | Purpose | Risk Level | Data Policy |
|------|---------------|---------|------------|-------------|
| ChatGPT | GPT-4o | General coding, design assistance | Medium | Do not share sensitive data |
| GitHub Copilot | Latest | Inline code generation | Low | IDE integration only |
| Claude | 3.5 Sonnet | Architectural review, complex refactoring | Medium | Do not share sensitive data |

---

## Usage Record Template

Copy and fill out this template for each significant AI usage instance.

### Record ID: AIL-YYYY-MM-DD-NNN
- **Date and Time:** 
- **Team Member:** 
- **AI Tool and Model Version:** 
- **SDLC Phase:** [Requirements | Design | Implementation | Testing | Documentation]
- **Objective/Task Description:** 
- **Context:** 
- **Related Components/Files:** 
- **Original Prompt:** 
- **Refined Prompt:** 
- **AI Response Summary:** 
- **Decision:** [Accepted | Partially Accepted | Modified | Rejected]
- **Accepted Parts:** 
- **Modified Parts:** 
- **Rejected Parts:** 
- **Modification Details:** 
- **Reason for Decision:** 
- **Verification Method:** 
- **Verification Results:** 
- **Security Review:** 
- **Related Git Commit(s):** 
- **Traceability:** 
- **Lessons Learned:** 

---

## Example Records

### Record ID: AIL-2023-10-25-001
- **Date and Time:** 2023-10-25 14:30
- **Team Member:** Jane Doe
- **AI Tool and Model Version:** ChatGPT (GPT-4)
- **SDLC Phase:** Implementation
- **Objective/Task Description:** Write a robust input validation function for user registration.
- **Context:** We need to validate email, password strength, and username formats before sending data to the backend API.
- **Related Components/Files:** `src/utils/validators.ts`, `src/components/RegistrationForm.tsx`
- **Original Prompt:** Write a regex for email and password validation in TypeScript.
- **Refined Prompt:** Write a TypeScript module exported as `validateRegistrationInfo(email, password, username)`. Email must be RFC 5322 compliant. Password must be at least 12 chars, 1 uppercase, 1 number, 1 special char. Username must be alphanumeric, 3-20 chars. Include detailed error messages for each failure.
- **AI Response Summary:** Provided a complete TypeScript module with regex patterns and a returning structure containing a boolean `isValid` and a `string[]` of error messages.
- **Decision:** Partially Accepted
- **Accepted Parts:** The overall module structure, the username regex, and the error reporting mechanism.
- **Modified Parts:** The email regex.
- **Rejected Parts:** The original RFC 5322 regex was overly complex and prone to catastrophic backtracking.
- **Modification Details:** Replaced the AI's email regex with a simpler, standard HTML5 email validation regex, and added a check for string length.
- **Reason for Decision:** Security and performance. Complex regexes can lead to ReDoS attacks. The simpler regex covers 99% of valid cases safely.
- **Verification Method:** Unit testing with valid and invalid inputs, including edge cases and long strings.
- **Verification Results:** Pass. All 25 test cases succeeded. No ReDoS vulnerability detected.
- **Security Review:** Validated against OWASP recommendations for input validation. Safe.
- **Related Git Commit(s):** `a1b2c3d` - "feat: add registration input validators"
- **Traceability:** REQ-Auth-002, TC-VAL-01 to TC-VAL-25
- **Lessons Learned:** Always review AI-generated regex for potential ReDoS vulnerabilities.

### Record ID: AIL-2023-10-26-002
- **Date and Time:** 2023-10-26 10:15
- **Team Member:** John Smith
- **AI Tool and Model Version:** Claude 3.5 Sonnet
- **SDLC Phase:** Design
- **Objective/Task Description:** Design database schema for the inventory system.
- **Context:** System needs to track items, categories, suppliers, and stock transactions across multiple warehouses.
- **Related Components/Files:** `docs/database_schema.md`, `src/db/migrations/001_init.sql`
- **Original Prompt:** Design a database for an inventory system tracking items in warehouses.
- **Refined Prompt:** Create a relational database schema in PostgreSQL for an inventory system. Include tables for Items, Categories, Suppliers, Warehouses, and StockTransactions. We need to track historical stock movements and current stock levels per warehouse. Apply normalization (3NF) and suggest appropriate indexes.
- **AI Response Summary:** Provided SQL DDL statements for 5 tables with primary/foreign keys, plus a trigger-based approach for updating current stock levels based on transactions. Suggested indexes on foreign keys.
- **Decision:** Modified
- **Accepted Parts:** Tables, relationships, and basic indexes.
- **Modified Parts:** The method of tracking current stock.
- **Rejected Parts:** The trigger-based current stock update.
- **Modification Details:** Removed the triggers and the `current_stock` column on the `Items` table. Instead, created a materialized view for current stock levels to be refreshed periodically, or calculated on the fly.
- **Reason for Decision:** Triggers can cause hidden side-effects and concurrency issues in high-transaction environments. A materialized view or direct query approach is more predictable and maintainable for our expected load.
- **Verification Method:** Peer review of the schema (ADR-2023-10-26-001) and DB initialization tests.
- **Verification Results:** Pass.
- **Security Review:** N/A (Internal schema structure, no direct user input involved).
- **Related Git Commit(s):** `f9e8d7c` - "docs: add initial inventory db schema"
- **Traceability:** REQ-Inv-001, Design-DB-01
- **Lessons Learned:** AI often suggests "clever" database features like triggers which might violate team architectural standards.

---

## Monthly Summary Statistics

| Month/Year | Total AI Queries | Accepted As-Is | Modified | Rejected | Critical Defects Found in AI Code | Time Saved (Est. Hours) |
|------------|------------------|----------------|----------|----------|-----------------------------------|-------------------------|
| Oct 2023   | 45               | 10 (22%)       | 30 (67%) | 5 (11%)  | 0                                 | 20                      |
| Nov 2023   |                  |                |          |          |                                   |                         |
