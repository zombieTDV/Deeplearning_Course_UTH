# AI Decision Record (ADR)

Use this document to record major architectural, design, or implementation decisions where AI assistance played a significant role. 

---

## ADR ID: ADR-YYYY-MM-DD-NNN
**Title:** [Short, descriptive title]
**Date:** [YYYY-MM-DD]
**Status:** [Proposed | Accepted | Deprecated | Superseded]
**Decision Maker(s):** [Names of team members involved]
**SDLC Phase:** [Design / Implementation etc.]
**Related AI Usage Log Entry:** [AIL-YYYY-MM-DD-NNN]

### 1. Context and Problem Statement
[Describe the technical problem being solved. Why do we need to make a decision?]

### 2. AI Consultation Details
- **AI Tool Used:** [e.g., ChatGPT-4, Claude]
- **Prompt Used:** [Reference Prompt Library if applicable, or paste prompt]
- **AI's Proposed Solution Summary:** [What did the AI suggest?]

### 3. Considered Alternatives
- **Option A: AI's Suggestion**
  - Pros: 
  - Cons: 
- **Option B: Manual / Traditional Approach**
  - Pros: 
  - Cons: 
- **Option C: Hybrid / Alternative Approach**
  - Pros: 
  - Cons: 

### 4. Decision Outcome
- **Chosen approach:** [Which option did you pick?]
- **What was accepted from AI:** 
- **What was modified:** 
- **What was rejected:** 

### 5. Rationale
- **Technical justification:** [Why this is the best technical path]
- **Security considerations:** [How security is maintained]
- **Performance considerations:** [Impact on speed/scale]
- **Maintainability considerations:** [Impact on future code changes]

### 6. Consequences
- **Positive consequences:** [What gets better]
- **Negative consequences / risks:** [What trade-offs we accepted]
- **Follow-up actions needed:** [New tasks created based on this decision]

### 7. Verification Plan
- **How the decision will be verified:** [Code review, performance test, etc.]
- **Success criteria:** [Measurable goal]
- **Test cases:** [References to specific test cases]

### 8. Traceability
- **Related requirement(s):** [REQ-ID]
- **Related commit(s):** [Commit hash]
- **Related test case(s):** [TC-ID]

---

## Example 1: Database Schema Design Decision

### ADR ID: ADR-2023-11-01-001
**Title:** Use materialized views for real-time inventory aggregation instead of triggers
**Date:** 2023-11-01
**Status:** Accepted
**Decision Maker(s):** John Smith, Jane Doe
**SDLC Phase:** Design
**Related AI Usage Log Entry:** AIL-2023-10-26-002

### 1. Context and Problem Statement
We need to efficiently display real-time stock levels for thousands of items across multiple warehouses. Calculating this on-the-fly by summing transaction history is too slow for the dashboard view.

### 2. AI Consultation Details
- **AI Tool Used:** Claude 3.5 Sonnet
- **Prompt Used:** PL-SYS-003 (Caching Strategy variation)
- **AI's Proposed Solution Summary:** AI proposed creating an `after insert` SQL trigger on the `Transactions` table to immediately update a `current_stock` column in the `Inventory` table.

### 3. Considered Alternatives
- **Option A: AI's Suggestion (DB Triggers)**
  - Pros: Data is always 100% accurate instantly; no application logic needed.
  - Cons: Hidden side effects, potential lock contention on the `Inventory` table during high transaction volumes, harder to debug.
- **Option B: Application-level calculation**
  - Pros: No DB magic, clear logic in the code.
  - Cons: Requires large data pulls, very slow for the dashboard.
- **Option C: Materialized Views with async refresh**
  - Pros: Highly performant read access, separates read logic from write transactions.
  - Cons: Data might be slightly stale (e.g., up to 5 minutes old).

### 4. Decision Outcome
- **Chosen approach:** Option C (Materialized Views)
- **What was accepted from AI:** The database schema layout (tables and relationships).
- **What was modified:** Replaced triggers with materialized views.
- **What was rejected:** The trigger-based synchronization approach entirely.

### 5. Rationale
- **Technical justification:** Avoids database write locks and concurrency bottlenecks.
- **Security considerations:** None.
- **Performance considerations:** Vastly improves dashboard load times at the cost of slight data staleness.
- **Maintainability considerations:** Easier to version control and test than database triggers.

### 6. Consequences
- **Positive consequences:** System can handle high write volume without locking the read tables.
- **Negative consequences / risks:** The UI must be designed to inform users that stock levels might be delayed by a few minutes.
- **Follow-up actions needed:** Create a CRON job to refresh the materialized view every 5 minutes.

### 7. Verification Plan
- **How the decision will be verified:** Load testing.
- **Success criteria:** Dashboard loads in < 500ms even with 100 concurrent write transactions happening.
- **Test cases:** TC-PERF-015

### 8. Traceability
- **Related requirement(s):** REQ-PERF-01
- **Related commit(s):** `db45a1f`
- **Related test case(s):** TC-PERF-015

---

## Example 2: Authentication Implementation Decision

### ADR ID: ADR-2023-11-15-002
**Title:** Adopt JWT for stateless API authentication
**Date:** 2023-11-15
**Status:** Accepted
**Decision Maker(s):** Alice Wang
**SDLC Phase:** Implementation
**Related AI Usage Log Entry:** AIL-2023-11-14-005

### 1. Context and Problem Statement
Our backend REST API needs a secure way to authenticate mobile and web clients without storing session state in memory on the server, to allow horizontal scaling.

### 2. AI Consultation Details
- **AI Tool Used:** ChatGPT-4o
- **Prompt Used:** PL-SYS-002 (Trade-offs)
- **AI's Proposed Solution Summary:** Suggested JSON Web Tokens (JWT) stored in HTTP-only cookies for web, and secure storage for mobile. Provided sample middleware implementation.

### 3. Considered Alternatives
- **Option A: AI's Suggestion (JWT)**
  - Pros: Stateless, scales easily, standard libraries available.
  - Cons: Token invalidation (logout) is complex before expiry.
- **Option B: Redis-backed Sessions (Session IDs)**
  - Pros: Easy to revoke, standard paradigm.
  - Cons: Requires maintaining a Redis cluster, adds network hop for every authenticated request.

### 4. Decision Outcome
- **Chosen approach:** Option A (JWT) with slight modification.
- **What was accepted from AI:** The overall JWT architecture, the short-lived access token pattern.
- **What was modified:** Added a long-lived refresh token stored in the database to handle the token revocation issue.
- **What was rejected:** AI's suggestion to use local storage for web clients (changed to HTTP-only cookies for XSS protection).

### 5. Rationale
- **Technical justification:** Best balance of scalability and security for our dual-client (web/mobile) needs.
- **Security considerations:** Refresh tokens allow us to revoke access if a device is compromised, while short-lived JWTs limit the window of vulnerability.
- **Performance considerations:** Backend doesn't need to hit the DB/Redis for every request to validate sessions.
- **Maintainability considerations:** Standard industry practice, well-documented libraries.

### 6. Consequences
- **Positive consequences:** Backend is stateless and easily deployable behind a load balancer.
- **Negative consequences / risks:** Implementation of the refresh token flow is complex and requires strict testing.
- **Follow-up actions needed:** Implement strict CORS policies since we are using HTTP-only cookies.

### 7. Verification Plan
- **How the decision will be verified:** Security audit and penetration testing of the auth endpoints.
- **Success criteria:** Cannot access protected routes without valid token; token expires correctly; refresh token issues new access token.
- **Test cases:** TC-SEC-001 to TC-SEC-010

### 8. Traceability
- **Related requirement(s):** REQ-SEC-02
- **Related commit(s):** `8f2a9bc`
- **Related test case(s):** TC-SEC-001
