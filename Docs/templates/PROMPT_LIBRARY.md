# Prompt Library

**Purpose:** This document catalogs effective, reusable prompts for the team to use when interacting with AI tools. Sharing successful prompts improves team efficiency, consistency, and output quality across the Software Development Life Cycle (SDLC).

## Categories
1. Requirements Analysis Prompts
2. System Design Prompts
3. Database Design Prompts
4. API Design Prompts
5. Code Generation Prompts
6. Code Review Prompts
7. Testing Prompts
8. Documentation Prompts
9. Debugging Prompts
10. Security Analysis Prompts
11. Refactoring Prompts

---

## 1. Requirements Analysis Prompts

### Prompt ID: PL-REQ-001
- **Category:** Requirements Analysis
- **Purpose:** Break down a high-level epic into actionable user stories.
- **Template:** "Break down the following epic into standard user stories (As a [role], I want to [action], so that [benefit]). For each user story, provide 3-5 clear acceptance criteria. Epic: {{EPIC_DESCRIPTION}}. Constraints: {{CONSTRAINTS}}"
- **Usage Notes:** Best used with large feature requests.
- **Quality Criteria:** User stories must be independent, negotiable, valuable, estimable, small, and testable (INVEST).
- **Example Input:** Epic: "User profile management." Constraints: "Must be GDPR compliant."
- **Example Output Summary:** 4 user stories generated (Update info, Delete account, Export data, Change password) with acceptance criteria including GDPR deletion timelines.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

### Prompt ID: PL-REQ-002
- **Category:** Requirements Analysis
- **Purpose:** Identify edge cases in a proposed feature.
- **Template:** "Analyze the following feature requirement and list potential edge cases, negative scenarios, and error states that need to be handled. Feature: {{FEATURE_DESCRIPTION}}"
- **Usage Notes:** Crucial for hardening requirements before design begins.
- **Quality Criteria:** Should identify network failures, invalid data inputs, concurrent actions, and boundary values.
- **Example Input:** Feature: "Checkout process for e-commerce cart."
- **Example Output Summary:** Identified edge cases: Item goes out of stock during checkout, payment gateway timeout, applying expired discount code, concurrent checkouts on same account.
- **Version History:** 1.0
- **Effectiveness Rating:** 5/5

### Prompt ID: PL-REQ-003
- **Category:** Requirements Analysis
- **Purpose:** Draft Non-Functional Requirements (NFRs).
- **Template:** "Based on this system description: {{SYSTEM_DESC}}, generate a list of Non-Functional Requirements (NFRs) categorized by Performance, Security, Reliability, and Usability. Quantify the requirements where possible."
- **Usage Notes:** Helps ensure system quality attributes are not ignored.
- **Quality Criteria:** NFRs must be measurable and testable.
- **Example Input:** System: "Real-time chat application for university students."
- **Example Output Summary:** Latency < 200ms, Uptime 99.9%, Passwords hashed with bcrypt, Support for 500 concurrent users.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

---

## 2. System Design Prompts

### Prompt ID: PL-SYS-001
- **Category:** System Design
- **Purpose:** Generate a high-level architecture proposal.
- **Template:** "Propose a high-level architecture for a {{SYSTEM_TYPE}} using {{TECH_STACK}}. The system must handle {{SCALE_REQUIREMENTS}}. Describe the key components, data flow, and draw a Mermaid.js diagram representing the architecture."
- **Usage Notes:** Good starting point for architectural discussions.
- **Quality Criteria:** Diagram should compile. Components should be loosely coupled.
- **Example Input:** System: "Ride-sharing backend", Tech Stack: "Node.js, Redis, PostgreSQL", Scale: "10,000 active drivers".
- **Example Output Summary:** Proposed microservices architecture with a pub/sub mechanism in Redis for location tracking. Included a working Mermaid diagram.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

### Prompt ID: PL-SYS-002
- **Category:** System Design
- **Purpose:** Evaluate architectural trade-offs.
- **Template:** "Compare {{OPTION_A}} vs {{OPTION_B}} for implementing {{FEATURE_REQUIREMENT}}. Analyze them based on complexity, scalability, cost, and maintainability. Recommend one option for a team of {{TEAM_SIZE}} students with {{EXPERIENCE_LEVEL}} experience."
- **Usage Notes:** Useful when deciding between major technologies or patterns.
- **Quality Criteria:** Objective comparison with a tailored recommendation.
- **Example Input:** Option A: "REST API", Option B: "GraphQL", Feature: "Mobile app data fetching", Team: "5", Experience: "Beginner".
- **Example Output Summary:** Recommended REST due to lower learning curve and existing team knowledge, acknowledging GraphQL's data-fetching efficiency as a trade-off.
- **Version History:** 1.0
- **Effectiveness Rating:** 5/5

### Prompt ID: PL-SYS-003
- **Category:** System Design
- **Purpose:** Design a caching strategy.
- **Template:** "Design a caching strategy for {{DATA_TYPE}} in our {{SYSTEM_TYPE}}. Suggest the caching layer (e.g., Redis, Memcached), cache invalidation policies (TTL, event-driven), and potential pitfalls (e.g., stale data, cache stampede)."
- **Usage Notes:** Use when database load becomes a concern.
- **Quality Criteria:** Must address cache invalidation clearly.
- **Example Input:** Data Type: "User profiles and settings", System Type: "Social media dashboard".
- **Example Output Summary:** Suggested Redis with Write-Through caching and a 1-hour TTL as a fallback to prevent stale data.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

---

## 3. Database Design Prompts

### Prompt ID: PL-DB-001
- **Category:** Database Design
- **Purpose:** Generate initial relational schema.
- **Template:** "Create a relational database schema for a {{DOMAIN}} application. Entities include: {{ENTITIES}}. Provide the SQL DDL statements (PostgreSQL dialect) with appropriate data types, primary keys, and foreign keys. Ensure it is in 3rd Normal Form."
- **Usage Notes:** Review outputs carefully for correct normal forms.
- **Quality Criteria:** Syntactically correct DDL, proper constraints, no redundant data.
- **Example Input:** Domain: "Library Management", Entities: "Books, Authors, Members, Loans".
- **Example Output Summary:** Generated 4 tables with a junction table for Book-Authors if many-to-many was assumed. Added foreign key constraints for Loans.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

### Prompt ID: PL-DB-002
- **Category:** Database Design
- **Purpose:** Identify necessary database indexes.
- **Template:** "Given the following table schema: {{SCHEMA_DDL}} and the primary query patterns: {{QUERY_PATTERNS}}, recommend which columns or combinations of columns should be indexed. Explain why."
- **Usage Notes:** Optimize slow queries.
- **Quality Criteria:** Avoids over-indexing. Suggests composite indexes where appropriate.
- **Example Input:** Schema: "Users(id, email, status, created_at)", Queries: "Find active users ordered by creation date".
- **Example Output Summary:** Suggested a composite index on `(status, created_at)` for efficient filtering and sorting.
- **Version History:** 1.0
- **Effectiveness Rating:** 5/5

### Prompt ID: PL-DB-003
- **Category:** Database Design
- **Purpose:** Design NoSQL document structure.
- **Template:** "Design a MongoDB document schema for {{DATA_REQUIREMENT}}. Should we embed the {{SUB_ENTITY}} data or reference it? Explain the trade-offs based on a read-heavy vs write-heavy workload."
- **Usage Notes:** For deciding between denormalization and referencing in NoSQL.
- **Quality Criteria:** Considers document size limits and atomicity.
- **Example Input:** Data Requirement: "Blog posts and comments", Sub Entity: "Comments".
- **Example Output Summary:** Recommended embedding for small comment counts, but referencing for unbounded growth to avoid the 16MB document limit.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

---

## 4. API Design Prompts

### Prompt ID: PL-API-001
- **Category:** API Design
- **Purpose:** Generate OpenAPI/Swagger specification.
- **Template:** "Write an OpenAPI 3.0 YAML specification for a REST API managing {{RESOURCE_NAME}}. Include GET (list), POST (create), GET (by ID), PUT (update), and DELETE (remove) endpoints. Include basic schemas for the request and response bodies, and standard error responses (400, 404, 500)."
- **Usage Notes:** Great for contract-first development.
- **Quality Criteria:** Valid YAML, adheres to RESTful conventions.
- **Example Input:** Resource Name: "Products".
- **Example Output Summary:** Provided a complete, valid OpenAPI spec with schemas for `Product`, `ProductInput`, and `Error`.
- **Version History:** 1.0
- **Effectiveness Rating:** 5/5

### Prompt ID: PL-API-002
- **Category:** API Design
- **Purpose:** Design API pagination and filtering.
- **Template:** "Design the query parameters for filtering, sorting, and cursor-based pagination for a REST API endpoint: `GET /{{RESOURCE_COLLECTION}}`. Provide examples of the URL and the JSON response structure."
- **Usage Notes:** Cursor pagination is preferred for large datasets.
- **Quality Criteria:** Response includes `next_cursor` and avoids offset/limit performance issues.
- **Example Input:** Resource: "transactions".
- **Example Output Summary:** Generated URL `GET /transactions?status=completed&sort=-date&cursor=xyz123`, and a response body including `data` array and `meta` object.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

### Prompt ID: PL-API-003
- **Category:** API Design
- **Purpose:** Standardize error payload responses.
- **Template:** "Propose a standard JSON error response format for our API. It needs to handle validation errors (multiple fields) as well as general server errors. Provide example payloads for a 400 Bad Request (validation failure) and a 500 Internal Server Error."
- **Usage Notes:** Use to establish team-wide API consistency early on.
- **Quality Criteria:** Adheres to RFC 7807 (Problem Details for HTTP APIs) or similar structured format.
- **Example Input:** (None required, purely standard generation).
- **Example Output Summary:** Suggested a standard wrapper with `code`, `message`, and a `details` array mapping field names to specific validation messages.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

---

## 5. Code Generation Prompts

### Prompt ID: PL-CODE-001
- **Category:** Code Generation
- **Purpose:** Scaffold a standard component/module.
- **Template:** "Write a {{LANGUAGE}}/{{FRAMEWORK}} component that implements {{FUNCTIONALITY}}. Strictly follow these constraints: {{CONSTRAINTS}}. Do not use external libraries unless specified. Include inline comments explaining complex logic."
- **Usage Notes:** Always specify strict constraints to avoid hallucinated dependencies.
- **Quality Criteria:** Code compiles, follows idiomatic language patterns, adheres to constraints.
- **Example Input:** Language: "React/TypeScript", Functionality: "A toggle switch button", Constraints: "Use Tailwind CSS, implement accessibility (aria-labels), accept 'checked' and 'onChange' props."
- **Example Output Summary:** Clean React component using functional syntax, Tailwind classes, and ARIA attributes for screen readers.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5

### Prompt ID: PL-CODE-002
- **Category:** Code Generation
- **Purpose:** Write an algorithm or data processing function.
- **Template:** "Write a pure function in {{LANGUAGE}} to process {{INPUT_DATA_DESCRIPTION}} and output {{OUTPUT_DATA_DESCRIPTION}}. Ensure the time complexity is at most {{TARGET_COMPLEXITY}}. Handle edge cases like empty inputs or null values."
- **Usage Notes:** Best for isolated logic, utility functions, or data transformation.
- **Quality Criteria:** Must achieve the target Big O complexity and handle edge cases gracefully.
- **Example Input:** Language: "Python", Input: "List of dictionaries representing orders", Output: "Total revenue grouped by product ID", Target Complexity: "O(n)".
- **Example Output Summary:** Provided an efficient Python function using `collections.defaultdict` to aggregate data in a single pass O(n).
- **Version History:** 1.0
- **Effectiveness Rating:** 5/5

### Prompt ID: PL-CODE-003
- **Category:** Code Generation
- **Purpose:** Generate boilerplate DB repository/DAO class.
- **Template:** "Generate a Data Access Object (DAO) class in {{LANGUAGE}} for the {{ENTITY_NAME}} entity using {{ORM_OR_DB_DRIVER}}. Include basic CRUD methods. Use dependency injection for the database connection and handle database exceptions gracefully."
- **Usage Notes:** Adjust to your specific ORM/Driver.
- **Quality Criteria:** Follows DI patterns, avoids SQL injection (uses parameterized queries or ORM methods safely).
- **Example Input:** Language: "Java/Spring Boot", Entity: "Customer", ORM: "Spring Data JPA".
- **Example Output Summary:** Generated a standard Spring Data Repository interface and a service layer wrapper handling `DataAccessException`.
- **Version History:** 1.0
- **Effectiveness Rating:** 4/5
