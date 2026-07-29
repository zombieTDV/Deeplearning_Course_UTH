# Codebase Audit Checklist

Use this checklist to review AI-generated code before running full tests or committing (Section 7 of [`agents/WORKFLOW.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/WORKFLOW.md)).

```markdown
## Codebase Audit Checklist

### 1. Architecture Check (Kiểm tra Kiến trúc)
- [ ] Code is placed in the correct directory layer (`data/`, `models/`, `training/`, `evaluation/`, `scripts/`)
- [ ] No monolithic code: Logic is cleanly separated across files
- [ ] Dependencies are imported correctly without hallucinated packages

### 2. Detailed Plan Alignment (Kiểm tra Chi tiết Kế hoạch)
- [ ] All steps defined in the Task Execution Plan have been fulfilled
- [ ] Tensor input and output shapes match the specification
- [ ] Error handling and edge case validation are implemented

### 3. Coding Conventions & Style Audit (Kiểm tra Chuẩn Code)
- [ ] Files & Functions use `snake_case`
- [ ] Classes use `PascalCase`
- [ ] Constants use `UPPER_SNAKE_CASE`
- [ ] Tensor shape comments are present on complex transformations
- [ ] Random seeds are explicitly configured for reproducibility
```
