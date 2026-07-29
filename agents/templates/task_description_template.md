# Task Description Template

Use this Markdown template when starting a new task before AI generates code (Section 3 of [`agents/WORKFLOW.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/WORKFLOW.md)).

```markdown
## Task Specification: <Task Name>

### 1. Goal (Mục tiêu)
<What feature, model layer, or script needs to be implemented>

### 2. Plan (Kế hoạch)
- [ ] Step 1: Define interface / input shapes
- [ ] Step 2: Implement core algorithm / model module
- [ ] Step 3: Run Smoke Test verification

### 3. Target Pipeline (Phân loại Pipeline)
- [ ] Data Preparation
- [ ] Model Architecture
- [ ] Training
- [ ] Evaluation
- [ ] Deployment

### 4. Detailed Execution Plan (Các bước chi tiết)
1. Input: <expected tensor shape or data format>
2. Logic: <processing step 1, step 2>
3. Output: <saved artifact or returned value>

### 5. Definition of Done (Tiêu chuẩn hoàn thành)
- [ ] Code runs without syntax errors
- [ ] Passes Smoke Test (`python3 tests/smoke_test.py`)
- [ ] Adheres to `snake_case` / `PascalCase` / `camelCase` naming standards
- [ ] Placed in correct pipeline folder (`models/`, `training/`, etc.)
```
