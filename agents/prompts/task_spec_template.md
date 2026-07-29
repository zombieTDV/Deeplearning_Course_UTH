# Task Prompt Specification Template

Use this Markdown template for creating AI prompt specifications as mandated by Section 2 of [`agents/WORKFLOW.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/WORKFLOW.md).

```markdown
Name: <Task / Module Name>
Description: <Brief summary of what needs to be built>
Purpose: <Target outcome and goal>
Input: <Input data types, shapes, existing code context>
Output: <Expected artifact, class, function, or module output>

How to do:
1. Step 1: Initialize module in correct directory.
2. Step 2: Implement core logic adhering to naming conventions (snake_case / PascalCase).
3. Step 3: Add input validation and error handling.
4. Step 4: Create unit test / smoke test case.

Examples:
- Input Example: batch of tensor shape (B, C, H, W)
- Output Example: loss scalar and logits tensor (B, num_classes)
```
