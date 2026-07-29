# Behavioral Layer Prompt Specification Template

Use this Markdown template to define prompts for AI code generation (Section 2 & 5 of [`agents/WORKFLOW.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/WORKFLOW.md)).

```markdown
Name: <Pipeline Task Name>
Description: <Brief description of what this module does>
Purpose: <Goal & target outcome>
Input: <Data structure / Tensor shape / Config input>
Output: <Output data / Output tensor shape / Saved artifact>

How to do:
1. Load & validate inputs
2. Implement core module following snake_case / PascalCase rules
3. Add inline comments with tensor shapes
4. Return output artifact & verify via Smoke Test

Examples:
Input:  batch tensor of shape (64, 1, 28, 28)
Output: logits tensor of shape (64, 10)
```
