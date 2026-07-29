# Tests Directory (Data Science Testing)

This directory contains the automated test scripts for the project, implementing the **Two Testing Modes** specified in Section 9 of [`agents/WORKFLOW.md`](file:///home/bush/Desktop/Deeplearning_Course_UTH/agents/WORKFLOW.md):

## Testing Structure
- `smoke_test.py`: **Smoke Test (< 10s)** — Instant sanity check for package imports, CUDA availability, and tensor shape alignment on dummy data.
- `normal_test.py`: **Normal Test (Full Check)** — Complete verification suite for model forward/backward pass, loss computation, and directory integrity.
- `unit/`: Thư mục chứa Unit Tests cho custom DataLoaders, model layers, và loss functions.

## Commands
```bash
# Fast Smoke Test (< 10s)
python3 tests/smoke_test.py

# Full Normal Test
python3 tests/normal_test.py
```
