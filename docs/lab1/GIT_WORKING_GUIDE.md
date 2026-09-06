# Git Working Guide — AI Agent Collaboration

## Principles

- **One commit per logical unit of work.** Don't bundle unrelated changes.
- **Commit messages explain *why* and *what*, not *how*.** The diff shows *how*.
- **Review before committing.** Always check `git status`, `git diff`, `git log --oneline -5` first.
- **Never commit secrets.** API keys, tokens, passwords, `.env` files.
- **Never force-push to shared branches.** Rebase only on your own feature branches.

## Branching

```
main
├── LAB1-FashionMNIST-Classification/main
├── LAB1-FashionMNIST-Classification/member1-data-eda
├── LAB1-FashionMNIST-Classification/member2-baseline
└── ...
```

- Parent branch (`LAB1-X/main`) holds shared scaffolding.
- Sub-branches are short-lived per-member feature branches.
- Sub-branches merge **into** the parent branch, never directly into `main`.
- Sub-branches are **preserved after merging** for audit trail and future reference.

## Workflow

1. Create/switch to the appropriate branch first.
2. Stage only intended files: `git add <file>` (never `git add .` blindly).
3. Verify staged content with `git status` and `git diff --staged`.
4. Write a structured commit message:

```
<subject line — 50 chars max, imperative mood>

<blank line>

<body — wrap at 72 chars. Explain what and why, one bullet or paragraph per change.>
```

5. Push: `git push` (set upstream on first push with `-u`).

## Commit Message Examples

```
Add data loader for FashionMNIST with train/test split

- Loads FashionMNIST via torchvision.datasets
- Returns (X_train, y_train, X_test, y_test) as numpy arrays
- Normalizes pixel values to [0, 1]
```

```
Refactor training loop into separate Trainer class

Moves training logic out of the notebook into src/trainer.py
for reuse across experiments. No behavioral change.
```

## What NOT to do

| Pitfall                                                 | Why                                                |
| ------------------------------------------------------- | -------------------------------------------------- |
| `git add .` or `git add -A` without review          | Staging junk (cache files, temp files, large data) |
| Vague messages like "fix stuff", "update"               | Impossible to understand later                     |
| Committing commented-out code                           | Dead code should be deleted, not commented         |
| Mixing formatting/whitespace changes with logic changes | Review is harder                                   |
| Committing large binary/data files to git               | Bloats the repo; use`.gitignore` or DVC          |
| Amending pushed commits                                 | Rewrites history others may have based work on     |
| Using`git commit --no-verify` routinely               | Skips hooks that catch issues                      |

## Per-Project Rules (this repo)

- `data/` — never commit raw datasets. Use `.gitignore`.
- `model/backup/`, `model/baseline/` — never commit `.pt`, `.pth`, `.onnx` files.
- `labs/*/results/` — generated plots, logs, metrics are local only.
- `scripts/` — name with date prefix: `YYYY-MM-DD_desc.py`.
- `src/` — reusable modules (not experiment scripts).
