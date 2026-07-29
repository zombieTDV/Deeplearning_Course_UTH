# AI Reference (Project Standards & Conventions)

This reference document defines the strict coding conventions and folder structure standards that all AI assistants and developers must follow 100% when generating or reviewing code.

---

## 1. Naming Conventions

| Scope | Style | Rule | Example |
|---|---|---|---|
| **Files & Modules** | `snake_case` | Lowercase with underscores | `data_loader.py`, `train_pipeline.py` |
| **Classes** | `PascalCase` | Capitalize each word | `ResNetBackbone`, `Trainer`, `Evaluator` |
| **Functions & Methods** | `snake_case` / `camelCase` | Lowercase or camelCase | `compute_loss()`, `computeLoss()` |
| **Variables** | `snake_case` / `camelCase` | Descriptive lowercase or camelCase | `learning_rate`, `learningRate` |
| **Constants** | `UPPER_SNAKE_CASE` | All caps with underscores | `BATCH_SIZE`, `NUM_CLASSES`, `DEVICE` |
| **Config Files** | `snake_case` | YAML/JSON format | `default_config.yaml`, `model_config.json` |

---

## 2. Folder Structure Standard

AI must place generated code and files in the designated directory layers:

```text
project/
├── data/           # Raw & Processed datasets
│   ├── raw/        # Original immutable datasets
│   └── processed/  # Preprocessed & normalized data
├── models/         # Model architecture definitions (nn.Module classes)
├── training/       # Training loops, loss functions, optimizers, checkpointing
├── evaluation/     # Metrics calculation, plot generation, report export
├── configs/        # Configuration files (YAML / JSON)
├── agents/         # AI Knowledge Store, prompt specs, workflow rules
├── tests/          # Smoke tests (< 10s) and Unit tests
└── scripts/        # Entrypoint scripts (e.g., train.py, evaluate.py)
```

---

## 3. Code Generation & Context Window Rules for LLMs

1. **Context Window Loading**: Before generating code, load the relevant prompt spec (`agents/prompts/<pipeline>/`), `agents/AI_REFERENCE.md`, and `agents/knowledge_store/` into the active LLM context window.
2. **No Monolithic Scripts**: Always separate logic across `models/`, `training/`, and `evaluation/`.
3. **Type Annotations**: Provide Python type hints for all function arguments and return values.
4. **Explicit Shape Annotations**: Comment tensor shapes on every transformation (e.g., `# shape: (B, C, H, W)`).
5. **Reproducibility**: Set random seeds (`torch.manual_seed`, `np.random.seed`) at script startup.
