# PYTORCH_FRAMEWORK_RULES.md — PyTorch Architecture & Device Governance

- **Motivation/Background**: This project uses PyTorch and the Hugging Face ecosystem as its core deep learning framework. All model building, data pipeline transforms, training loops, and evaluations must adhere to standard PyTorch patterns.
- **Purpose**: Define mandatory PyTorch architectural rules, device management, reproducibility standards, and Hugging Face integration guidelines.
- **Overview Pipeline**: Enforced across all model definitions (`src/*/models/`), dataloaders (`src/*/data/`), training loops (`src/*/training/`), and evaluations (`src/*/eval/`).
- **Detailed Plan**: §1 PyTorch Core Principles; §2 Device Agnostic Execution & VRAM Targets; §3 Determinism & Seeding; §4 Hugging Face & Cleanlab Rules; §5 Safe Checkpoint Loading.
- **References**: `torch`, `torchvision`, `transformers`, `agents/rules/LOGGING_CHECKPOINT_RULES.md`.
- **Created**: 2026-09-06T13:05:18+07:00
- **Last Updated**: 2026-09-06T21:25:00+07:00

---


## 1. PyTorch Core Principles

1. **Framework Baseline**:
   - PyTorch (`torch`) is the primary backend framework.
   - All custom dataset objects must inherit from `torch.utils.data.Dataset`.
   - Data loading must utilize `torch.utils.data.DataLoader` (or Hugging Face `DataLoader` / `DataCollator` wrappers).

2. **Device Agnostic Execution (CPU/GPU)**:
   - Always specify device dynamically:
     ```python
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     ```
   - Never hardcode `"cuda:0"` or assume CUDA is available.
   - For VRAM optimization across team hardware (1 team member has 8GB VRAM, 1 team member has 4GB VRAM):
     - **Strict VRAM Target:** Target ≤3.5GB VRAM usage (ceiling 4GB) so the codebase runs smoothly on all team machines.
     - **VRAM Optimizations:** Use mixed precision (`torch.cuda.amp.autocast()` or `fp16=True`), small per-device batch sizes (e.g. 8 or 16) with gradient accumulation steps (e.g. 2 or 4), and `gradient_checkpointing=True` if needed.
     - Enable `pin_memory=True` in `DataLoader` when CUDA is available.

3. **Reproducibility & Seeding**:
   - All training entry points must set seeds explicitly:
     ```python
     def set_seed(seed: int = 42):
         torch.manual_seed(seed)
         torch.cuda.manual_seed_all(seed)
         np.random.seed(seed)
         random.seed(seed)
         torch.backends.cudnn.deterministic = True
     ```

4. **Model Building & Serialization**:
   - Custom neural network components must inherit from `torch.nn.Module`.
   - Forward passes must accept tensors and return PyTorch/HuggingFace model outputs.
   - Checkpoints **must** be loaded with `weights_only=True` for security:
     ```python
     checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
     ```

5. **Training Monitoring & TensorBoard**:
   - All training loops and finetuning scripts MUST integrate PyTorch TensorBoard (`torch.utils.tensorboard.SummaryWriter` or `--tb` flag in HF `TrainingArguments`).
   - TensorBoard logs must be stored in `experiments/runs/<ts>_<run_name>/tensorboard/`.
   - Log key metrics per step/epoch: training loss, validation loss, accuracy, learning rate, and VRAM allocation.

6. **Hugging Face & PyTorch Integration**:
   - For Hugging Face models (`AutoModelForSequenceClassification`), output tensors/logits must be handled via standard PyTorch loss functions or the HF `Trainer`.
   - When converting HF `Dataset` objects for PyTorch:
     ```python
     dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
     ```

---

## 2. Checklist for PyTorch Compliance

- [ ] Device set dynamically with `torch.device`.
- [ ] Seeds initialized using `set_seed()`.
- [ ] PyTorch tensors formatted correctly before passing into model `forward()`.
- [ ] `weights_only=True` enforced on all `torch.load` calls.
- [ ] TensorBoard summary writer integrated and outputs directed to `experiments/runs/<ts>_<run>/tensorboard/`.
- [ ] Gradient computation explicitly disabled (`with torch.no_grad():` or `@torch.no_grad()`) during evaluation and inference.
