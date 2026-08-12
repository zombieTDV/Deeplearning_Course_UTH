# PYTORCH_FRAMEWORK_RULES.md — PyTorch Framework Usage & Architecture Rules

- **Motivation/Background**: This project uses PyTorch and the Hugging Face ecosystem as its core deep learning framework. All model building, data pipeline transforms, training loops, and evaluations must adhere to standard PyTorch patterns.
- **Purpose**: Define mandatory PyTorch architectural rules, device management, reproducibility standards, and Hugging Face integration guidelines.
- **Overview Pipeline**: Applied across `src/data/`, `src/models/`, `src/training/`, and `src/eval/`.

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

5. **Hugging Face & PyTorch Integration**:
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
- [ ] Gradient computation explicitly disabled (`with torch.no_grad():` or `@torch.no_grad()`) during evaluation and inference.
