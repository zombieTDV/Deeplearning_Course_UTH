# src/models — Model Layer

## Architecture Overview (LAB2)

Builders produce the pretrained CIFAR-10 classifiers consumed by the training
scripts (`src/training/train_lab2_models.py`) and analysis notebooks:

| Builder | Trainable layers |
|---|---|
| `build_resnet18(mode="frozen")` | fc only (feature extraction) |
| `build_resnet18(mode="finetune")` | layer4 + fc |
| `build_resnet18_full_sota()` | layer3 + layer4 + fc (EXP-07 SOTA) |
| `build_densenet121(mode="frozen")` | classifier only |
| `build_densenet121(mode="finetune")` | denseblock4 + norm5 + classifier |
| `build_densenet121_full_sota()` | denseblock3 + denseblock4 + norm5 + classifier |
| `build_resnet18_cifar_stem()` | 3x3 stem variant for native 32x32 input |
| `build_convnext_tiny()` / `build_efficientnet_b0()` | modern-arch sweeps (EXP-05/06) |

SOTA variants use **Layer-wise Discriminative Learning Rates** (LLRD) with the
param groups from `get_resnet18_lrd_param_groups()` /
`get_densenet121_lrd_param_groups()` (head > tail > stem).

Helpers: `set_parameter_requires_grad()`, `count_trainable_params()`,
`count_all_params()`.

## Usage

```python
from src.models.build_model import build_resnet18_full_sota, get_resnet18_lrd_param_groups
model = build_resnet18_full_sota(num_classes=10, device=device)
groups = get_resnet18_lrd_param_groups(model, base_lr=3e-4)
```

See [agents/phases/MODEL.md](../../agents/phases/MODEL.md).
