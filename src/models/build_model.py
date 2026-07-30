"""
build_model.py — Load pretrained ResNet18 / DenseNet121, replace final
classification layer for CIFAR-10 (10 classes), configure freeze/unfreeze.

Usage:
    from src.models.build_model import build_resnet18, build_densenet121

    model = build_resnet18(num_classes=10, mode="frozen")
    model = build_resnet18(num_classes=10, mode="finetune")
    model = build_densenet121(num_classes=10, mode="frozen")
    model = build_densenet121(num_classes=10, mode="finetune")

Modes:
    "frozen"   — only the final classification layer is trainable
                 (feature extraction / transfer learning).
    "finetune" — the last residual/dense block + final layer are
                 trainable; everything else is frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    densenet121,
    resnet18,
)

# ---------------------------------------------------------------------------
# Supported architectures and their final-layer attributes
# ---------------------------------------------------------------------------
_ARCH_FC_ATTR = {
    "resnet18": "fc",
    "densenet121": "classifier",
}

_ARCH_IN_FEATURES = {
    "resnet18": 512,
    "densenet121": 1024,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_parameter_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Freeze (requires_grad=False) or thaw all parameters in *model*."""
    for param in model.parameters():
        param.requires_grad = requires_grad


def count_trainable_params(model: nn.Module) -> int:
    """Return number of parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    """Return total number of parameters."""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_resnet18(
    num_classes: int = 10,
    mode: str = "frozen",
    device: torch.device | None = None,
) -> nn.Module:
    """Load ImageNet-pretrained ResNet18 and adapt for *num_classes*.

    Args:
        num_classes: Number of output classes (default 10 for CIFAR-10).
        mode: ``"frozen"`` (feature extraction, default) or ``"finetune"``
              (last residual block + FC trainable).
        device: Target device (e.g. ``torch.device("cuda")``).  If ``None``
                (default), the model stays on CPU.

    Returns:
        Model in ``eval()`` mode on the requested device with the freeze
        configuration applied.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)

    if mode == "frozen":
        # Freeze everything except the new FC layer
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.fc, True)

    elif mode == "finetune":
        # Freeze everything first, then unfreeze layer4 + fc
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.layer4, True)
        set_parameter_requires_grad(model.fc, True)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'frozen' or 'finetune'.")

    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def build_densenet121(
    num_classes: int = 10,
    mode: str = "frozen",
    device: torch.device | None = None,
) -> nn.Module:
    """Load ImageNet-pretrained DenseNet121 and adapt for *num_classes*.

    Args:
        num_classes: Number of output classes (default 10 for CIFAR-10).
        mode: ``"frozen"`` (feature extraction, default) or ``"finetune"``
              (last dense block + classifier trainable).
        device: Target device (e.g. ``torch.device("cuda")``).  If ``None``
                (default), the model stays on CPU.

    Returns:
        Model in ``eval()`` mode on the requested device with the freeze
        configuration applied.
    """
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Linear(in_features, num_classes)

    if mode == "frozen":
        # Freeze everything except the new classifier
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.classifier, True)

    elif mode == "finetune":
        # Freeze everything first, then unfreeze denseblock4 + norm5 + classifier
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.features.denseblock4, True)
        set_parameter_requires_grad(model.features.norm5, True)
        set_parameter_requires_grad(model.classifier, True)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'frozen' or 'finetune'.")

    model.eval()
    if device is not None:
        model = model.to(device)
    return model
