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


def build_resnet18_cifar_stem(
    num_classes: int = 10,
    mode: str = "finetune",
    device: torch.device | None = None,
) -> nn.Module:
    """Build ResNet18 adapted with 3x3 Conv1 stem for native 32x32 CIFAR-10 images."""
    weights = ResNet18_Weights.DEFAULT if mode in ["frozen", "finetune"] else None
    model = resnet18(weights=weights)

    # 1. Replace 7x7 stride=2 conv1 with 3x3 stride=1 conv1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 2. Bypass MaxPool to preserve 32x32 resolution
    model.maxpool = nn.Identity()

    # 3. Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 4. Freeze/Unfreeze configuration
    if mode == "frozen":
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.conv1, True)
        set_parameter_requires_grad(model.fc, True)
    elif mode == "finetune":
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.conv1, True)
        set_parameter_requires_grad(model.layer4, True)
        set_parameter_requires_grad(model.fc, True)

    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def build_convnext_tiny(
    num_classes: int = 10,
    mode: str = "finetune",
    device: torch.device | None = None,
) -> nn.Module:
    """Load ConvNeXt-Tiny and adapt for *num_classes*."""
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    if mode == "frozen":
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.classifier, True)
    elif mode == "finetune":
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.features[7], True)  # Last stage
        set_parameter_requires_grad(model.classifier, True)

    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def build_efficientnet_b0(
    num_classes: int = 10,
    mode: str = "finetune",
    device: torch.device | None = None,
) -> nn.Module:
    """Load EfficientNet-B0 and adapt for *num_classes*."""
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if mode == "frozen":
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.classifier, True)
    elif mode == "finetune":
        set_parameter_requires_grad(model, False)
        set_parameter_requires_grad(model.features[8], True)  # Last stage
        set_parameter_requires_grad(model.classifier, True)

    model.eval()
    if device is not None:
        model = model.to(device)
    return model

