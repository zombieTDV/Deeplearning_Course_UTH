"""
evaluate_model.py — Test-set evaluation, confusion matrices, per-class metrics.

Usage:
    from src.eval.evaluate_model import evaluate, per_class_accuracy
    from src.eval.evaluate_model import load_checkpoint, format_comparison_table

    loss, acc = evaluate(model, test_loader, device)
    per_class, cm = per_class_accuracy(model, test_loader, device)
    model = load_checkpoint(model, "experiments/checkpoints/run_best.pt", device)
    print(format_comparison_table(results))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as sk_cm
import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute average loss and top-1 accuracy on the test set.

    Returns:
        (average_loss, top1_accuracy_pct)
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.inference_mode()
def per_class_accuracy(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[list[float], np.ndarray]:
    """Compute per-class accuracy and full confusion matrix.

    Returns:
        (per_class_acc_pct_list, confusion_matrix_numpy)
    """
    model.eval()
    model.to(device)
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    cm = sk_cm(all_labels, all_preds, labels=list(range(num_classes)))
    # Per-class accuracy = diagonal / row sum
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div-by-zero
    per_class = (cm.diagonal() / row_sums.squeeze() * 100).tolist()
    return per_class, cm


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """Load trained weights into a model instance.

    The model architecture must match the saved state dict.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(
    results: dict[str, dict],
) -> str:
    """Build a formatted comparison table string.

    Each result dict expects keys: ``test_loss``, ``test_acc``, ``per_class_acc``,
    ``mode``, ``best_val_loss``, ``best_val_acc``.
    """
    lines: list[str] = []
    lines.append(f"{'Model':25s} {'Mode':10s} {'Test Loss':12s} {'Test Acc':10s} "
                 f"{'Best Val Acc':13s} {'Class Acc Range':18s}")
    lines.append("-" * 90)

    for name, res in results.items():
        mode = res.get("mode", "?")
        tl = res.get("test_loss", float("nan"))
        ta = res.get("test_acc", float("nan"))
        va = res.get("best_val_acc", float("nan"))
        pca = res.get("per_class_acc", [])
        if pca:
            range_str = f"{min(pca):.1f}-{max(pca):.1f}%"
        else:
            range_str = "?"
        lines.append(
            f"{name:25s} {mode:10s} {tl:<12.4f} {ta:<9.2f}% "
            f"{va:<12.2f}% {range_str}"
        )

    return "\n".join(lines)
