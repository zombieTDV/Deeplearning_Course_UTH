"""
evaluate_model.py — Test-set evaluation, confusion matrices, per-class metrics.

Usage:
    from src.lab2.eval.evaluate_model import evaluate, per_class_accuracy
    from src.lab2.eval.evaluate_model import load_checkpoint, format_comparison_table

    loss, acc = evaluate(model, test_loader, device)
    per_class, cm = per_class_accuracy(model, test_loader, device)
    model = load_checkpoint(model, "experiments/checkpoints/run_best.pt", device)
    print(format_comparison_table(results))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix as sk_cm,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

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
        if pca is not None and len(pca) > 0:
            range_str = f"{min(pca):.1f}-{max(pca):.1f}%"
        else:
            range_str = "?"
        lines.append(
            f"{name:25s} {mode:10s} {tl:<12.4f} {ta:<9.2f}% "
            f"{va:<12.2f}% {range_str}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comprehensive Metrics, ASCII Table & Seaborn Heatmap Table
# ---------------------------------------------------------------------------
def compute_model_metrics(
    y_true: np.ndarray | list[int],
    probs: np.ndarray,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for a single model.

    Computes:
        - Overall Top-1 Accuracy (%)
        - Empirical Cross-Entropy Loss
        - Macro-averaged Precision, Recall, and F1 (%)
        - Weighted-averaged F1 (%)
        - One-vs-Rest (OvR) Macro & Micro ROC-AUC
        - Per-class accuracy (%)
        - Confusion matrix

    Args:
        y_true: Ground-truth integer class labels of shape (N,).
        probs: Predicted class probability distribution of shape (N, num_classes).
        num_classes: Number of target classes (default 10 for CIFAR-10).

    Returns:
        Dictionary mapping metric names to computed float/list values.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(probs, dtype=float)
    preds = np.argmax(p_arr, axis=1)

    acc = float(np.mean(preds == y_arr) * 100.0)
    true_probs = np.clip(p_arr[np.arange(len(y_arr)), y_arr], 1e-15, 1.0)
    loss = float(-np.mean(np.log(true_probs)))

    macro_prec = float(precision_score(y_arr, preds, average="macro", zero_division=0) * 100.0)
    macro_rec = float(recall_score(y_arr, preds, average="macro", zero_division=0) * 100.0)
    macro_f1 = float(f1_score(y_arr, preds, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_arr, preds, average="weighted", zero_division=0) * 100.0)

    y_bin = label_binarize(y_arr, classes=list(range(num_classes)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    try:
        macro_auc = float(roc_auc_score(y_bin, p_arr, multi_class="ovr", average="macro"))
        micro_auc = float(roc_auc_score(y_bin, p_arr, multi_class="ovr", average="micro"))
    except Exception:
        macro_auc, micro_auc = float("nan"), float("nan")

    cm = sk_cm(y_arr, preds, labels=list(range(num_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    per_class = (cm.diagonal() / row_sums.squeeze() * 100.0).tolist()

    return {
        "test_loss": loss,
        "test_acc": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "per_class_acc": per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_comprehensive_benchmark(
    models_probs_dict: dict[str, np.ndarray],
    y_true: np.ndarray | list[int],
    num_classes: int = 10,
) -> dict[str, dict[str, Any]]:
    """Compute comprehensive benchmark metrics across multiple model variants.

    Args:
        models_probs_dict: Dictionary mapping model variant names to (N, C) probability arrays.
        y_true: Ground-truth target labels.
        num_classes: Total class count.

    Returns:
        Nested dictionary mapping model names to their comprehensive metric dictionaries.
    """
    benchmark_dict: dict[str, dict[str, Any]] = {}
    for name, probs in models_probs_dict.items():
        benchmark_dict[name] = compute_model_metrics(y_true, probs, num_classes=num_classes)
    return benchmark_dict


def format_ascii_metrics_table(
    benchmark_metrics: dict[str, dict[str, Any]],
) -> str:
    """Format benchmark metrics into an aligned, publication-grade ASCII table.

    Includes:
        Model, Test Loss, Accuracy (%), Macro Precision (%), Macro Recall (%),
        Macro F1 (%), Weighted F1 (%), Macro ROC-AUC, Micro ROC-AUC.
    """
    sep = (
        "+" + "-" * 32 + "+" + "-" * 11 + "+" + "-" * 11 + "+"
        + "-" * 13 + "+" + "-" * 12 + "+" + "-" * 11 + "+"
        + "-" * 13 + "+" + "-" * 11 + "+" + "-" * 11 + "+"
    )
    header = (
        f"| {'Model':<30s} | {'Test Loss':<9s} | {'Accuracy':<9s} | "
        f"{'Macro Prec':<11s} | {'Macro Rec':<10s} | {'Macro F1':<9s} | "
        f"{'Weighted F1':<11s} | {'Macro AUC':<9s} | {'Micro AUC':<9s} |"
    )

    lines: list[str] = [sep, header, sep]
    for name, res in benchmark_metrics.items():
        tl = res.get("test_loss", float("nan"))
        ta = res.get("test_acc", float("nan"))
        mp = res.get("macro_precision", float("nan"))
        mr = res.get("macro_recall", float("nan"))
        mf1 = res.get("macro_f1", float("nan"))
        wf1 = res.get("weighted_f1", float("nan"))
        mauc = res.get("macro_auc", float("nan"))
        uauc = res.get("micro_auc", float("nan"))

        lines.append(
            f"| {name:<30s} | {tl:>9.4f} | {ta:>8.2f}% | "
            f"{mp:>10.2f}% | {mr:>9.2f}% | {mf1:>8.2f}% | "
            f"{wf1:>10.2f}% | {mauc:>9.4f} | {uauc:>9.4f} |"
        )
    lines.append(sep)
    return "\n".join(lines)


def plot_metrics_heatmap_table(
    benchmark_metrics: dict[str, dict[str, Any]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (13, 6),
    title: str = "Comprehensive Classification Benchmark Metrics (CIFAR-10 Test Set)",
) -> plt.Figure:
    """Render a Seaborn color-coded summary table heatmap across all evaluated models.

    Column metrics are normalized to visually highlight rank performance (1.0 = best),
    with individual cells annotated with exact formatted score values.

    Args:
        benchmark_metrics: Dictionary mapping model names to metric dictionaries.
        save_path: Optional file path to persist the generated figure.
        figsize: Figure dimensions (width, height).
        title: Plot super title.

    Returns:
        The generated Matplotlib Figure instance.
    """
    models = list(benchmark_metrics.keys())
    metric_cols = [
        "Test Loss", "Accuracy (%)", "Macro Prec (%)", "Macro Rec (%)",
        "Macro F1 (%)", "Weighted F1 (%)", "Macro AUC", "Micro AUC",
    ]
    keys = [
        "test_loss", "test_acc", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "macro_auc", "micro_auc",
    ]

    raw_vals = np.array(
        [[benchmark_metrics[m].get(k, 0.0) for k in keys] for m in models],
        dtype=float,
    )

    annot_matrix = []
    for m in models:
        res = benchmark_metrics[m]
        row = [
            f"{res.get('test_loss', 0.0):.4f}",
            f"{res.get('test_acc', 0.0):.2f}%",
            f"{res.get('macro_precision', 0.0):.2f}%",
            f"{res.get('macro_recall', 0.0):.2f}%",
            f"{res.get('macro_f1', 0.0):.2f}%",
            f"{res.get('weighted_f1', 0.0):.2f}%",
            f"{res.get('macro_auc', 0.0):.4f}",
            f"{res.get('micro_auc', 0.0):.4f}",
        ]
        annot_matrix.append(row)
    annot_matrix = np.array(annot_matrix)

    # Column-wise normalization (higher is better, test_loss inverted)
    norm_vals = np.zeros_like(raw_vals, dtype=float)
    for c in range(raw_vals.shape[1]):
        col = raw_vals[:, c]
        rng = col.max() - col.min()
        denom = rng if rng > 1e-12 else 1.0
        if keys[c] == "test_loss":
            norm_vals[:, c] = (col.max() - col) / denom
        else:
            norm_vals[:, c] = (col - col.min()) / denom

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        norm_vals,
        annot=annot_matrix,
        fmt="",
        cmap="YlGnBu",
        xticklabels=metric_cols,
        yticklabels=models,
        cbar=True,
        cbar_kws={"label": "Normalized Performance Rank (Column-wise: 1.0 = Peak SOTA)"},
        linewidths=1.2,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 9.5, "fontweight": "bold"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=300, bbox_inches="tight")

    return fig


