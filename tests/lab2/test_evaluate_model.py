"""test_evaluate_model.py — evaluation metrics + comparison table (TST-2)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.lab2.eval.evaluate_model import (
    CIFAR10_CLASSES,
    compute_comprehensive_benchmark,
    compute_model_metrics,
    evaluate,
    format_ascii_metrics_table,
    format_comparison_table,
    per_class_accuracy,
    plot_metrics_heatmap_table,
)


def _model_and_loader(n=32, bs=8, num_classes=4):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, num_classes))
    x = torch.randn(n, 8)
    y = torch.randint(0, num_classes, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=bs)
    return model, loader


def test_evaluate_returns_loss_and_accuracy():
    model, loader = _model_and_loader()
    loss, acc = evaluate(model, loader, torch.device("cpu"))
    assert 0.0 <= acc <= 100.0
    assert loss >= 0.0


def test_per_class_accuracy_shape():
    model, loader = _model_and_loader()
    pca, cm = per_class_accuracy(model, loader, torch.device("cpu"), num_classes=4)
    assert len(pca) == 4
    assert cm.shape == (4, 4)


def test_format_comparison_table_returns_string():
    rows = {"ResNet18-sota": {"mode": "sota", "test_loss": 0.12, "test_acc": 95.64,
                              "best_val_acc": 95.66, "per_class_acc": [90.0, 95.0]}}
    table = format_comparison_table(rows)
    assert isinstance(table, str) and "ResNet18-sota" in table and "95.64" in table


def test_cifar10_classes_constant():
    assert len(CIFAR10_CLASSES) == 10
    assert CIFAR10_CLASSES[3] == "cat" and CIFAR10_CLASSES[5] == "dog"


def test_compute_model_metrics():
    np.random.seed(42)
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    probs = np.zeros((8, 4))
    for i, y in enumerate(y_true):
        probs[i, y] = 0.9
        probs[i, (y + 1) % 4] = 0.1

    metrics = compute_model_metrics(y_true, probs, num_classes=4)
    assert metrics["test_acc"] == 100.0
    assert metrics["test_loss"] > 0.0
    assert metrics["macro_precision"] == 100.0
    assert metrics["macro_recall"] == 100.0
    assert metrics["macro_f1"] == 100.0
    assert metrics["macro_auc"] == 1.0
    assert metrics["micro_auc"] == 1.0
    assert len(metrics["per_class_acc"]) == 4


def test_compute_comprehensive_benchmark_and_ascii_table():
    np.random.seed(42)
    y_true = np.array([0, 1, 2, 3] * 5)
    probs_a = np.random.dirichlet(np.ones(4), size=len(y_true))
    probs_b = np.random.dirichlet(np.ones(4), size=len(y_true))

    models_probs = {"Model-A": probs_a, "Model-B": probs_b}
    benchmark = compute_comprehensive_benchmark(models_probs, y_true, num_classes=4)
    assert "Model-A" in benchmark and "Model-B" in benchmark

    table_str = format_ascii_metrics_table(benchmark)
    assert "Model-A" in table_str and "Model-B" in table_str
    assert "Macro Prec" in table_str and "Micro AUC" in table_str


def test_plot_metrics_heatmap_table(tmp_path):
    benchmark = {
        "Model-A": {"test_loss": 0.5, "test_acc": 85.0, "macro_precision": 84.0, "macro_recall": 85.0,
                    "macro_f1": 84.5, "weighted_f1": 84.5, "macro_auc": 0.95, "micro_auc": 0.96},
        "Model-B": {"test_loss": 0.3, "test_acc": 92.0, "macro_precision": 91.5, "macro_recall": 92.0,
                    "macro_f1": 91.8, "weighted_f1": 91.8, "macro_auc": 0.98, "micro_auc": 0.99},
    }
    save_file = tmp_path / "metrics_heatmap.png"
    fig = plot_metrics_heatmap_table(benchmark, save_path=save_file)
    assert save_file.exists()
    assert fig is not None


