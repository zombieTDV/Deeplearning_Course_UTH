"""test_evaluate_model.py — evaluation metrics + comparison table (TST-2)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.eval.evaluate_model import (
    CIFAR10_CLASSES,
    evaluate,
    format_comparison_table,
    per_class_accuracy,
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
