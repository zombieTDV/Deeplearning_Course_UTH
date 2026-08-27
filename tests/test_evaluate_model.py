"""test_evaluate_model.py — evaluation metrics + benchmark table unit tests."""

import numpy as np

from src.eval.evaluate_model import (
    compute_comprehensive_benchmark,
    compute_model_metrics,
    format_ascii_metrics_table,
    plot_metrics_heatmap_table,
)


def test_compute_model_metrics():
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probs = np.zeros((8, 2))
    for i, y in enumerate(y_true):
        probs[i, y] = 0.9
        probs[i, 1 - y] = 0.1

    metrics = compute_model_metrics(y_true, probs, num_classes=2)
    assert metrics["test_acc"] == 100.0
    assert metrics["test_loss"] > 0.0
    assert metrics["macro_precision"] == 100.0
    assert metrics["macro_recall"] == 100.0
    assert metrics["macro_f1"] == 100.0
    assert metrics["macro_auc"] == 1.0
    assert metrics["micro_auc"] == 1.0
    assert len(metrics["per_class_acc"]) == 2


def test_compute_comprehensive_benchmark_and_ascii_table():
    np.random.seed(42)
    y_true = np.array([0, 1] * 10)
    probs_a = np.random.dirichlet(np.ones(2), size=len(y_true))
    probs_b = np.random.dirichlet(np.ones(2), size=len(y_true))

    models_probs = {"Model-A": probs_a, "Model-B": probs_b}
    benchmark = compute_comprehensive_benchmark(models_probs, y_true, num_classes=2)
    assert "Model-A" in benchmark and "Model-B" in benchmark

    table_str = format_ascii_metrics_table(benchmark)
    assert "Model-A" in table_str and "Model-B" in table_str
    assert "Macro Prec" in table_str and "Micro AUC" in table_str


def test_plot_metrics_heatmap_table(tmp_path):
    benchmark = {
        "Model-A": {
            "test_loss": 0.5,
            "test_acc": 85.0,
            "macro_precision": 84.0,
            "macro_recall": 85.0,
            "macro_f1": 84.5,
            "weighted_f1": 84.5,
            "macro_auc": 0.95,
            "micro_auc": 0.96,
        },
        "Model-B": {
            "test_loss": 0.3,
            "test_acc": 92.0,
            "macro_precision": 91.5,
            "macro_recall": 92.0,
            "macro_f1": 91.8,
            "weighted_f1": 91.8,
            "macro_auc": 0.98,
            "micro_auc": 0.99,
        },
    }
    save_file = tmp_path / "metrics_heatmap.png"
    fig = plot_metrics_heatmap_table(benchmark, save_path=save_file)
    assert save_file.exists()
    assert fig is not None
