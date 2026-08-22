"""
test_eda_umap.py — Unit tests for UMAP feature extraction & dimensionality reduction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.eda.umap_cifar10_features import (
    compute_cluster_metrics,
    extract_model_features,
    run_umap,
)
from src.models.build_model import build_densenet121, build_resnet18


def test_extract_model_features_resnet18():
    """Verify feature hook extracts 512-dim features from ResNet18."""
    device = torch.device("cpu")
    model = build_resnet18(num_classes=10, mode="frozen", device=device)

    dummy_images = torch.randn(8, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (8,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=4)

    X, y = extract_model_features(model, "ResNet18-frozen", loader, device)
    assert X.shape == (8, 512), f"Expected shape (8, 512), got {X.shape}"
    assert y.shape == (8,), f"Expected shape (8,), got {y.shape}"
    assert X.dtype == np.float32


def test_extract_model_features_densenet121():
    """Verify feature hook extracts 1024-dim features from DenseNet121."""
    device = torch.device("cpu")
    model = build_densenet121(num_classes=10, mode="frozen", device=device)

    dummy_images = torch.randn(8, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (8,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=4)

    X, y = extract_model_features(model, "DenseNet121-frozen", loader, device)
    assert X.shape == (8, 1024), f"Expected shape (8, 1024), got {X.shape}"
    assert y.shape == (8,), f"Expected shape (8,), got {y.shape}"
    assert X.dtype == np.float32


def test_compute_cluster_metrics():
    """Verify cluster quality metrics calculation."""
    # Synthetic well-separated 2-cluster embeddings
    cluster_0 = np.random.normal(loc=[-5.0, -5.0], scale=0.5, size=(50, 2))
    cluster_1 = np.random.normal(loc=[5.0, 5.0], scale=0.5, size=(50, 2))
    X_emb = np.vstack([cluster_0, cluster_1])
    labels = np.array([0] * 50 + [1] * 50)

    metrics = compute_cluster_metrics(X_emb, labels)
    assert "silhouette_score" in metrics
    assert "davies_bouldin_index" in metrics
    assert "calinski_harabasz_index" in metrics
    assert metrics["silhouette_score"] > 0.5  # Well separated
    assert metrics["davies_bouldin_index"] < 1.0


def test_run_umap_smoke():
    """Verify UMAP reducer produces expected (N, 2) output."""
    X = np.random.randn(30, 64).astype(np.float32)
    embeddings = run_umap(X, n_neighbors=5, min_dist=0.1, seed=42)
    assert embeddings.shape == (30, 2)
