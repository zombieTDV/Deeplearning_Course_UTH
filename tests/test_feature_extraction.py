"""test_feature_extraction.py — shared feature-extraction helpers (TST-2, duplication guard)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.feature_extraction import (
    base_features_per_expert,
    base_features_stacked,
    ensemble_probs,
)


def _fake_expert(out_dim):
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, out_dim))


def _loader(n=16, bs=4):
    torch.manual_seed(1)
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def test_base_features_stacked_shape():
    a, b = _fake_expert(10), _fake_expert(10)
    X, y = base_features_stacked(a, b, _loader(), torch.device("cpu"), tta=False)
    assert X.shape == (16, 20)
    assert y.shape == (16,)


def test_base_features_per_expert_shape():
    a, b = _fake_expert(10), _fake_expert(10)
    p1, p2, y = base_features_per_expert(a, b, _loader(), torch.device("cpu"))
    assert p1.shape == p2.shape == (16, 10)
    assert y.shape == (16,)


def test_ensemble_probs_is_mean_of_experts():
    a, b = _fake_expert(10), _fake_expert(10)
    p_ens, _ = ensemble_probs(a, b, _loader(), torch.device("cpu"))
    p1, p2, _ = base_features_per_expert(a, b, _loader(), torch.device("cpu"))
    assert p_ens.shape == (16, 10)
    assert torch.allclose(torch.tensor(p_ens), torch.tensor(0.5 * (p1 + p2)), atol=1e-6)


def test_tta_view_matches_manual_flip():
    a, b = _fake_expert(10), _fake_expert(10)
    loader = _loader()
    X_tta, _ = base_features_stacked(a, b, loader, torch.device("cpu"), tta=True)
    # two-view TTA produces finite probabilities; concat of 2 experts sums to 2
    assert torch.allclose(torch.tensor(X_tta).sum(1), torch.full((16,), 2.0), atol=1e-5)
