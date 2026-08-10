"""
feature_extraction.py — shared expert-probability feature extraction.

Single source of truth for extracting softmax probabilities from the two SOTA
experts (ResNet18 + DenseNet121) with optional 2-view hflip TTA. Both the
training scripts (``src/experiments/stacking_mlp_train.py``,
``src/experiments/moe_router_train.py``) and the analysis notebooks import
these helpers, so the feature semantics can never drift between the artifacts
a script persists and the data a notebook analyzes.

Contracts (choose the return shape you need):
    base_features_stacked(model_a, model_b, loader, device, tta=False)
        -> X[N, 20], y[N]     # X = concat([p_a, p_b]) per-expert probabilities
    base_features_per_expert(model_a, model_b, loader, device, tta=False)
        -> p1[N, 10], p2[N, 10], y[N]
    ensemble_probs(model_a, model_b, loader, device, tta=False)
        -> p_ens[N, 10], y[N]  # 0.5 * (p1 + p2) — fixed soft-voting baseline
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


@torch.inference_mode()
def _extract(model_a: nn.Module, model_b: nn.Module, loader, device, tta: bool):
    """Yield (p_a[N,10], p_b[N,10], y[N]) for all samples in *loader*."""
    p1s, p2s, ys = [], [], []
    for x, y in loader:
        x = x.to(device)
        if tta:
            views = [x, torch.flip(x, dims=[3])]
            a = torch.stack([torch.softmax(model_a(v), 1) for v in views]).mean(0)
            b = torch.stack([torch.softmax(model_b(v), 1) for v in views]).mean(0)
        else:
            a = torch.softmax(model_a(x), 1)
            b = torch.softmax(model_b(x), 1)
        p1s.append(a.cpu())
        p2s.append(b.cpu())
        ys.append(y)
    return (torch.cat(p1s).numpy(), torch.cat(p2s).numpy(), torch.cat(ys).numpy())


def base_features_per_expert(model_a, model_b, loader, device, tta=False):
    """Return (p1[N,10], p2[N,10], y[N]) for ResNet & DenseNet experts."""
    return _extract(model_a, model_b, loader, device, tta)


def base_features_stacked(model_a, model_b, loader, device, tta=False):
    """Return (X[N,20], y[N]) with X = concat([p_a, p_b]) (stacking features)."""
    p1, p2, y = _extract(model_a, model_b, loader, device, tta)
    return np.concatenate([p1, p2], axis=1), y


def ensemble_probs(model_a, model_b, loader, device, tta=False):
    """Return (p_ens[N,10], y[N]) — fixed 0.5/0.5 soft-voting probabilities.

    Note: when the stacked features are already available, prefer the cheaper
    ``0.5 * (X[:, :10] + X[:, 10:])`` derivation over re-running the experts.
    """
    p1, p2, y = _extract(model_a, model_b, loader, device, tta)
    return 0.5 * (p1 + p2), y
