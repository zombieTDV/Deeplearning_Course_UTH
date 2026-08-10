"""test_build_model.py — trainable-parameter structure per mode (TST-2, CQ-4)."""

import torch

from src.models.build_model import (
    build_densenet121_full_sota,
    build_resnet18,
    build_resnet18_full_sota,
    count_trainable_params,
)
from tests.conftest import requires_pretrained


@requires_pretrained
def test_resnet18_modes_unfreeze_expected_layers():
    # frozen: only the head is trainable
    m = build_resnet18(num_classes=10, mode="frozen", device=torch.device("cpu"))
    trainable = {name for name, p in m.named_parameters() if p.requires_grad}
    assert trainable and all("fc" in n for n in trainable), trainable

    # finetune: layer4 + head
    m = build_resnet18(num_classes=10, mode="finetune", device=torch.device("cpu"))
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("layer4.") for n in trainable)
    assert "fc.weight" in trainable
    assert not any(n.startswith("layer1.") for n in trainable)


@requires_pretrained
def test_resnet18_full_sota_unfreezes_layer3_layer4_head():
    m = build_resnet18_full_sota(num_classes=10, device=torch.device("cpu"))
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("layer3.") for n in trainable)
    assert any(n.startswith("layer4.") for n in trainable)
    assert "fc.weight" in trainable
    assert count_trainable_params(m) > 0


@requires_pretrained
def test_densenet121_full_sota_unfreezes_deep_blocks():
    m = build_densenet121_full_sota(num_classes=10, device=torch.device("cpu"))
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("features.denseblock3.") for n in trainable)
    assert any(n.startswith("features.denseblock4.") for n in trainable)
    assert any(n.startswith("features.norm5.") for n in trainable)
    assert any(n.startswith("classifier.") for n in trainable)
