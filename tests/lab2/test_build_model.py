"""test_build_model.py — trainable-parameter structure per mode (TST-2, CQ-4)."""

import torch
import torch.nn as nn

from src.lab2.models.build_model import (
    build_densenet121,
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


@requires_pretrained
def test_dropout_zero_keeps_plain_linear_head_backward_compatible():
    # dropout_rate=0 (default) must keep a plain nn.Linear head so existing
    # checkpoints (state_dict key "fc.weight") remain loadable.
    m = build_resnet18(num_classes=10, mode="frozen", device=torch.device("cpu"),
                       dropout_rate=0.0)
    assert isinstance(m.fc, nn.Linear)
    assert "fc.weight" in m.state_dict()
    m = build_densenet121(num_classes=10, mode="frozen", device=torch.device("cpu"),
                          dropout_rate=0.0)
    assert isinstance(m.classifier, nn.Linear)
    assert "classifier.weight" in m.state_dict()


@requires_pretrained
def test_dropout_wraps_head_in_sequential_dropout_linear():
    m = build_resnet18(num_classes=10, mode="frozen", device=torch.device("cpu"),
                       dropout_rate=0.5)
    assert isinstance(m.fc, nn.Sequential)
    assert isinstance(m.fc[0], nn.Dropout) and m.fc[0].p == 0.5
    assert isinstance(m.fc[1], nn.Linear) and m.fc[1].out_features == 10
    # wrapped head produces Sequential state_dict keys
    assert "fc.1.weight" in m.state_dict()


@requires_pretrained
def test_dropout_head_forwards_and_trains_only():
    m = build_resnet18(num_classes=10, mode="frozen", device=torch.device("cpu"),
                       dropout_rate=0.5)
    x = torch.randn(4, 3, 224, 224)
    # train mode: dropout active, outputs differ from inputs determinism-wise
    m.train()
    out_train = m(x)
    assert out_train.shape == (4, 10)
    # eval mode: dropout disabled
    m.eval()
    out_eval = m(x)
    assert out_eval.shape == (4, 10)


@requires_pretrained
def test_dropout_propagates_through_full_sota_builders():
    rn = build_resnet18_full_sota(num_classes=10, device=torch.device("cpu"),
                                  dropout_rate=0.3)
    assert isinstance(rn.fc, nn.Sequential) and isinstance(rn.fc[0], nn.Dropout)
    assert rn.fc[0].p == 0.3
    dn = build_densenet121_full_sota(num_classes=10, device=torch.device("cpu"),
                                     dropout_rate=0.3)
    assert isinstance(dn.classifier, nn.Sequential)
    assert isinstance(dn.classifier[0], nn.Dropout) and dn.classifier[0].p == 0.3
    # all-head dropout + deep-feature unfreeze still yields trainable params
    assert count_trainable_params(rn) > 0
    assert count_trainable_params(dn) > 0

