"""Smoke test for LAB1 modules."""

def test_lab1_imports():
    from src.lab1 import data_utils, eval_utils, model_utils, train_utils, vis_utils
    assert data_utils is not None
    assert eval_utils is not None
    assert model_utils is not None
    assert train_utils is not None
    assert vis_utils is not None
