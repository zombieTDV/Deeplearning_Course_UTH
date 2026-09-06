"""Tests for training model initialization, classifier dropout kwarg, SWA, LLRD, and early-stopping checkpoint state."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.lab3.training.imdb_sentiment_train import _full_state, _get_llrd_optimizer_grouped_parameters
from src.lab3.utils.checkpoint_utils import average_checkpoints, safe_load_checkpoint, save_checkpoint


def test_classifier_dropout_constructor_kwarg():
    """Verify classifier dropout is applied during model instantiation via kwarg."""
    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2, seq_classif_dropout=0.35)
    assert config.seq_classif_dropout == 0.35

    model = AutoModelForSequenceClassification.from_config(config)
    assert model.dropout.p == 0.35


def test_full_state_early_stop_triggered():
    """Verify _full_state correctly records the early_stop_triggered flag."""
    mock_trainer = MagicMock()
    mock_trainer.model.state_dict.return_value = {"param": torch.tensor([1.0])}
    mock_trainer.optimizer.state_dict.return_value = {}
    mock_trainer.lr_scheduler = None
    mock_trainer.state.epoch = 2.0
    mock_trainer.state.global_step = 600

    state_stopped = _full_state(
        trainer=mock_trainer,
        config={"test": True},
        history=[],
        best=None,
        early_stop_triggered=True,
    )
    assert state_stopped["early_stop_triggered"] is True
    assert state_stopped["epoch"] == 2
    assert state_stopped["global_step"] == 600

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_last.pt"
        save_checkpoint(ckpt_path, state_stopped)

        loaded = safe_load_checkpoint(ckpt_path)
        assert loaded["early_stop_triggered"] is True
        assert loaded["epoch"] == 2


def test_average_checkpoints_swa():
    """Verify Stochastic Weight Averaging (SWA) averages parameters across checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt1 = tmp_path / "c1.pt"
        ckpt2 = tmp_path / "c2.pt"
        out_ckpt = tmp_path / "swa.pt"

        state1 = {"model_state_dict": {"w": torch.tensor([2.0, 4.0])}, "config": {}}
        state2 = {"model_state_dict": {"w": torch.tensor([4.0, 8.0])}, "config": {}}

        save_checkpoint(ckpt1, state1)
        save_checkpoint(ckpt2, state2)

        result_path = average_checkpoints([ckpt1, ckpt2], out_ckpt)
        assert result_path.exists()

        averaged = safe_load_checkpoint(out_ckpt)
        assert torch.allclose(averaged["model_state_dict"]["w"], torch.tensor([3.0, 6.0]))
        assert averaged["swa_num_checkpoints"] == 2


def test_llrd_parameter_groups():
    """Verify Layer-wise Learning Rate Decay builds non-empty parameter groups with decaying LRs."""
    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config)

    groups = _get_llrd_optimizer_grouped_parameters(model, base_lr=2e-5, weight_decay=0.01, decay_factor=0.8)
    assert len(groups) > 0
    # Head learning rate should equal base_lr (2e-5)
    assert groups[0]["lr"] == 2e-5
    # Bottom layer learning rates should decay (< 2e-5)
    lrs = [g["lr"] for g in groups]
    assert min(lrs) < 2e-5


def test_full_state_callback_best_metric_accuracy():
    """Verify FullStateCallback tracks maximum eval_accuracy when specified."""
    from src.lab3.training.imdb_sentiment_train import FullStateCallback

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.model.state_dict.return_value = {"weight": torch.tensor([1.0])}
        mock_trainer.optimizer.state_dict.return_value = {}
        mock_trainer.lr_scheduler = None
        mock_trainer.state.epoch = 1.0
        mock_trainer.state.global_step = 300

        callback = FullStateCallback(
            run_dir=Path(tmp_dir),
            run_name="test_run",
            config={},
            logger=mock_logger,
            metric_name="eval_accuracy",
            greater_is_better=True,
        )
        callback.attach(mock_trainer)

        # Eval 1: Loss 0.25, Accuracy 0.88
        state_mock = MagicMock(epoch=1.0, global_step=300)
        control_mock = MagicMock(should_training_stop=False)
        callback.on_evaluate(None, state_mock, control_mock, metrics={"eval_loss": 0.25, "eval_accuracy": 0.88, "eval_f1": 0.88})

        assert callback.best["eval_accuracy"] == 0.88

        # Eval 2: Loss 0.30 (higher/worse), Accuracy 0.91 (higher/better)
        state_mock = MagicMock(epoch=2.0, global_step=600)
        callback.on_evaluate(None, state_mock, control_mock, metrics={"eval_loss": 0.30, "eval_accuracy": 0.91, "eval_f1": 0.91})

        # Best metric must be updated to Accuracy 0.91 even though loss went up!
        assert callback.best["eval_accuracy"] == 0.91
        assert callback.best["eval_loss"] == 0.30

