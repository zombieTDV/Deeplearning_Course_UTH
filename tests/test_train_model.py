"""test_train_model.py — 1-epoch training smoke, checkpointing + resume (TST-2)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.train_model import (
    load_checkpoint_state,
    save_checkpoint,
    train_model,
)
from src.utils.run_logger import RunLogger


def _tiny_loader(n=64, bs=16):
    torch.manual_seed(0)
    x = torch.randn(n, 8)
    y = torch.randint(0, 4, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True)


def _model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_train_model_one_epoch_smoke(tmp_path):
    logger = RunLogger("smoke", runs_root=tmp_path)
    model, opt = _model(), torch.optim.Adam(_model().parameters(), lr=1e-3)
    loader = _tiny_loader()
    res = train_model(model, loader, loader, nn.CrossEntropyLoss(), opt,
                      torch.device("cpu"), num_epochs=1, run_name="smoke",
                      logger=logger, seed=42, early_stopping=False)
    logger.close()

    assert res["completed_epochs"] == 1
    assert len(res["train_losses"]) == 1 and len(res["val_losses"]) == 1
    assert res["best_val_acc"] >= 0.0
    # both checkpoints exist and contain full state (model + rng + history)
    from pathlib import Path

    assert Path(res["best_state_path"]).exists()
    state = load_checkpoint_state(res["last_state_path"], torch.device("cpu"))
    assert state["epoch"] == 1
    assert "optimizer_state_dict" in state
    assert "rng" in state
    assert state["history"]["train_losses"] == res["train_losses"]


def test_save_load_checkpoint_roundtrip(tmp_path):
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model=model, optimizer=opt, epoch=3, global_step=9,
                    best_val_loss=0.5, best_val_acc=90.0, best_epoch=2,
                    history={"train_losses": [1.0, 0.8, 0.6]}, config={"run_name": "x"})
    state = load_checkpoint_state(path, torch.device("cpu"))
    assert state["epoch"] == 3
    assert state["global_step"] == 9
    assert state["early_stop_counter"] == 0
    assert state["history"]["train_losses"] == [1.0, 0.8, 0.6]


def test_exact_resume_matches_uninterrupted_run(tmp_path):
    """Resuming after 1 epoch reproduces an uninterrupted 2-epoch run (exact resume)."""

    def fresh():
        torch.manual_seed(1234)  # deterministic init, identical every call
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        return m, torch.optim.Adam(m.parameters(), lr=1e-3)

    def run(epochs, resume_from=None, tag=""):
        lg = RunLogger(tag, runs_root=tmp_path)
        m, o = fresh()
        r = train_model(m, _tiny_loader(), _tiny_loader(), nn.CrossEntropyLoss(), o,
                        torch.device("cpu"), num_epochs=epochs, run_name=tag,
                        logger=lg, seed=42, resume_from=resume_from,
                        early_stopping=False)
        lg.close()
        return r, lg

    rA, _ = run(epochs=2, tag="A")                       # uninterrupted 2-epoch
    rB1, lgB1 = run(epochs=1, tag="B")                    # 1 epoch
    rB2, _ = run(epochs=2, resume_from=lgB1.run_dir, tag="B")  # resume to 2
    assert rB2["train_losses"] == rA["train_losses"]     # exact continuation
    assert rB2["best_val_loss"] == rA["best_val_loss"]


def test_resume_past_budget_is_noop(tmp_path):
    lg = RunLogger("g", runs_root=tmp_path)
    m, o = _model(), torch.optim.Adam(_model().parameters(), lr=1e-3)
    r1 = train_model(m, _tiny_loader(), _tiny_loader(), nn.CrossEntropyLoss(), o,
                     torch.device("cpu"), num_epochs=2, run_name="g",
                     logger=lg, seed=42, early_stopping=False)
    lg.close()
    lg2 = RunLogger("g", runs_root=tmp_path)
    m2, o2 = _model(), torch.optim.Adam(_model().parameters(), lr=1e-3)
    r2 = train_model(m2, _tiny_loader(), _tiny_loader(), nn.CrossEntropyLoss(), o2,
                     torch.device("cpu"), num_epochs=2, run_name="g",
                     logger=lg2, seed=42, resume_from=r1["last_state_path"],
                     early_stopping=False)
    lg2.close()
    assert r2.get("already_complete") is True
    assert r2["completed_epochs"] == 2
    assert len(r2["train_losses"]) == 2


def _hist(n):
    return {"train_losses": [float(i) for i in range(n)],
            "val_losses": [float(i) for i in range(n)],
            "train_accs": [float(i) for i in range(n)],
            "val_accs": [float(i) for i in range(n)]}


def test_resume_respects_early_stop(tmp_path):
    """A run that already early-stopped must not silently continue past it."""
    lg = RunLogger("es", runs_root=tmp_path)
    m, o = _model(), torch.optim.Adam(_model().parameters(), lr=1e-3)
    save_checkpoint(lg.checkpoint_dir / "es_last.pt", model=m, optimizer=o, epoch=18,
                    global_step=100, best_val_loss=1.0, best_val_acc=90.0, best_epoch=17,
                    history=_hist(18), config={"run_name": "es"}, early_stop_triggered=True)
    lg.close()

    lg2 = RunLogger("es", runs_root=tmp_path)
    m2, o2 = _model(), torch.optim.Adam(_model().parameters(), lr=1e-3)
    res = train_model(m2, _tiny_loader(), _tiny_loader(), nn.CrossEntropyLoss(), o2,
                      torch.device("cpu"), num_epochs=20, run_name="es",
                      logger=lg2, seed=42, resume_from=lg.run_dir, early_stopping=True)
    lg2.close()
    assert res.get("early_stopped_complete") is True
    assert res["completed_epochs"] == 18     # halted at the stop epoch
    assert res["best_epoch"] == 17           # best epoch preserved


def test_force_resume_from_best_continues(tmp_path):
    """--force-resume rewinds to the best epoch with a fresh early-stopping budget."""
    lg = RunLogger("fr", runs_root=tmp_path)
    m, o = _model(), torch.optim.Adam(_model().parameters(), lr=1e-3)
    save_checkpoint(lg.checkpoint_dir / "fr_best.pt", model=m, optimizer=o, epoch=17,
                    global_step=90, best_val_loss=0.9, best_val_acc=92.0, best_epoch=17,
                    history=_hist(17), config={"run_name": "fr"}, early_stop_triggered=True)
    save_checkpoint(lg.checkpoint_dir / "fr_last.pt", model=m, optimizer=o, epoch=18,
                    global_step=100, best_val_loss=0.9, best_val_acc=92.0, best_epoch=17,
                    history=_hist(18), config={"run_name": "fr"}, early_stop_triggered=True)
    lg.close()

    lg2 = RunLogger("fr", runs_root=tmp_path)
    m2, o2 = _model(), torch.optim.Adam(_model().parameters(), lr=1e-3)
    res = train_model(m2, _tiny_loader(), _tiny_loader(), nn.CrossEntropyLoss(), o2,
                      torch.device("cpu"), num_epochs=18, run_name="fr",
                      logger=lg2, seed=42, resume_from=lg.run_dir,
                      resume_from_best=True, early_stopping=True)
    lg2.close()
    assert res["resumed_from_best"] is True
    assert res["completed_epochs"] == 18        # resumed at best_epoch+1 (17->18)
    assert res["best_epoch"] == 17              # best floor preserved
    assert len(res["train_losses"]) == 18       # 17 prior + 1 new epoch
