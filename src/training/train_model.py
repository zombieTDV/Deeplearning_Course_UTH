"""
train_model.py — Training loop, validation, full-state checkpointing, resumability.

Purpose
-------
Every training run must be fully logged and resumable: loss curves, optimizer
state, scheduler state, model parameters, random seeds and RNG states are all
persisted so an interrupted run can continue EXACTLY where it stopped.
See ``agents/rules/LOGGING_CHECKPOINT_RULES.md`` for the authoritative rules.

Checkpoint format (``torch.save`` dict, loaded with ``weights_only=True``):
    {
        "run_name": str,
        "epoch": int,                     # last completed epoch
        "global_step": int,               # total optimizer steps so far
        "best_val_loss": float, "best_val_acc": float, "best_epoch": int,
        "model_state_dict": dict,
        "optimizer_state_dict": dict,
        "scheduler_state_dict": dict | None,
        "history": {train_losses, val_losses, train_accs, val_accs},
        "config": {hyperparams, seed, transforms, ...},
        "rng": {seeds + serialized torch/numpy/python RNG states},
        "timestamp": str, "commit": str | None,
    }

Files per run (under ``experiments/runs/<ts>_<run_name>/``):
    checkpoints/<run_name>_best.pt   — best validation-loss state
    checkpoints/<run_name>_last.pt   — every-epoch state (for resume)
    logs/<run_name>.log              — timestamped log lines
    metrics/<run_name>_config.json   — run configuration snapshot
    metrics/<run_name>_history.jsonl — one JSON line per epoch
    tensorboard/                     — optional TensorBoard event files

Usage:
    from src.utils.run_logger import RunLogger
    from src.training.train_model import train_model

    logger = RunLogger("ResNet18-frozen", runs_root="experiments/runs")
    res = train_model(model, train_loader, val_loader, criterion, optimizer,
                      device, num_epochs=20, run_name="ResNet18-frozen",
                      logger=logger, seed=42)
    # resume after interruption:
    res = train_model(model, train_loader, val_loader, criterion, optimizer,
                      device, num_epochs=20, run_name="ResNet18-frozen",
                      logger=logger, seed=42, resume_from=logger.run_dir)
"""

from __future__ import annotations

import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.utils.run_logger import RunLogger, format_eta


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    num_epochs: int = 1,
    logger: RunLogger | None = None,
    progress_every: int = 1,
    max_batches_per_epoch: int = 0,
) -> tuple[float, float]:
    """Train model for one epoch with optional real-time batch progress.

    Args:
        max_batches_per_epoch: Cap on batches consumed in this epoch (0 = all).
            Used by smoke tests / CI to validate the loop cheaply.

    Returns:
        (average_loss, top1_accuracy_pct)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)
    if max_batches_per_epoch > 0:
        n_batches = min(n_batches, max_batches_per_epoch)
    t0 = time.perf_counter()
    batch_loss = 0.0
    batch_correct = 0
    batch_total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
            break
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item() * images.size(0)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        batch_total += labels.size(0)
        total += labels.size(0)
        batch_correct += predicted.eq(labels).sum().item()
        correct += predicted.eq(labels).sum().item()

        if logger is not None and progress_every > 0 and (batch_idx + 1) % progress_every == 0:
            bl = batch_loss / max(batch_total, 1)
            ba = 100.0 * batch_correct / max(batch_total, 1)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - t0
            done = (epoch - 1) * n_batches + (batch_idx + 1)
            total_batches = num_epochs * n_batches
            eta = elapsed / max(done, 1) * max(total_batches - done, 0)
            logger.progress(
                f"epoch {epoch:3d}/{num_epochs} "
                f"[{batch_idx + 1:4d}/{n_batches}] "
                f"loss={bl:.4f} acc={ba:5.1f}% lr={lr:.2e} "
                f"eta={format_eta(eta)}"
            )
            batch_loss = batch_correct = batch_total = 0

    if logger is not None:
        logger.end_progress()

    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on validation/test set.

    Returns:
        (average_loss, top1_accuracy_pct)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


# ---------------------------------------------------------------------------
# EarlyStopping Callback
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Early stops training if validation loss does not improve after a specified patience."""
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Counter: {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return self.early_stop


# ---------------------------------------------------------------------------
# Random-state capture/restore (exact resume)
# ---------------------------------------------------------------------------
def capture_rng_state() -> dict:
    """Serialize seeds + full RNG states of torch (CPU/GPU), numpy, and Python.

    Stored as hex strings / primitive values so the checkpoint stays safe to
    load with ``weights_only=True`` and is JSON-serializable.
    """
    torch_state = torch.random.get_rng_state().numpy().tobytes().hex()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = [s.numpy().tobytes().hex()
                           for s in torch.cuda.get_rng_state_all()]
        except Exception:
            cuda_states = None
    np_state = np.random.get_state()
    py_state = random.getstate()
    return {
        "seeds": {
            "torch": torch.initial_seed(),
            "numpy": int(np_state[2]),
            "python": int(py_state[1][0]),
        },
        "torch_cpu_hex": torch_state,
        "torch_cuda_hex": cuda_states,
        "numpy": {"version": np_state[0],
                  "keys": np_state[1].tolist(),
                  "pos": int(np_state[2]),
                  "has_gauss": bool(np_state[3]),
                  "gauss": np_state[4]},
        "python": list(py_state),
    }


def restore_rng_state(state: dict) -> None:
    """Restore RNG states previously captured by :func:`capture_rng_state`."""
    seeds = state.get("seeds", {})
    torch.manual_seed(seeds.get("torch", 42))
    if state.get("torch_cpu_hex"):
        buf = np.frombuffer(bytes.fromhex(state["torch_cpu_hex"]), dtype=np.uint8).copy()
        torch.set_rng_state(torch.from_numpy(buf))
    if state.get("torch_cuda_hex") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([
                torch.from_numpy(np.frombuffer(bytes.fromhex(h), dtype=np.uint8).copy())
                for h in state["torch_cuda_hex"]
            ])
        except Exception:
            pass
    if "numpy" in state:
        n = state["numpy"]
        np.random.set_state((n["version"], np.array(n["keys"], dtype=np.uint32),
                             n["pos"], n["has_gauss"], n["gauss"]))
    if "python" in state:
        random.setstate(tuple(state["python"]))


# ---------------------------------------------------------------------------
# Full-state checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_val_acc: float,
    best_epoch: int,
    history: dict,
    config: dict | None = None,
    early_stop_counter: int = 0,
    early_stop_triggered: bool = False,
) -> str:
    """Save a full training state (model + optimizer + scheduler + RNG + history).

    The result can be restored with :func:`load_checkpoint_state` and used to
    resume training exactly after an interruption. The file is written
    atomically (temp file + rename) so a crash mid-save never leaves a
    truncated checkpoint that a later ``--resume`` would pick up.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": config.get("run_name") if config else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "early_stop_counter": int(early_stop_counter),
        "early_stop_triggered": bool(early_stop_triggered),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "config": config or {},
        "rng": capture_rng_state(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": _git_commit(),
    }
    tmp_path = str(path) + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, str(path))
    return str(path)


def load_checkpoint_state(
    path: str | Path,
    device: torch.device | None = None,
) -> dict:
    """Load a full training-state checkpoint (safe ``weights_only=True``)."""
    return torch.load(path, map_location=device, weights_only=True)


def _git_commit() -> str | None:
    """Best-effort current git commit (for provenance in checkpoints)."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    epochs: int | None = None,
    run_name: str | None = None,
    experiment_name: str | None = None,
    scheduler: object | None = None,
    writer: SummaryWriter | None = None,
    save_dir: str = "experiments/checkpoints",
    early_stopping: bool = True,
    patience: int = 3,
    min_delta: float = 1e-4,
    logger: RunLogger | None = None,
    resume_from: str | Path | None = None,
    resume_from_best: bool = False,
    seed: int | None = None,
    config: dict | None = None,
    progress_every: int = 1,
    max_batches_per_epoch: int = 0,
) -> dict:
    """Run a full training loop with validation, logging, scheduler,
    EarlyStopping, full-state checkpointing, and resume support.

    New optional arguments (backward compatible):
        logger:        :class:`RunLogger` — real-time progress + file/JSONL logs.
        resume_from:   Path to a run directory or a ``*_last.pt`` checkpoint.
                       Restores model/optimizer/scheduler/RNG/history and
                       continues from the saved epoch.
        seed:          If given and *not* resuming, seeds all RNGs for
                       reproducibility (restored on resume from checkpoint).
        config:          Arbitrary run metadata written into the checkpoint and
                         ``metrics/<run>_config.json`` (5W1H description etc).
        progress_every:  Report batch progress every N batches (0 disables).
        max_batches_per_epoch: Cap batches per epoch (0 = unlimited). Used by
                         smoke tests / CI to validate the loop cheaply.
        resume_from_best: If resuming, load ``<run>_best.pt`` (rewind to the best
                         epoch) instead of ``<run>_last.pt``, reset the
                         early-stopping budget, and continue from best_epoch + 1.
                         Also overrides an already early-stopped run.

    Returns dict with keys: run_name, num_epochs, train_losses, val_losses,
    train_accs, val_accs, best_val_loss, best_val_acc, best_epoch,
    best_state_path, last_state_path, resume_from, completed_epochs,
    resumed_from_best.
    """
    if epochs is not None:
        num_epochs = epochs
    if experiment_name is not None:
        run_name = experiment_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Resolve checkpoint location -------------------------------------------------
    if logger is not None:
        ckpt_dir = Path(logger.checkpoint_dir)
    else:
        ckpt_dir = Path(save_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_state_path = str(ckpt_dir / f"{run_name}_best.pt")
    last_state_path = str(ckpt_dir / f"{run_name}_last.pt")

    # --- Resume state ---------------------------------------------------------------
    resume_payload = None
    start_epoch = 1
    global_step = 0
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            resume_name = f"{run_name}_{'best' if resume_from_best else 'last'}.pt"
            cand = resume_path / "checkpoints" / resume_name
            resume_path = cand if cand.exists() else resume_path
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if Path(resume_path).is_dir():
            raise FileNotFoundError(
                f"Cannot resume: no {run_name}_{'best' if resume_from_best else 'last'}.pt "
                f"inside {resume_path}. See "
                "agents/rules/LOGGING_CHECKPOINT_RULES.md#5-resume-procedure.")
        resume_payload = load_checkpoint_state(resume_path, device=device)
        was_early_stopped = bool(resume_payload.get("early_stop_triggered", False))
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload.get("global_step", 0))
        model.load_state_dict(resume_payload["model_state_dict"])
        if "optimizer_state_dict" in resume_payload and resume_payload["optimizer_state_dict"]:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if (scheduler is not None and "scheduler_state_dict" in resume_payload
                and resume_payload["scheduler_state_dict"]):
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if "rng" in resume_payload:
            restore_rng_state(resume_payload["rng"])
        if logger is not None:
            logger.log(f"[resume] continuing {run_name} from epoch {start_epoch} "
                       f"(global_step={global_step}, src={resume_path})")

        # Respect a completed early stop: an already-stopped run must not silently
        # keep training. Use resume_from_best=True (--force-resume) to rewind to the
        # best epoch and continue with a fresh early-stopping budget.
        if was_early_stopped and not resume_from_best:
            msg = (f"[resume] {run_name} already stopped early at epoch "
                   f"{resume_payload['epoch']} (best epoch {resume_payload['best_epoch']}, "
                   f"best_val_acc={resume_payload['best_val_acc']:.2f}%). Resume halted — "
                   f"the run is complete. Use --force-resume to continue from the best "
                   f"epoch with a fresh early-stopping budget.")
            if logger is not None:
                logger.log(msg)
            else:
                print(msg)
            return {
                "run_name": run_name,
                "num_epochs": num_epochs,
                "completed_epochs": start_epoch - 1,
                "train_losses": resume_payload.get("history", {}).get("train_losses", []),
                "val_losses": resume_payload.get("history", {}).get("val_losses", []),
                "train_accs": resume_payload.get("history", {}).get("train_accs", []),
                "val_accs": resume_payload.get("history", {}).get("val_accs", []),
                "best_val_loss": float(resume_payload["best_val_loss"]),
                "best_val_acc": float(resume_payload["best_val_acc"]),
                "best_epoch": int(resume_payload["best_epoch"]),
                "best_state_path": best_state_path,
                "last_state_path": last_state_path,
                "resume_from": str(resume_from),
                "global_step": global_step,
                "already_complete": True,
                "early_stopped_complete": True,
            }

    if start_epoch == 1:
        torch.manual_seed(seed if seed is not None else 42)
        np.random.seed(seed if seed is not None else 42)
        random.seed(seed if seed is not None else 42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed if seed is not None else 42)

    # --- History ---------------------------------------------------------------------
    if resume_payload is not None:
        history = dict(resume_payload.get("history", {}))
    else:
        history = {"train_losses": [], "val_losses": [],
                   "train_accs": [], "val_accs": []}

    best_val_loss = float(resume_payload["best_val_loss"]) if resume_payload else float("inf")
    best_val_acc = float(resume_payload["best_val_acc"]) if resume_payload else 0.0
    best_epoch = int(resume_payload["best_epoch"]) if resume_payload else -1

    run_config = dict(config or {})
    run_config.setdefault("run_name", run_name)
    run_config.setdefault("num_epochs", num_epochs)
    run_config.setdefault("seed", seed if seed is not None else 42)
    run_config.setdefault("start_epoch", start_epoch)
    if logger is not None and not Path(logger.config_path).exists():
        logger.write_config(run_config)

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None
    if resume_payload is not None and early_stopper is not None:
        early_stopper.best_loss = best_val_loss
        if resume_from_best:
            # Rewinding to the best epoch -> fresh early-stopping budget.
            early_stopper.counter = 0
        else:
            early_stopper.counter = int(resume_payload.get("early_stop_counter", 0))
        # Snapshot the resumed weights so an early stop triggered after resume
        # can still restore a best-weight snapshot instead of being skipped.
        early_stopper.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    epoch_iter = range(start_epoch, num_epochs + 1)
    completed_epochs = start_epoch - 1

    # --- Resume past the requested epoch budget: nothing to do ---
    if start_epoch > num_epochs:
        msg = (f"[resume] {run_name} already trained through epoch {start_epoch - 1} "
               f"(num_epochs={num_epochs}); nothing to do — increase --epochs to continue.")
        if logger is not None:
            logger.log(msg)
        else:
            print(msg)
        return {
            "run_name": run_name,
            "num_epochs": num_epochs,
            "completed_epochs": min(completed_epochs, num_epochs),
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "train_accs": history["train_accs"],
            "val_accs": history["val_accs"],
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_state_path": best_state_path,
            "last_state_path": last_state_path,
            "resume_from": str(resume_from) if resume_from else None,
            "global_step": global_step,
            "already_complete": True,
        }

    for epoch in epoch_iter:
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_epochs=num_epochs,
            logger=logger, progress_every=progress_every,
            max_batches_per_epoch=max_batches_per_epoch,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        global_step += len(train_loader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        elapsed = time.perf_counter() - epoch_start

        # --- Logging ---
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            if optimizer.param_groups:
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if logger is not None:
            logger.epoch_summary(epoch, {
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_acc": round(val_acc, 2),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(elapsed, 1),
            })
        elif (epoch % 5 == 0) or (epoch == 1) or (epoch == num_epochs):
            print(
                f"  Epoch {epoch:2d}/{num_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%  "
                f"[{elapsed:.1f}s]"
            )

        # --- Checkpoint + Early Stopping (one save; terminal state is flagged) ---
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_val_acc, best_epoch = val_loss, val_acc, epoch
        stopped = bool(early_stopper is not None and early_stopper(val_loss, model))
        save_checkpoint(last_state_path, model=model, optimizer=optimizer,
                        scheduler=scheduler, epoch=epoch, global_step=global_step,
                        best_val_loss=best_val_loss, best_val_acc=best_val_acc,
                        best_epoch=best_epoch, history=history, config=run_config,
                        early_stop_counter=early_stopper.counter if early_stopper else 0,
                        early_stop_triggered=stopped)
        if is_best:
            shutil.copyfile(last_state_path, best_state_path)

        if stopped:
            if logger is not None:
                logger.log(f"[EarlyStopping] Triggered at epoch {epoch}. "
                           f"Restoring best model weights...")
            else:
                print(f"\n  [EarlyStopping] Triggered at epoch {epoch}. Restoring best model weights...")
            if early_stopper.best_state_dict is not None:
                model.load_state_dict(early_stopper.best_state_dict)
            break

        completed_epochs = epoch
    if logger is not None:
        logger.log(f"[done] best epoch {best_epoch} "
                   f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%) "
                   f"checkpoints: {best_state_path}, {last_state_path}")

    return {
        "run_name": run_name,
        "num_epochs": num_epochs,
        "completed_epochs": completed_epochs,
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "train_accs": history["train_accs"],
        "val_accs": history["val_accs"],
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_state_path": best_state_path,
        "last_state_path": last_state_path,
        "resume_from": str(resume_from) if resume_from else None,
        "global_step": global_step,
        "resumed_from_best": resume_from_best,
    }
