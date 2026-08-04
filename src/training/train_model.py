"""
train_model.py — Training loop, validation, TensorBoard logging.

Usage:
    from src.training.train_model import train_one_epoch, validate, train_model

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Or use the full wrapper:
    results = train_model(model, train_loader, val_loader, criterion,
                          optimizer, device, num_epochs=10, run_name="...")
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train model for one epoch.

    Returns:
        (average_loss, top1_accuracy_pct)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
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

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
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
) -> dict:
    """Run a full training loop with validation, logging, scheduler, EarlyStopping, and checkpointing."""
    if epochs is not None:
        num_epochs = epochs
    if experiment_name is not None:
        run_name = experiment_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    best_state_path: str | None = None

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

    epoch_iter = range(1, num_epochs + 1)

    for epoch in epoch_iter:
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        elapsed = time.time() - epoch_start

        # --- Logging ---
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            if optimizer.param_groups:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/lr", lr, epoch)

        if (epoch % 5 == 0) or (epoch == 1) or (epoch == num_epochs):
            print(
                f"  Epoch {epoch:2d}/{num_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%  "
                f"[{elapsed:.1f}s]"
            )

        # --- Checkpoint best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs(save_dir, exist_ok=True)
            best_state_path = os.path.join(save_dir, f"{run_name}_best.pt")
            torch.save(model.state_dict(), best_state_path)

        # --- Early Stopping Check ---
        if early_stopper is not None:
            if early_stopper(val_loss, model):
                print(f"\n  [EarlyStopping] Triggered at epoch {epoch}. Restoring best model weights...")
                if early_stopper.best_state_dict is not None:
                    model.load_state_dict(early_stopper.best_state_dict)
                break

    print(
        f"\n  Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%)  "
        f"Checkpoint: {best_state_path}"
    )

    return {
        "run_name": run_name,
        "num_epochs": num_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_state_path": best_state_path,
    }

