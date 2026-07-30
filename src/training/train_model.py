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
    run_name: str | None = None,
    writer: SummaryWriter | None = None,
    save_dir: str = "experiments/checkpoints",
) -> dict:
    """Run a full training loop with validation, logging, and checkpointing.

    Args:
        model: PyTorch model (in ``eval()`` mode; will switch to train).
        train_loader: Training data.
        val_loader: Validation data.
        criterion: Loss function (e.g. ``nn.CrossEntropyLoss()``).
        optimizer: PyTorch optimizer.
        device: ``torch.device``.
        num_epochs: Number of full passes over the training set.
        run_name: Identifier for this run.  Used for checkpoint filename and
                  TensorBoard tag.  Auto-generated if ``None``.
        writer: ``SummaryWriter`` for TensorBoard logging.  If ``None``,
                logging is skipped.
        save_dir: Directory to save the best checkpoint.

    Returns:
        Dict with keys:
            - run_name, num_epochs
            - train_losses, val_losses (lists of float)
            - train_accs, val_accs (lists of float)
            - best_val_loss, best_epoch
            - best_state_path (path to saved .pt or None)
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_path: str | None = None

    epoch_iter = range(1, num_epochs + 1)

    for epoch in epoch_iter:
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - epoch_start

        # --- Logging ---
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            # Log learning rate (first param group)
            if optimizer.param_groups:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/lr", lr, epoch)

        if (epoch % 5 == 0) or (epoch == 1):
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

    # --- Final summary ---
    print(
        f"\n  Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f})  "
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
        "best_epoch": best_epoch,
        "best_state_path": best_state_path,
    }
