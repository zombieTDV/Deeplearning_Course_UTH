"""
EXP-06: ConvNeXt-Tiny SOTA Combination (ConvNeXt-Tiny + RandAugment + Label Smoothing + CosineAnnealing)
Run command:
    python -m src.experiments.exp_06_convnext_advanced_sota --epochs 10
"""

from __future__ import annotations

import argparse
import logging
import time
import torch
import torch.nn as nn

from src.data.dataloader import get_cifar10_loaders
from src.data.transforms import get_advanced_train_transform, get_eval_transform
from src.models.build_model import build_convnext_tiny
from src.training.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exp_06(epochs: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("==========================================================")
    logger.info("  STARTING EXP-06: ConvNeXt-Tiny SOTA Combination Run")
    logger.info("==========================================================")

    # 1. Advanced Data Transforms
    train_transform = get_advanced_train_transform(
        resize_size=224,
        use_randaugment=True,
        use_random_erasing=True,
    )
    eval_transform = get_eval_transform(resize_size=224)

    train_loader, val_loader, _ = get_cifar10_loaders(
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=64,
    )

    # 2. Build ConvNeXt-Tiny Model
    model = build_convnext_tiny(num_classes=10, mode="finetune", device=device)

    # 3. Setup Optimizer, Scheduler, and Loss
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_time = time.time()

    # 4. Train Model
    results = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=epochs,
        run_name="exp06_convnext_sota_combination",
        scheduler=scheduler,
    )

    elapsed = time.time() - start_time

    logger.info("==========================================================")
    logger.info(f"  EXP-06 FINAL RESULT: Best Val Accuracy = {results['best_val_acc']:.2f}%")
    logger.info(f"  Best Val Loss = {results['best_val_loss']:.4f}")
    logger.info(f"  Total Execution Time = {elapsed / 60:.2f} minutes ({elapsed / epochs:.1f}s / epoch)")
    logger.info("==========================================================")

    return results

def main():
    parser = argparse.ArgumentParser(description="EXP-06: ConvNeXt-Tiny SOTA Combination")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    run_exp_06(epochs=args.epochs)

if __name__ == "__main__":
    main()
