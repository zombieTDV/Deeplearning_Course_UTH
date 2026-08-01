"""
EXP-04: Native 32x32 Conv Stem Long-Epoch Training
Run command:
    python -m src.experiments.exp_04_stem_native_long_train --epochs 15
"""

from __future__ import annotations

import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataloader import get_cifar10_loaders
from data.transforms import get_train_transform, get_eval_transform
from src.models.build_model import build_resnet18_cifar_stem
from src.training.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exp_04(epochs: int = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Native 32x32 transforms (No upsampling to 224x224)
    train_transform = get_train_transform(resize_size=32, crop_size=32, augmentation=True)
    eval_transform = get_eval_transform(resize_size=32)

    train_loader, val_loader, _ = get_cifar10_loaders(
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=128,
    )

    # Build 32x32 Native Stem ResNet18
    model = build_resnet18_cifar_stem(num_classes=10, mode="finetune", device=device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"--- Running EXP-04: ResNet18 Native 32x32 Stem ({epochs} Epochs) ---")
    results = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=epochs,
        run_name="exp04_resnet18_native_32x32_long",
        scheduler=scheduler,
    )


    logger.info(f"=== EXP-04 RESULT: Best Val Accuracy = {results['best_val_acc']:.2f}% ===")


def main():
    parser = argparse.ArgumentParser(description="EXP-04: Native 32x32 Stem Extended Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    args = parser.parse_args()

    run_exp_04(epochs=args.epochs)


if __name__ == "__main__":
    main()
