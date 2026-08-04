"""
EXP-03: Advanced Data Augmentation (RandAugment & RandomErasing) & Label Smoothing
Run command:
    python -m src.experiments.exp_03_advanced_augmentations --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataloader import get_cifar10_loaders
from src.data.transforms import get_advanced_train_transform, get_eval_transform
from src.models.build_model import build_resnet18
from src.training.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exp_03(epochs: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Advanced Transforms
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

    model = build_resnet18(num_classes=10, mode="finetune", device=device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    logger.info("--- Running EXP-03: Advanced Augmentations + Label Smoothing ---")
    results = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=epochs,
        run_name="exp03_randaug_labelsmooth",
        scheduler=scheduler,
    )


    logger.info(f"=== EXP-03 RESULT: Best Val Accuracy = {results['best_val_acc']:.2f}% ===")


def main():
    parser = argparse.ArgumentParser(description="EXP-03: Advanced Data Augmentation Experiment")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    run_exp_03(epochs=args.epochs)


if __name__ == "__main__":
    main()
