"""
EXP-05: Modern Vision Architecture (ConvNeXt & EfficientNet) Benchmark
Run command:
    python -m src.experiments.exp_05_model_arch_sweep --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import time
import torch
import torch.nn as nn

from data import get_cifar10_loaders
from src.models.build_model import (
    build_resnet18,
    build_densenet121,
    build_convnext_tiny,
    build_efficientnet_b0,
    count_all_params,
    count_trainable_params,
)
from src.training.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exp_05(epochs: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_cifar10_loaders(batch_size=64)

    models_to_test = {
        "ResNet18": build_resnet18,
        "DenseNet121": build_densenet121,
        "ConvNeXt_Tiny": build_convnext_tiny,
        "EfficientNet_B0": build_efficientnet_b0,
    }

    matrix = []

    for name, builder_fn in models_to_test.items():
        logger.info(f"--- Running EXP-05 Benchmark for Architecture: {name} ---")
        model = builder_fn(num_classes=10, mode="finetune", device=device)

        total_p = count_all_params(model)
        train_p = count_trainable_params(model)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-4,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        results = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            run_name=f"exp05_{name}",
            scheduler=scheduler,
        )

        elapsed = time.time() - start_time

        matrix.append({
            "name": name,
            "total_params": total_p,
            "trainable_params": train_p,
            "total_time_s": elapsed,
            "time_per_epoch": elapsed / epochs,
            "best_val_loss": results["best_val_loss"],
            "best_val_acc": results["best_val_acc"],
        })

    logger.info("=== EXP-05 COMPARISON MATRIX ===")
    logger.info(f"{'Model':<16} | {'Total P':<10} | {'Train P':<10} | {'Time/Ep (s)':<12} | {'Best Val Acc':<12}")
    logger.info("-" * 75)
    for row in matrix:
        logger.info(f"{row['name']:<16} | {row['total_params']:<10} | {row['trainable_params']:<10} | {row['time_per_epoch']:<12.2f} | {row['best_val_acc']:<12.2f}%")


def main():
    parser = argparse.ArgumentParser(description="EXP-05: Architecture Sweep Benchmark")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per model")
    args = parser.parse_args()

    run_exp_05(epochs=args.epochs)


if __name__ == "__main__":
    main()
