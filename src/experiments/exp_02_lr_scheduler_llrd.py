"""
EXP-02: Learning Rate Schedulers & Layer-wise Learning Rate Decay (LLRD)
Run command:
    python -m src.experiments.exp_02_lr_scheduler_llrd --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import torch
import torch.nn as nn

from data import get_cifar10_loaders
from src.models.build_model import build_resnet18
from src.training.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exp_02(epochs: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_cifar10_loaders(batch_size=64)

    schedulers_to_test = ["None", "ReduceLROnPlateau", "CosineAnnealingLR"]

    summary_results = {}

    for sched_name in schedulers_to_test:
        logger.info(f"--- Running EXP-02 with Scheduler: {sched_name} ---")
        model = build_resnet18(num_classes=10, mode="finetune", device=device)

        # LLRD parameter grouping
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad], "lr": 1e-3},
            {"params": [p for n, p in model.named_parameters() if "layer4" in n and p.requires_grad], "lr": 1e-4},
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        if sched_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.5)
        elif sched_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss()

        results = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            run_name=f"exp02_{sched_name}",
            scheduler=scheduler,
        )


        summary_results[sched_name] = results["best_val_acc"]

    logger.info("=== EXP-02 SUMMARY RESULTS ===")
    for k, v in summary_results.items():
        logger.info(f"Scheduler [{k}]: Best Val Acc = {v:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="EXP-02: LR Schedulers & LLRD Experiment")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per run")
    args = parser.parse_args()

    run_exp_02(epochs=args.epochs)


if __name__ == "__main__":
    main()
