"""
EXP-01: Automated Hyperparameter Optimization (Optuna Sweep)
Run command:
    python -m src.experiments.exp_01_optuna_hpo --n-trials 15 --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import optuna

from data import get_cifar10_loaders
from src.models.build_model import build_resnet18
from src.training.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial, epochs: int = 3, device: str = "cuda") -> float:
    # 1. Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    opt_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # 2. Load DataLoaders
    train_loader, val_loader, _ = get_cifar10_loaders(
        batch_size=batch_size,
    )

    # 3. Build Model
    model = build_resnet18(num_classes=10, mode="finetune", device=device_obj)

    # 4. Setup Optimizer
    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    criterion = nn.CrossEntropyLoss()

    # 5. Train & Evaluate
    results = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device_obj,
        num_epochs=epochs,
        run_name=f"optuna_trial_{trial.number}",
    )


    val_acc = results["best_val_acc"]
    trial.set_user_attr("val_loss", results["best_val_loss"])

    return val_acc


def main():
    parser = argparse.ArgumentParser(description="EXP-01: Optuna HPO Sweep")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument("--db-path", type=str, default="experiments/optuna_study.db", help="SQLite DB output path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    storage = f"sqlite:///{args.db_path}"

    logger.info(f"Starting Optuna HPO Study: {args.n_trials} trials, {args.epochs} epochs each...")

    study = optuna.create_study(
        study_name="cifar10_resnet18_hpo",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, epochs=args.epochs), n_trials=args.n_trials)

    logger.info("=== OPTUNA STUDY COMPLETED ===")
    logger.info(f"Best Trial #{study.best_trial.number}")
    logger.info(f"Best Val Accuracy: {study.best_trial.value:.2f}%")
    logger.info(f"Best Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
