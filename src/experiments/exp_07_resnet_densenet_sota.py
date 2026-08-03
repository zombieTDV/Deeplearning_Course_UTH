"""
exp_07_resnet_densenet_sota.py — EXP-07: ResNet18 & DenseNet121 Peak Accuracy Fine-Tuning & Ensemble.

Run command:
    python -m src.experiments.exp_07_resnet_densenet_sota --epochs 10
"""

from __future__ import annotations

import argparse
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataloader import get_cifar10_loaders
from data.transforms import get_advanced_train_transform, get_eval_transform
from src.models.build_model import build_resnet18, build_densenet121, set_parameter_requires_grad
from src.training.train_model import train_model, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_resnet18_full_sota(num_classes: int = 10, device: torch.device | None = None) -> nn.Module:
    """Build ResNet18 with deep feature unfreezing (Layer3 + Layer4 + FC) for peak accuracy."""
    model = build_resnet18(num_classes=num_classes, mode="frozen", device=device)
    set_parameter_requires_grad(model.layer3, True)
    set_parameter_requires_grad(model.layer4, True)
    set_parameter_requires_grad(model.fc, True)
    return model

def build_densenet121_full_sota(num_classes: int = 10, device: torch.device | None = None) -> nn.Module:
    """Build DenseNet121 with deep feature unfreezing (DenseBlock3 + DenseBlock4 + Classifier) for peak accuracy."""
    model = build_densenet121(num_classes=num_classes, mode="frozen", device=device)
    set_parameter_requires_grad(model.features.denseblock3, True)
    set_parameter_requires_grad(model.features.denseblock4, True)
    set_parameter_requires_grad(model.features.norm5, True)
    set_parameter_requires_grad(model.classifier, True)
    return model

def evaluate_ensemble(model_resnet: nn.Module, model_densenet: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate soft-voting Ensemble of ResNet18 + DenseNet121."""
    model_resnet.eval()
    model_densenet.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            out_resnet = model_resnet(images)
            out_densenet = model_densenet(images)

            # Soft-voting average of softmax probabilities
            prob_resnet = torch.softmax(out_resnet, dim=1)
            prob_densenet = torch.softmax(out_densenet, dim=1)
            prob_ensemble = 0.5 * (prob_resnet + prob_densenet)

            loss = criterion(prob_ensemble.log(), labels)
            total_loss += loss.item() * images.size(0)

            _, preds = prob_ensemble.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = total_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc

def run_exp_07(epochs: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("==========================================================")
    logger.info("  STARTING EXP-07: ResNet18 & DenseNet121 SOTA Peak Run")
    logger.info("==========================================================")

    # 1. Setup Advanced Transforms & Loaders
    train_transform = get_advanced_train_transform(
        resize_size=224,
        use_randaugment=True,
        use_random_erasing=True,
    )
    eval_transform = get_eval_transform(resize_size=224)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=64,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # -------------------------------------------------------------------------
    # 2. Train ResNet18 SOTA Peak Strategy
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 1/3: Training ResNet18 Peak Accuracy Strategy ---")
    model_resnet = build_resnet18_full_sota(num_classes=10, device=device)

    # Layer-wise Discriminative Learning Rates for ResNet18
    param_groups_resnet = [
        {"params": model_resnet.fc.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
        {"params": model_resnet.layer4.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": model_resnet.layer3.parameters(), "lr": 3e-5, "weight_decay": 1e-4},
    ]
    optimizer_resnet = torch.optim.AdamW(param_groups_resnet)
    scheduler_resnet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_resnet, T_max=epochs, eta_min=1e-6)

    start_resnet = time.time()
    results_resnet = train_model(
        model_resnet,
        train_loader,
        val_loader,
        criterion,
        optimizer_resnet,
        device,
        num_epochs=epochs,
        run_name="exp07_resnet18_sota_peak",
        scheduler=scheduler_resnet,
    )
    time_resnet = time.time() - start_resnet

    # -------------------------------------------------------------------------
    # 3. Train DenseNet121 SOTA Peak Strategy
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 2/3: Training DenseNet121 Peak Accuracy Strategy ---")
    model_densenet = build_densenet121_full_sota(num_classes=10, device=device)

    param_groups_densenet = [
        {"params": model_densenet.classifier.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
        {"params": model_densenet.features.norm5.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": model_densenet.features.denseblock4.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": model_densenet.features.denseblock3.parameters(), "lr": 3e-5, "weight_decay": 1e-4},
    ]
    optimizer_densenet = torch.optim.AdamW(param_groups_densenet)
    scheduler_densenet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_densenet, T_max=epochs, eta_min=1e-6)

    start_densenet = time.time()
    results_densenet = train_model(
        model_densenet,
        train_loader,
        val_loader,
        criterion,
        optimizer_densenet,
        device,
        num_epochs=epochs,
        run_name="exp07_densenet121_sota_peak",
        scheduler=scheduler_densenet,
    )
    time_densenet = time.time() - start_densenet

    # -------------------------------------------------------------------------
    # 4. Evaluate Soft-Voting Ensemble (ResNet18 + DenseNet121)
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 3/3: Evaluating ResNet18 + DenseNet121 Soft-Voting Ensemble ---")
    ens_loss, ens_acc = evaluate_ensemble(model_resnet, model_densenet, val_loader, nn.NLLLoss(), device)

    logger.info("==========================================================")
    logger.info("  EXP-07 FINAL BENCHMARK SUMMARY")
    logger.info("==========================================================")
    logger.info(f"  1. ResNet18 Peak SOTA      : Best Val Acc = {results_resnet['best_val_acc']:.2f}% | Val Loss = {results_resnet['best_val_loss']:.4f} ({time_resnet:.1f}s)")
    logger.info(f"  2. DenseNet121 Peak SOTA    : Best Val Acc = {results_densenet['best_val_acc']:.2f}% | Val Loss = {results_densenet['best_val_loss']:.4f} ({time_densenet:.1f}s)")
    logger.info(f"  3. Soft-Voting Ensemble 🏆  : Best Val Acc = {ens_acc:.2f}% | Val Loss = {ens_loss:.4f}")
    logger.info("==========================================================")

    return {
        "resnet18_results": results_resnet,
        "densenet121_results": results_densenet,
        "ensemble_val_acc": ens_acc,
        "ensemble_val_loss": ens_loss,
    }

def main():
    parser = argparse.ArgumentParser(description="EXP-07: ResNet18 & DenseNet121 Peak Accuracy Solution")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    run_exp_07(epochs=args.epochs)

if __name__ == "__main__":
    main()
