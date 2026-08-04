"""
feature_level_tta.py
====================

Feature-level experiment to push past the current SOTA ensemble ceiling:
    (a) test-time augmentation (TTA) -- horizontal-flip voting, and
    (b) feature-level fine-tuning -- unfreeze the top residual/dense block
        and re-train with LLRD + mild dog weighting on the full training set.

Measured on BOTH the full CIFAR-10 test set (10k) and the isolated cat/dog
test set (2k). Compares:
    baseline -> +TTA -> +feature fine-tune -> +feature fine-tune + TTA.

Setup
-----
- Base: ResNet18-sota_best.pt + DenseNet121-sota_best.pt (soft-voting ensemble).
- Data: CIFAR-10 from data/raw using the persisted project split.
- The fine-tune trains ONLY the unfrozen top block; all lower layers frozen,
  so it directly tests "unfreeze the top block."

Usage
-----
    python -m src.experiments.feature_level_tta --epochs 4 --batch 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.build_model import build_resnet18, build_densenet121
from src.eval.evaluate_model import load_checkpoint

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
CAT_IDX, DOG_IDX = 3, 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DATA_ROOT = str(PROJECT_ROOT / "data" / "raw")
SPLIT_FILE = str(PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(batch_size=64, num_workers=0, seed=0):
    """Full train/val/test loaders from data/raw using the persisted split."""
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    tr = Subset(torchvision.datasets.CIFAR10(DATA_ROOT, train=True,
                                             transform=train_transform()),
                split["train_indices"])
    va = Subset(torchvision.datasets.CIFAR10(DATA_ROOT, train=True,
                                             transform=eval_transform()),
                split["val_indices"])
    te = torchvision.datasets.CIFAR10(DATA_ROOT, train=False,
                                      transform=eval_transform())
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(va, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


def load_sota(device):
    ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints"
    rn = build_resnet18(num_classes=10, mode="finetune", device=device)
    dn = build_densenet121(num_classes=10, mode="finetune", device=device)
    rn = load_checkpoint(rn, str(ckpt_dir / "ResNet18-sota_best.pt"), device)
    dn = load_checkpoint(dn, str(ckpt_dir / "DenseNet121-sota_best.pt"), device)
    return rn, dn


@torch.inference_mode()
def predict_probs(model_a, model_b, loader, device, tta=False):
    """Soft-voting 10-class probabilities; optional hflip TTA (2 views)."""
    model_a.eval()
    model_b.eval()
    all_p, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        if tta:
            views = [x, torch.flip(x, dims=[3])]
            pa = torch.stack([torch.softmax(model_a(v), dim=1) for v in views]).mean(0)
            pb = torch.stack([torch.softmax(model_b(v), dim=1) for v in views]).mean(0)
        else:
            pa = torch.softmax(model_a(x), dim=1)
            pb = torch.softmax(model_b(x), dim=1)
        all_p.append((0.5 * (pa + pb)).cpu())
        all_y.append(y.clone())
    return torch.cat(all_p).numpy(), torch.cat(all_y).numpy()


def full_metrics(name, probs, targets):
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets).mean() * 100.0
    cm = confusion_matrix(targets, preds, labels=list(range(10)))
    rep = classification_report(targets, preds, labels=list(range(10)),
                                output_dict=True, zero_division=0)
    macro = rep["macro avg"]["f1-score"]
    return {"name": name, "accuracy": float(acc), "macro_f1": float(macro),
            "confusion_matrix": cm.tolist()}


def isolated_catdog(name, probs, targets):
    mask = (targets == CAT_IDX) | (targets == DOG_IDX)
    t, p = targets[mask], np.argmax(probs[mask], axis=1)
    acc = (p == t).mean() * 100.0
    cat_acc = (p[t == CAT_IDX] == CAT_IDX).mean() * 100.0
    dog_acc = (p[t == DOG_IDX] == DOG_IDX).mean() * 100.0
    cross = int(((t == CAT_IDX) & (p == DOG_IDX)).sum()
                + ((t == DOG_IDX) & (p == CAT_IDX)).sum())
    print(f"  [isolated] {name:34s} acc={acc:6.2f}%  cat={cat_acc:5.1f}%  "
          f"dog={dog_acc:5.1f}%  cross_conf={cross:3d}")
    return {"name": name, "isolated_acc": float(acc), "cat_acc": float(cat_acc),
            "dog_acc": float(dog_acc), "cross_confusions": cross}


def fine_tune_top_block(model, arch, loader, device, epochs=4,
                        lr_head=1e-3, lr_block=1e-5, dog_weight=1.5, seed=0):
    """Unfreeze the top block only and fine-tune with LLRD + dog weighting."""
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad = False
    if arch == "resnet18":
        for p in model.layer4.parameters():
            p.requires_grad = True
        param_groups = [
            {"params": model.layer4.parameters(), "lr": lr_block},
            {"params": model.fc.parameters(), "lr": lr_head},
        ]
    else:  # densenet121
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True
        for p in model.features.norm5.parameters():
            p.requires_grad = True
        param_groups = [
            {"params": model.features.denseblock4.parameters(), "lr": lr_block},
            {"params": model.features.norm5.parameters(), "lr": lr_block},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    weight = torch.ones(10)
    weight[DOG_IDX] = dog_weight
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    model.train()
    for ep in range(1, epochs + 1):
        tot, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            correct += (model(x).argmax(1) == y).sum().item()
            tot += y.size(0)
        print(f"    {arch}: epoch {ep}/{epochs}  train_acc={100*correct/tot:.2f}%")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-block", type=float, default=1e-5)
    ap.add_argument("--dog-weight", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=args.batch, num_workers=args.workers, seed=0)
    print(f"Train batches: {len(train_loader)}, test samples: {len(test_loader.dataset)}")

    rn, dn = load_sota(device)
    results_full, results_iso = [], []

    def evaluate(label, model_a, model_b, tta=False):
        p, t = predict_probs(model_a, model_b, test_loader, device, tta=tta)
        rf = full_metrics(label, p, t)
        results_full.append(rf)
        print(f"  [full ] {label:34s} accuracy={rf['accuracy']:6.2f}%  "
              f"macro_f1={rf['macro_f1']:.4f}")
        results_iso.append(isolated_catdog(label, p, t))
        return rf

    print("\n=== Baseline & TTA on the untouched SOTA ensemble ===")
    evaluate("Baseline ensemble (sota)", rn, dn, tta=False)
    evaluate("Baseline + TTA", rn, dn, tta=True)

    print("\n=== Feature-level fine-tune (unfreeze top block, full train) ===")
    rn_ft = build_resnet18(num_classes=10, mode="finetune", device=device)
    dn_ft = build_densenet121(num_classes=10, mode="finetune", device=device)
    rn_ft = load_checkpoint(rn_ft, str(PROJECT_ROOT / "experiments" / "checkpoints"
                                       / "ResNet18-sota_best.pt"), device)
    dn_ft = load_checkpoint(dn_ft, str(PROJECT_ROOT / "experiments" / "checkpoints"
                                       / "DenseNet121-sota_best.pt"), device)
    print("  Fine-tuning ResNet18 top block (layer4 + fc)...")
    rn_ft = fine_tune_top_block(rn_ft, "resnet18", train_loader, device,
                                epochs=args.epochs, lr_head=args.lr_head,
                                lr_block=args.lr_block, dog_weight=args.dog_weight)
    print("  Fine-tuning DenseNet121 top block (denseblock4 + norm5 + classifier)...")
    dn_ft = fine_tune_top_block(dn_ft, "densenet121", train_loader, device,
                                epochs=args.epochs, lr_head=args.lr_head,
                                lr_block=args.lr_block, dog_weight=args.dog_weight)
    del rn, dn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Fine-tuned ensemble ===")
    evaluate("Fine-tuned ensemble", rn_ft, dn_ft, tta=False)
    evaluate("Fine-tuned ensemble + TTA", rn_ft, dn_ft, tta=True)

    # ---- Summary vs baseline ----
    base = results_full[0]["accuracy"]
    base_iso = results_iso[0]["isolated_acc"]
    base_cross = results_iso[0]["cross_confusions"]
    print("\n=== Summary (vs baseline) ===")
    print(f"{'Method':34s} {'full_acc':>9s} {'dFull':>7s} {'isolated':>9s} "
          f"{'cross_conf':>10s}")
    for rf, ri in zip(results_full, results_iso):
        print(f"{rf['name']:34s} {rf['accuracy']:8.2f}% "
              f"{rf['accuracy']-base:+6.2f}% {ri['isolated_acc']:8.2f}% "
              f"{ri['cross_confusions']:>9d} "
              f"({base_cross-ri['cross_confusions']:+d})")

    out = {"device": str(device), "epochs": args.epochs, "lr_head": args.lr_head,
           "lr_block": args.lr_block, "dog_weight": args.dog_weight,
           "full": results_full, "isolated": results_iso}
    res_dir = PROJECT_ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "feature_level_tta.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nSaved -> experiments/results/feature_level_tta.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nElapsed: {time.time() - t0:.1f}s")
