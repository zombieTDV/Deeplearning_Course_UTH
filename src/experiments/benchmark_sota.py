"""
benchmark_sota.py
=================

Real benchmark of the SOTA checkpoints on the CIFAR-10 test set.

Models evaluated (10-class, CIFAR-10):
    - ResNet18-sota_best.pt
    - DenseNet121-sota_best.pt
    - Soft-Voting Ensemble  (0.5 * P_ResNet + 0.5 * P_DenseNet)

Metrics reported (full test set, 10,000 samples):
    - test loss + overall top-1 accuracy
    - per-class precision / recall / F1 (sklearn)
    - macro-F1 and micro-F1
    - 10x10 confusion matrix

Isolated cat/dog benchmark (2000 cat/dog test samples only):
    - isolated accuracy, per-class cat/dog accuracy
    - cat<->dog cross-confusion count
    (target reference from saved results: 93.30% / cross_conf 91)

Setup
-----
- Data: CIFAR-10 auto-loaded from data/raw (no re-download).
- Checkpoints: experiments/checkpoints/<name>_sota_best.pt
- Uses ImageNet-compatible eval transform (Resize 224 + Normalize).

Usage
-----
    python -m src.experiments.benchmark_sota --batch 64
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
from torch.utils.data import DataLoader
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


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_test_loader(batch_size=64, num_workers=0):
    """CIFAR-10 test DataLoader with the ImageNet-compatible eval transform."""
    tform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_full = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, transform=tform)
    loader = DataLoader(test_full, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    return loader, test_full


@torch.inference_mode()
def collect_predictions(model, loader, device):
    """Return (logits[N,10], targets[N]) aligned to loader order."""
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        targets.append(y.clone())
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def metrics_from_preds(name, logits, targets, num_classes=10):
    """Full-test metrics: accuracy, per-class P/R/F1, confusion matrix."""
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets).mean() * 100.0
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    # per-class P/R/F1
    rep = classification_report(targets, preds, labels=list(range(num_classes)),
                                output_dict=True, zero_division=0)
    per_class = []
    for c in range(num_classes):
        per_class.append({
            "class": CIFAR10_CLASSES[c],
            "precision": rep[str(c)]["precision"],
            "recall": rep[str(c)]["recall"],
            "f1": rep[str(c)]["f1-score"],
        })
    macro_f1 = rep["macro avg"]["f1-score"]
    micro_f1 = rep["weighted avg"]["f1-score"]
    return {
        "name": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def isolated_catdog_metrics(name, logits, targets):
    """Isolated cat/dog benchmark (true cat or dog test samples only)."""
    mask = (targets == CAT_IDX) | (targets == DOG_IDX)
    t, lg = targets[mask], logits[mask]
    preds = np.argmax(lg, axis=1)
    acc = (preds == t).mean() * 100.0
    cat_mask, dog_mask = t == CAT_IDX, t == DOG_IDX
    cat_acc = (preds[cat_mask] == CAT_IDX).mean() * 100.0
    dog_acc = (preds[dog_mask] == DOG_IDX).mean() * 100.0
    cross = int(((t == CAT_IDX) & (preds == DOG_IDX)).sum()
                + ((t == DOG_IDX) & (preds == CAT_IDX)).sum())
    print(f"  [isolated] {name:30s} acc={acc:6.2f}%  cat={cat_acc:5.1f}%  "
          f"dog={dog_acc:5.1f}%  cross_conf={cross:3d}")
    return {"name": name, "isolated_acc": float(acc), "cat_acc": float(cat_acc),
            "dog_acc": float(dog_acc), "cross_confusions": cross}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    loader, _ = build_test_loader(batch_size=args.batch, num_workers=args.workers)
    ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints"

    # ---- Load SOTA models ----
    rn = build_resnet18(num_classes=10, mode="finetune", device=device)
    dn = build_densenet121(num_classes=10, mode="finetune", device=device)
    rn = load_checkpoint(rn, str(ckpt_dir / "ResNet18-sota_best.pt"), device)
    dn = load_checkpoint(dn, str(ckpt_dir / "DenseNet121-sota_best.pt"), device)
    print("Loaded ResNet18-sota_best.pt and DenseNet121-sota_best.pt\n")

    # ---- Collect logits ----
    z_rn, y = collect_predictions(rn, loader, device)
    z_dn, _ = collect_predictions(dn, loader, device)
    z_ens = 0.5 * (z_rn + z_dn)  # soft-voting on logits == avg probabilities
    del rn, dn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Full-test metrics ----
    print("=== Full test-set benchmark (10,000 samples) ===")
    results = []
    for name, z in [("ResNet18-sota", z_rn), ("DenseNet121-sota", z_dn),
                    ("Soft-Voting Ensemble", z_ens)]:
        r = metrics_from_preds(name, z, y)
        results.append(r)
        print(f"  {name:24s} accuracy={r['accuracy']:6.2f}%  "
              f"macro_f1={r['macro_f1']:.4f}  micro_f1={r['micro_f1']:.4f}")

    # ---- Isolated cat/dog benchmark ----
    print("\n=== Isolated cat/dog benchmark (2,000 samples) ===")
    iso = []
    for name, z in [("ResNet18-sota", z_rn), ("DenseNet121-sota", z_dn),
                    ("Soft-Voting Ensemble", z_ens)]:
        iso.append(isolated_catdog_metrics(name, z, y))

    # ---- Reference from saved results ----
    try:
        ref = json.loads((PROJECT_ROOT / "experiments" / "results"
                          / "resnet_densenet_sota_ensemble_results.json").read_text())
        cm = np.array(ref["test_benchmark"]["cm_baseline"])
        ref_acc = 100.0 * (cm[CAT_IDX, CAT_IDX] + cm[DOG_IDX, DOG_IDX]) \
            / (cm[CAT_IDX].sum() + cm[DOG_IDX].sum())
        ref_cross = int(cm[CAT_IDX, DOG_IDX] + cm[DOG_IDX, CAT_IDX])
        print(f"\n  [reference] persisted ensemble isolated: acc={ref_acc:.2f}% "
              f"cross_conf={ref_cross}")
    except Exception as e:  # noqa: BLE001
        print(f"\n  [reference] not available ({e})")

    # ---- Persist ----
    out = {"device": str(device), "results": results, "isolated": iso}
    res_dir = PROJECT_ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "sota_benchmark.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nSaved -> experiments/results/sota_benchmark.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nElapsed: {time.time() - t0:.1f}s")
