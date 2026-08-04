"""
catdog_confusion_reduction.py
=============================

Isolated evaluation & targeted reduction of cat/dog confusion on CIFAR-10.

Rationale
---------
The error analysis showed cat<->dog is the dominant error cluster of the
CIFAR-10 classifier (e.g. 61 true dogs predicted as cat + 30 true cats
predicted as dog = 91 cross-confusions in the soft-voting ensemble).
Accuracy has plateaued because these are concentrated, semantic errors.

This module runs an *isolated* benchmark on ONLY the cat/dog test samples
(no full-test-set benchmarking), then implements three targeted strategies to
reduce cat/dog confusion, and re-runs the isolated benchmark to measure
improvement.

Isolated benchmark
------------------
- Subset: CIFAR-10 test samples whose true class is cat(3) or dog(5) (2000).
- Metrics: isolated accuracy, per-class cat/dog accuracy, and the
  cat<->dog cross-confusion count (true-cat predicted-dog + true-dog
  predicted-cat).
- No full test-set numbers are used as the acceptance criterion.

Strategy 1 -- Hard-negative mining
    Freeze the ResNet18 backbone, train a 512->2 cat/dog arbiter head on a
    class-balanced cat/dog training subset in which the *hard negatives*
    (cat/dog training samples the base ensemble misclassifies) are
    oversampled.

Strategy 2 -- Focal loss / class-weighting
    Train the same cat/dog arbiter head with Focal Loss (gamma=2) and class
    weights that penalise misclassifying dog as cat, so the head stops
    "defaulting" ambiguous animals toward cat.

Strategy 3 -- Specialized ensemble member
    Train a small standalone CNN as a cat-vs-dog specialist; the ensemble
    delegates the cat/dog decision to it (arbitrated against the main
    ensemble's cat/dog belief) when the main ensemble is deciding between
    cat and dog.

All strategies leave every non-cat/dog decision unchanged, so any measured
improvement is attributable specifically to reduced cat/dog confusion.

Usage
-----
    python -m src.experiments.catdog_confusion_reduction --epochs 5
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.build_model import build_resnet18, build_densenet121
from src.eval.evaluate_model import load_checkpoint

# cat=3, dog=5 (CIFAR-10 class order)
CAT_IDX, DOG_IDX = 3, 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_ROOT = str(PROJECT_ROOT / "data" / "raw")
SPLIT_FILE = str(PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def eval_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Data: load CIFAR-10 from data/raw + the persisted project split
# ---------------------------------------------------------------------------
class CatDogSet(Subset):
    """Cat/dog (true class 3 or 5) subset preserving dataset order."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.labels = [dataset[i][1] for i in indices]


def load_catdog_splits(batch_size=64, num_workers=0, seed=0):
    """Return cat/dog train/val/test (feature-order + shuffled) loaders."""
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_idx, val_idx, test_idx = (split["train_indices"],
                                    split["val_indices"],
                                    split["test_indices"])

    tform = eval_transform()
    train_full = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, transform=tform)
    val_full = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, transform=tform)
    test_full = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, transform=tform)

    def catdog(dataset, indices):
        keep = [i for i in indices if dataset[i][1] in (CAT_IDX, DOG_IDX)]
        return CatDogSet(dataset, keep)

    tr_ds = catdog(train_full, train_idx)
    va_ds = catdog(val_full, val_idx)
    te_ds = catdog(test_full, test_idx)

    # Feature-order loaders (shuffle=False) -> order matches dataset indices.
    tr_feat = DataLoader(tr_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    va_feat = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    te_feat = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    # Shuffled loader for CNN specialist training.
    tr_shuf = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, generator=torch.Generator().manual_seed(seed))
    return (tr_feat, va_feat, te_feat, tr_shuf, tr_ds, va_ds, te_ds)


# ---------------------------------------------------------------------------
# ResNet18 feature extractor (512-dim embedding before fc)
# ---------------------------------------------------------------------------
def resnet18_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


@torch.inference_mode()
def embed_model(model, loader, device):
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(resnet18_features(model, x.to(device)).cpu())
        ys.append(y.clone())
    return torch.cat(feats), torch.cat(ys)


@torch.inference_mode()
def ensemble_probs(model_a, model_b, loader, device):
    model_a.eval()
    model_b.eval()
    all_p = []
    for x, _ in loader:
        x = x.to(device)
        pa = torch.softmax(model_a(x), dim=1)
        pb = torch.softmax(model_b(x), dim=1)
        all_p.append((0.5 * (pa + pb)).cpu())
    return torch.cat(all_p)


# ---------------------------------------------------------------------------
# Focal loss (multiclass, gamma) + optional class weight
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Correct focal loss: (1 - pt)**gamma * CE, class weight applied AFTER pt.

    The standard focal loss is FL = -(1 - p_t)^gamma * log(p_t), which
    DOWN-weights easy examples. The earlier implementation used pt**gamma
    (which up-weights easy examples) and folded the class weight into CE
    *before* computing pt = exp(-ce), which annihilated gradients on
    misclassified (weighted) examples and caused the S2 collapse to cat.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)                       # unweighted true-class prob
        focal = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            focal = focal * self.weight[targets]  # class weight applied last
        return focal.mean()


# ---------------------------------------------------------------------------
# Arbiter head (512 -> 2, cat vs dog) training on frozen ResNet18 features
# ---------------------------------------------------------------------------
def train_arbiter(X, y, mode="hard_neg", epochs=12, lr=1e-2, hard_idx=None,
                  seed=0, device=None, gamma=2.0, dog_weight=1.5):
    """Train a binary cat/dog head on frozen ResNet18 features.

    mode:
      'hard_neg' -> class-balanced, hard negatives (train errors) oversampled (S1)
      'focal'    -> Correct Focal Loss + mild class weighting (S2).
                    dog_weight should stay mild (1.0-1.5); a strong weight
                    (e.g. 3.0) over-corrects and collapses the cat class.
    """
    torch.manual_seed(seed)
    head = nn.Linear(512, 2).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    if mode == "hard_neg":
        n_cat = int((y == 0).sum())
        n_dog = int((y == 1).sum())
        big = max(n_cat, n_dog)
        w_cat = big / max(n_cat, 1)
        w_dog = big / max(n_dog, 1)
        # hard_idx may be a numpy array, list, tuple, empty, or None.
        # Use len() for the emptiness check -- `hard_idx or []` would call
        # bool(array) and raise on multi-element numpy arrays.
        if hard_idx is not None and len(hard_idx) > 0:
            hard_set = {int(i) for i in hard_idx}
        else:
            hard_set = set()
        sel = []
        for i in range(len(y)):
            base = w_cat if int(y[i]) == 0 else w_dog
            rep = int(round(base * (3 if i in hard_set else 1.0)))
            sel.extend([i] * max(rep, 1))
        # Sanity guard: oversampling should stay O(original * small factor).
        # A large sel here signals a class-weight/label bug that would otherwise
        # blow up memory (e.g. the DOG_IDX - CAT_IDX == 2 remap bug -> ~46 GB).
        assert len(sel) <= 10 * len(y), (
            f"oversampled selection too large ({len(sel)} for {len(y)} samples)")
        X_sel, y_sel = X[sel], y[sel]
        criterion = nn.CrossEntropyLoss()
    else:  # focal / class-weight
        weight = torch.tensor([1.0, dog_weight], dtype=torch.float, device=device)
        criterion = FocalLoss(gamma=gamma, weight=weight)
        X_sel, y_sel = X, y

    n = len(X_sel)
    idx = np.arange(n)
    for ep in range(epochs):
        rng = np.random.RandomState(seed + ep)
        rng.shuffle(idx)
        head.train()
        for bi in range(0, n, 256):
            b = idx[bi:bi + 256]
            xb = X_sel[b].to(device)
            yb = y_sel[b].to(device)
            opt.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            opt.step()
    return head


# ---------------------------------------------------------------------------
# Small CNN specialist for cat vs dog (Strategy 3)
# ---------------------------------------------------------------------------
class CatDogSpecialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.net(x))


def train_specialist(loader, epochs=5, lr=1e-3, device=None, seed=0):
    torch.manual_seed(seed)
    model = CatDogSpecialist().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            yb = (y == DOG_IDX).long().to(device)
            opt.zero_grad()
            loss = criterion(model(x.to(device)), yb)
            loss.backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# Isolated benchmark metrics (cat/dog only)
# ---------------------------------------------------------------------------
def report_isolated(name, preds, true):
    preds = np.asarray(preds)
    true = np.asarray(true)
    acc = (preds == true).mean() * 100.0
    cat_mask = true == CAT_IDX
    dog_mask = true == DOG_IDX
    cat_acc = (preds[cat_mask] == CAT_IDX).mean() * 100.0
    dog_acc = (preds[dog_mask] == DOG_IDX).mean() * 100.0
    cross = int(((true == CAT_IDX) & (preds == DOG_IDX)).sum()
                + ((true == DOG_IDX) & (preds == CAT_IDX)).sum())
    res = {"name": name, "isolated_acc": float(acc), "cat_acc": float(cat_acc),
           "dog_acc": float(dog_acc), "cross_confusions": cross}
    print(f"  {name:32s} isolated_acc={acc:6.2f}%  cat={cat_acc:5.1f}%  "
          f"dog={dog_acc:5.1f}%  cross_conf={cross:3d}")
    return res


# ---------------------------------------------------------------------------
# Arbitration: combine ensemble cat/dog belief with an arbiter posterior
# ---------------------------------------------------------------------------
def arbitrate(ens_probs, arbiter_logits, true):
    ens_probs = np.asarray(ens_probs)
    a = torch.softmax(torch.tensor(arbiter_logits).float(), dim=1).numpy()
    a_dog = a[:, 1]
    final = np.argmax(ens_probs, axis=1).copy()
    for i in range(len(final)):
        if final[i] in (CAT_IDX, DOG_IDX):
            e_cat, e_dog = ens_probs[i, CAT_IDX], ens_probs[i, DOG_IDX]
            norm_e_dog = e_dog / (e_cat + e_dog + 1e-12)
            combined_dog = 0.5 * norm_e_dog + 0.5 * a_dog[i]
            final[i] = DOG_IDX if combined_dog > 0.5 else CAT_IDX
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5, help="arbiter head epochs")
    ap.add_argument("--spec-epochs", type=int, default=5, help="specialist CNN epochs")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-dog-weight", type=float, default=1.5,
                    help="class weight on dog for S2 focal (mild, 1.0-1.5)")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    (tr_feat, va_feat, te_feat, tr_shuf, tr_ds, va_ds, te_ds) = load_catdog_splits(
        batch_size=args.batch, num_workers=0, seed=0)
    print(f"Cat/dog samples -> train: {len(tr_ds)}, val: {len(va_ds)}, "
          f"test: {len(te_ds)}")

    ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints"
    rn = build_resnet18(num_classes=10, mode="finetune", device=device)
    dn = build_densenet121(num_classes=10, mode="finetune", device=device)
    rn = load_checkpoint(rn, str(ckpt_dir / "ResNet18-finetune_best.pt"), device)
    dn = load_checkpoint(dn, str(ckpt_dir / "DenseNet121-finetune_best.pt"), device)
    rn.eval()
    dn.eval()

    # ---- Baseline on isolated cat/dog test ----
    ens_test = ensemble_probs(rn, dn, te_feat, device).numpy()
    te_true = np.array(te_ds.labels)
    baseline_preds = np.argmax(ens_test, axis=1)
    results = [report_isolated("Baseline ensemble (finetune x2)", baseline_preds, te_true)]

    # ---- Reference: persisted SOTA ensemble isolated (from saved results) ----
    try:
        ref = json.loads((PROJECT_ROOT / "experiments" / "results"
                          / "resnet_densenet_sota_ensemble_results.json").read_text())
        cm = np.array(ref["test_benchmark"]["cm_baseline"])
        ref_acc = 100.0 * (cm[CAT_IDX, CAT_IDX] + cm[DOG_IDX, DOG_IDX]) \
            / (cm[CAT_IDX].sum() + cm[DOG_IDX].sum())
        ref_cross = int(cm[CAT_IDX, DOG_IDX] + cm[DOG_IDX, CAT_IDX])
        print(f"\n  [reference] persisted SOTA ensemble isolated: "
              f"acc={ref_acc:.2f}%  cross_conf={ref_cross}")
    except Exception as e:  # noqa: BLE001
        print(f"  [reference] not available ({e})")

    # ---- Frozen ResNet18 features (leakage-free) ----
    X_tr, y_tr = embed_model(rn, tr_feat, device)
    X_te, y_te = embed_model(rn, te_feat, device)
    # Binary labels: cat->0, dog->1.  NOTE: (y - CAT_IDX) would give {0, 2}
    # because DOG_IDX - CAT_IDX == 2, not 1 -- that corrupted class weighting
    # and exploded the hard-negative oversampling (the ~46 GB alloc).
    y_tr_b = (y_tr == DOG_IDX).long()

    # Hard negatives = cat/dog TRAIN samples the base ensemble misclassifies.
    ens_tr = ensemble_probs(rn, dn, tr_feat, device).numpy()
    tr_pred = np.argmax(ens_tr, axis=1)
    tr_true = np.array(tr_ds.labels)
    hard_idx = np.where(tr_pred != tr_true)[0]
    print(f"Hard negatives on cat/dog train: {len(hard_idx)}")

    # ---- Strategy 1: hard-negative mining ----
    head_s1 = train_arbiter(X_tr, y_tr_b, mode="hard_neg", epochs=args.epochs,
                            hard_idx=hard_idx, seed=0, device=device)
    with torch.no_grad():
        arb_s1 = head_s1(X_te.to(device)).cpu().numpy()
    results.append(report_isolated("S1 Hard-negative mining",
                                   arbitrate(ens_test, arb_s1, te_true), te_true))

    # ---- Strategy 2: focal loss / class weighting ----
    head_s2 = train_arbiter(X_tr, y_tr_b, mode="focal", epochs=args.epochs,
                            seed=1, device=device, gamma=args.focal_gamma,
                            dog_weight=args.focal_dog_weight)
    with torch.no_grad():
        arb_s2 = head_s2(X_te.to(device)).cpu().numpy()
    results.append(report_isolated("S2 Focal loss / class-weight",
                                   arbitrate(ens_test, arb_s2, te_true), te_true))

    # ---- Strategy 3: specialized CNN member ----
    spec = train_specialist(tr_shuf, epochs=args.spec_epochs, device=device, seed=2)
    with torch.no_grad():
        all_logits = [spec(x.to(device)).cpu().numpy() for x, _ in te_feat]
    arb_s3 = np.concatenate(all_logits, axis=0)
    results.append(report_isolated("S3 Specialist CNN member",
                                   arbitrate(ens_test, arb_s3, te_true), te_true))

    # ---- Re-verify: improvement vs baseline ----
    base = results[0]["isolated_acc"]
    base_cross = results[0]["cross_confusions"]
    print("\n=== Re-verification (isolated cat/dog test) ===")
    print(f"{'Strategy':32s} {'dAcc':>7s} {'cross_conf':>11s} {'dCross':>7s}")
    for r in results[1:]:
        print(f"{r['name']:32s} {r['isolated_acc']-base:+6.2f}% "
              f"{r['cross_confusions']:>9d} {base_cross-r['cross_confusions']:+7d}")

    out = {"device": str(device), "focal_gamma": args.focal_gamma,
           "focal_dog_weight": args.focal_dog_weight, "epochs": args.epochs,
           "spec_epochs": args.spec_epochs, "results": results,
           "hard_negatives_train": int(len(hard_idx))}
    res_dir = PROJECT_ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "catdog_confusion_reduction.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("\nSaved -> experiments/results/catdog_confusion_reduction.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nElapsed: {time.time() - t0:.1f}s")
