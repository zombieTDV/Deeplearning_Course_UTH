"""
stacking_mlp_train.py — Train the stacking meta-model (script-only training).

Policy: notebooks never train.  This script runs the full stacking pipeline:

    1. load the two SOTA experts (checkpoints from the training script);
    2. extract 20-dim probability features on val/test (with & without TTA);
    3. train two small MLP meta-models (no-TTA and TTA feature sets);
    4. persist artifacts + a JSONL history + config for notebook analysis.

Artifacts (experiments/results/stacking_mlp/):
    stacking_mlp_artifacts.npz  # X_val, X_valT, X_test, X_testT, y_val, y_test,
                                # p_ens, p_ensT, p_mlp, p_mlpT (test probabilities)
    mlp_notta.pt, mlp_tta.pt    # meta-model state dicts
    config.json                 # 5W1H context: what/why/how/when/where/who
    history.jsonl               # per-epoch val accuracy of both meta-models

Usage:
    python -m src.experiments.stacking_mlp_train [--seed 0] [--epochs 40]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.build_model import build_resnet18, build_densenet121
from src.utils.checkpoint_utils import find_best_checkpoint, load_model_weights
from src.utils.feature_extraction import base_features_stacked
from src.utils.run_logger import RunLogger

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "stacking_mlp"
DATA_ROOT = str(PROJECT_ROOT / "data" / "raw")
SPLIT_FILE = str(PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")


class StackingMLP(nn.Module):
    """[20 -> 64 -> ReLU -> 10] meta-model over concatenated expert probabilities."""

    def __init__(self, in_dim: int = 20, hidden: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_loaders(batch_size: int = 64):
    """Build val/test loaders (224x224, ImageNet normalization, fixed split)."""
    import torchvision
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    tform = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    split = json.loads(Path(SPLIT_FILE).read_text())

    def make_loader(train, indices):
        ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=train, transform=tform)
        if indices is not None:
            ds = Subset(ds, indices)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    return (make_loader(True, split["val_indices"]),
            make_loader(False, None))


def load_experts(device):
    """Load the two SOTA experts from the latest training artifacts."""
    rn = build_resnet18(num_classes=10, mode="finetune", device=device)
    dn = build_densenet121(num_classes=10, mode="finetune", device=device)
    rp, dp = (find_best_checkpoint("ResNet18-sota"),
              find_best_checkpoint("DenseNet121-sota"))
    if rp is None or dp is None:
        raise FileNotFoundError(
            "SOTA expert checkpoints not found. Run first: "
            "python -m src.training.train_lab2_models --modes sota")
    rn = load_model_weights(rn, rp, device)
    dn = load_model_weights(dn, dp, device)
    return rn, dn


def train_mlp(X, y, X_val, y_val, device, epochs=40, lr=1e-3, seed=0,
              logger=None, tag=""):
    """Train one meta-model; logs per-epoch val accuracy."""
    torch.manual_seed(seed)
    model = StackingMLP(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.long).to(device)
    for ep in range(epochs):
        idx = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed + ep))
        model.train()
        for b in range(0, len(Xt), 256):
            bi = idx[b:b + 256]
            opt.zero_grad()
            loss = crit(model(Xt[bi].to(device)), yt[bi].to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xv).argmax(1).cpu().numpy() == y_val).mean() * 100.0)
        best = max(best, va)
        if logger is not None:
            logger.append_metrics(ep + 1, {f"val_acc{tag}": round(va, 2)})
    return model, best


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger = RunLogger("stacking_mlp", runs_root=PROJECT_ROOT / "experiments" / "runs")

    val_loader, test_loader = load_loaders()
    rn, dn = load_experts(device)
    logger.log(f"experts loaded; val={len(val_loader.dataset)}, "
               f"test={len(test_loader.dataset)} samples")

    # --- features ---
    X_val, y_val = base_features_stacked(rn, dn, val_loader, device, tta=False)
    X_valT, _ = base_features_stacked(rn, dn, val_loader, device, tta=True)
    X_test, y_test = base_features_stacked(rn, dn, test_loader, device, tta=False)
    X_testT, _ = base_features_stacked(rn, dn, test_loader, device, tta=True)
    # Fixed soft-voting baselines derive algebraically from the features that
    # base_features_stacked already computed — no re-running the experts.
    p_ens = 0.5 * (X_test[:, :10] + X_test[:, 10:])
    p_ensT = 0.5 * (X_testT[:, :10] + X_testT[:, 10:])
    logger.log(f"features: val {X_val.shape}, test {X_test.shape}")

    # --- train meta-models ---
    mlp_notta, val_acc_notta = train_mlp(
        X_val, y_val, X_val, y_val, device, epochs=args.epochs, seed=args.seed,
        logger=logger, tag="_notta")
    mlp_tta, val_acc_tta = train_mlp(
        X_valT, y_val, X_valT, y_val, device, epochs=args.epochs, seed=args.seed + 1,
        logger=logger, tag="_tta")
    logger.log(f"meta-model val acc: no-TTA {val_acc_notta:.2f}%, TTA {val_acc_tta:.2f}%")

    # --- test-set predictions (analysis artifact) ---
    with torch.no_grad():
        p_mlp = torch.softmax(
            mlp_notta(torch.tensor(X_test, dtype=torch.float32).to(device)),
            dim=1).cpu().numpy()
        p_mlpT = torch.softmax(
            mlp_tta(torch.tensor(X_testT, dtype=torch.float32).to(device)),
            dim=1).cpu().numpy()

    # --- persist ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "stacking_mlp_artifacts.npz",
        X_val=X_val, y_val=y_val, X_valT=X_valT,
        X_test=X_test, y_test=y_test, X_testT=X_testT,
        p_ens=p_ens, p_ensT=p_ensT, p_mlp=p_mlp, p_mlpT=p_mlpT,
    )
    torch.save(mlp_notta.state_dict(), OUT_DIR / "mlp_notta.pt")
    torch.save(mlp_tta.state_dict(), OUT_DIR / "mlp_tta.pt")
    run_config = {
        "run_name": "stacking_mlp",
        "meta_model": "MLP [20->64->10]",
        "epochs": args.epochs, "seed": args.seed, "lr": 1e-3,
        "features": "concat of SOTA expert softmax probs (20-dim), optional 2-view hflip TTA",
        "val_acc_notta": round(val_acc_notta, 2),
        "val_acc_tta": round(val_acc_tta, 2),
        "description": (
            "What: train a stacking meta-model over the SOTA ensemble's probability "
            "features. Why: test whether a learned combiner beats fixed 0.5/0.5 "
            "soft-voting. How: Adam lr=1e-3, 40 epochs, trained on the validation "
            "split only (leakage-free). When/Where: reproduced by this script; "
            "artifacts in experiments/results/stacking_mlp/. Who: LAB2 team."),
    }
    logger.write_config(run_config, extra_path=OUT_DIR / "config.json")
    logger.close()
    print(f"\nDone. val_acc (no TTA)={val_acc_notta:.2f}%  val_acc (TTA)={val_acc_tta:.2f}%")
    print(f"Artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
