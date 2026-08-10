"""
moe_router_train.py — Train the Phase-1 MoE soft router (script-only training).

Policy: notebooks never train.  This script runs the Phase-1 MoE pipeline:

    1. load the two SOTA experts (checkpoints from the training script);
    2. extract 10-dim expert probabilities on val/test;
    3. train the small soft-gated router (temperature + entropy regularizer +
       load-balancing loss) on the validation split;
    4. persist router weights, test probabilities and metrics for notebooks.

Artifacts (experiments/results/moe_phase1/):
    router_phase1.pt            # RouterMLP state dict
    moe_phase1_artifacts.npz    # p1_val, p2_val, y_val, p1_test, p2_test,
                                # y_test, p1_testT, p2_testT, g_test, g_testT
    config.json                 # 5W1H context + hyperparameters
    history.jsonl               # per-epoch val accuracy

Usage:
    python -m src.experiments.moe_router_train [--seed 0] [--epochs 40]

NOTE: Phase 2 (joint fine-tuning of experts + router) is intentionally not part
of the scripted deliverable; it did not beat Phase 1 in the experiment log
(see agents/experiments/MOE_EXPERIMENT.md).
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
from src.utils.feature_extraction import base_features_per_expert
from src.utils.run_logger import RunLogger

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "moe_phase1"
DATA_ROOT = str(PROJECT_ROOT / "data" / "raw")
SPLIT_FILE = str(PROJECT_ROOT / "data" / "processed" / "cifar10_split_seed42.json")


class RouterMLP(nn.Module):
    """[20 -> 64 -> ReLU -> 2] per-input gate over the two experts."""

    def __init__(self, in_dim: int = 20, hidden: int = 64, n_experts: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, n_experts))

    def forward(self, x):
        return self.net(x)


def load_loaders(batch_size: int = 64):
    """Val/test loaders (224x224, ImageNet normalization, fixed split)."""
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


def load_balance_loss(gates: torch.Tensor, n_experts: int = 2) -> torch.Tensor:
    """Empirical fraction routed to each expert x mean gate (collapse penalty)."""
    frac = gates.mean(0)
    return n_experts * (frac * gates.mean(0)).sum()


def train_router_soft(p1, p2, y, p1_val, p2_val, y_val, device, epochs=40,
                      lr=1e-3, alpha=0.1, gate_T=2.0, ent_w=0.1, seed=0,
                      logger=None):
    """Frozen-expert soft router (temperature + entropy reg + load balance)."""
    torch.manual_seed(seed)
    router = RouterMLP(in_dim=20).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    P1 = torch.tensor(p1, dtype=torch.float32)
    P2 = torch.tensor(p2, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.long)
    V1 = torch.tensor(p1_val, dtype=torch.float32).to(device)
    V2 = torch.tensor(p2_val, dtype=torch.float32).to(device)
    VY = torch.tensor(y_val, dtype=torch.long).to(device)
    best = 0.0
    for ep in range(epochs):
        idx = torch.randperm(len(P1), generator=torch.Generator().manual_seed(seed + ep))
        router.train()
        for b in range(0, len(P1), 256):
            bi = idx[b:b + 256]
            z = router(torch.cat([P1[bi], P2[bi]], 1).to(device)) / gate_T
            g = torch.softmax(z, 1)
            pmix = g[:, 0:1] * P1[bi].to(device) + g[:, 1:2] * P2[bi].to(device)
            ent = -(g * (g + 1e-9).log()).sum(1).mean()
            loss = crit(pmix, Y[bi].to(device)) + alpha * load_balance_loss(g) - ent_w * ent
            opt.zero_grad()
            loss.backward()
            opt.step()
        router.eval()
        with torch.no_grad():
            gv = torch.softmax(router(torch.cat([V1, V2], 1)) / gate_T, 1)
            pmixv = gv[:, 0:1] * V1 + gv[:, 1:2] * V2
            va = float((pmixv.argmax(1).cpu().numpy() == y_val).mean() * 100.0)
        best = max(best, va)
        if logger is not None:
            logger.append_metrics(ep + 1, {"val_acc": round(va, 2)})
    return router, best


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger = RunLogger("moe_router", runs_root=PROJECT_ROOT / "experiments" / "runs")

    val_loader, test_loader = load_loaders()
    rn, dn = load_experts(device)
    logger.log(f"experts loaded; val={len(val_loader.dataset)}, "
               f"test={len(test_loader.dataset)} samples")

    # --- probabilities (no-TTA primary, TTA for comparison) ---
    p1_val, p2_val, y_val = base_features_per_expert(rn, dn, val_loader, device, tta=False)
    p1_test, p2_test, y_test = base_features_per_expert(rn, dn, test_loader, device, tta=False)
    p1_testT, p2_testT, _ = base_features_per_expert(rn, dn, test_loader, device, tta=True)
    logger.log(f"features: val {p1_val.shape}, test {p1_test.shape}")

    # --- Phase 1: train soft router on validation split only ---
    router, best_val = train_router_soft(
        p1_val, p2_val, y_val, p1_val, p2_val, y_val, device,
        epochs=args.epochs, seed=args.seed, logger=logger)
    logger.log(f"Phase 1 router val acc: {best_val:.2f}%")

    # --- gate/test probabilities for notebook analysis ---
    router.eval()
    with torch.no_grad():
        g_test = torch.softmax(
            router(torch.tensor(np.concatenate([p1_test, p2_test], 1),
                                dtype=torch.float32).to(device)) / 2.0, 1).cpu().numpy()
        g_testT = torch.softmax(
            router(torch.tensor(np.concatenate([p1_testT, p2_testT], 1),
                                dtype=torch.float32).to(device)) / 2.0, 1).cpu().numpy()

    # --- persist ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "moe_phase1_artifacts.npz",
        p1_val=p1_val, p2_val=p2_val, y_val=y_val,
        p1_test=p1_test, p2_test=p2_test, y_test=y_test,
        p1_testT=p1_testT, p2_testT=p2_testT,
        g_test=g_test, g_testT=g_testT,
    )
    torch.save(router.state_dict(), OUT_DIR / "router_phase1.pt")
    run_config = {
        "run_name": "moe_phase1",
        "router": "MLP [20->64->2], softmax gate T=2.0",
        "epochs": args.epochs, "seed": args.seed, "lr": 1e-3,
        "regularizers": {"alpha": 0.1, "ent_w": 0.1, "gate_T": 2.0},
        "train_split": "validation only (leakage-free w.r.t. test)",
        "val_acc": round(best_val, 2),
        "description": (
            "What: train a per-input soft router over the two frozen SOTA experts "
            "(Phase 1 MoE). Why: test whether learned gating beats fixed 0.5/0.5 "
            "soft-voting. How: gate temperature 2.0 + entropy regularizer + "
            "load-balancing loss, Adam lr=1e-3, 40 epochs, validation split only. "
            "When/Where: reproduced by this script; artifacts in "
            "experiments/results/moe_phase1/. Who: LAB2 team."),
    }
    logger.write_config(run_config, extra_path=OUT_DIR / "config.json")
    logger.close()
    print(f"\nDone. Phase 1 router val acc = {best_val:.2f}%")
    print(f"Artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
