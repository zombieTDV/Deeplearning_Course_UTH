"""
train_lab2_models.py — CLI entry point for ALL LAB2 model training.

Policy (see agents/rules/LOGGING_CHECKPOINT_RULES.md)
------------------------------------------------------
Notebooks are restricted to testing, demos, unit tests, visualization and
analysis.  Training runs ONLY here, as a Python script, so that long runs can be
monitored in real time (console progress + optional TensorBoard) and resumed
exactly after an interruption (full-state checkpoints with optimizer, scheduler,
RNG states, history and config).

Trains the 6 deliverable variants:
    ResNet18/DenseNet121  x  {frozen, finetune, sota}

Usage:
    python -m src.training.train_lab2_models                          # all 6 variants, 20 epochs
    python -m src.training.train_lab2_models --modes frozen finetune  # subset
    python -m src.training.train_lab2_models --epochs 20 --seed 42    # explicit config
    python -m src.training.train_lab2_models --tb                     # + TensorBoard monitoring
    python -m src.training.train_lab2_models --resume                 # continue interrupted runs
    python -m src.training.train_lab2_models --smoke                  # 1 epoch, 2 batches (CI sanity)

Artifacts (per run, see LOGGING_CHECKPOINT_RULES.md):
    experiments/runs/<ts>_<run_name>/checkpoints/<run>_{best,last}.pt
    experiments/runs/<ts>_<run_name>/logs/<run>.log
    experiments/runs/<ts>_<run_name>/metrics/<run>_{config.json,history.jsonl}
    experiments/runs/registry.json          # run_name -> latest run dir
    experiments/results/training_history.json  # combined history for notebooks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import get_cifar10_loaders
from src.data.transforms import get_advanced_train_transform, get_eval_transform
from src.models.build_model import (
    build_densenet121,
    build_densenet121_full_sota,
    build_resnet18,
    build_resnet18_full_sota,
    get_densenet121_lrd_param_groups,
    get_resnet18_lrd_param_groups,
)
from src.training.train_model import train_model
from src.utils.checkpoint_utils import find_latest_run_dir
from src.utils.run_logger import RunLogger

DEFAULT_EPOCHS = 20
DEFAULT_SEED = 42
RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
REGISTRY_PATH = RUNS_ROOT / "registry.json"

# Mode -> (builder, base lr, label smoothing, use advanced augment loader, cosine sched)
VARIANTS = [
    ("ResNet18-frozen",    "frozen",   build_resnet18,         1e-3, 0.0, False, False),
    ("DenseNet121-frozen", "frozen",   build_densenet121,      1e-3, 0.0, False, False),
    ("ResNet18-finetune",  "finetune", build_resnet18,         1e-3, 0.0, False, False),
    ("DenseNet121-finetune", "finetune", build_densenet121,    1e-3, 0.0, False, False),
    ("ResNet18-sota",      "sota",     build_resnet18_full_sota, 3e-4, 0.1, True, True),
    ("DenseNet121-sota",   "sota",     build_densenet121_full_sota, 3e-4, 0.1, True, True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all LAB2 model variants (script-only training).")
    p.add_argument("--modes", nargs="+", default=["frozen", "finetune", "sota"],
                   choices=["frozen", "finetune", "sota"],
                   help="Which modes to train (default: all three).")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--resume", action="store_true",
                   help="Continue every selected variant from its latest *_last.pt checkpoint. "
                        "If a run already early-stopped, resume halts (the run is complete).")
    p.add_argument("--force-resume", action="store_true",
                   help="Resume from the best epoch (*_best.pt) with a fresh early-stopping "
                        "budget, overriding a completed early stop (rewind + continue).")
    p.add_argument("--tb", action="store_true",
                   help="Enable TensorBoard writers (lightweight console/file logging is always on).")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Cap batches per epoch (0 = unlimited). For smoke tests / CI.")
    p.add_argument("--smoke", action="store_true",
                   help="Quick sanity: 1 epoch, 2 batches, frozen mode only.")
    p.add_argument("--device", default=None, help="torch device override (default: auto).")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: nn.Module, mode: str, lr: float, weight_decay: float):
    """Replicates the notebook's optimizer construction (frozen/finetune/sota)."""
    if mode == "sota":
        if model.__class__.__name__ == "ResNet":
            groups = get_resnet18_lrd_param_groups(model, base_lr=lr, weight_decay=weight_decay)
        else:
            groups = get_densenet121_lrd_param_groups(model, base_lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(groups)
    if mode == "finetune":
        backbone, head = [], []
        for n, param in model.named_parameters():
            if param.requires_grad:
                if "classifier" in n or n in ("fc.weight", "fc.bias"):
                    head.append(param)
                else:
                    backbone.append(param)
        return torch.optim.AdamW([
            {"params": backbone, "lr": lr * 0.1},
            {"params": head, "lr": lr},
        ], weight_decay=weight_decay)
    return torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )


def _update_registry(run_name: str, run_dir: str) -> None:
    """Persist run_name -> latest run dir so notebooks can load artifacts."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}
    registry[run_name] = str(Path(run_dir).relative_to(PROJECT_ROOT))
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def _write_combined_history(all_results: dict[str, dict]) -> None:
    """Merge this run's histories into experiments/results/training_history.json.

    Existing entries for other variants are preserved (subset runs must not
    erase previously validated history). Written atomically.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR / "training_history.json"
    combined = {}
    if target.exists():
        try:
            combined = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            combined = {}
    for name, res in all_results.items():
        combined[name] = {
            "train_losses": [round(v, 4) for v in res["train_losses"]],
            "val_losses": [round(v, 4) for v in res["val_losses"]],
            "val_accs": [round(v, 2) for v in res["val_accs"]],
            "best_epoch": res["best_epoch"],
            "best_val_acc": round(res["best_val_acc"], 2),
            "checkpoint": res["best_state_path"],
        }
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs, args.max_batches, args.modes = 1, 2, ["frozen"]
        # Smoke runs go to a dedicated root so they never pollute the real
        # registry or the artifact search (find_best_checkpoint skips "_" dirs).
        global RUNS_ROOT
        RUNS_ROOT = RUNS_ROOT / "_smoke"

    _set_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available()
                           else ("mps" if torch.backends.mps.is_available() else "cpu")))
    print(f"Device: {device} | seed: {args.seed} | epochs: {args.epochs} | modes: {args.modes}")

    # ---- Data loaders (shared across variants of the same transform family) ----
    train_loader, val_loader, _ = get_cifar10_loaders(batch_size=64, num_workers=0)
    sota_train_loader, _, _ = get_cifar10_loaders(
        train_transform=get_advanced_train_transform(resize_size=224,
                                                     use_randaugment=True,
                                                     use_random_erasing=True),
        eval_transform=get_eval_transform(resize_size=224),
        batch_size=64, num_workers=0,
    )

    all_results: dict[str, dict] = {}
    for run_name, mode, builder, lr, smoothing, use_adv, use_cosine in VARIANTS:
        if mode not in args.modes:
            continue

        print(f"\n===== {run_name} ({mode.upper()}) =====")
        logger = RunLogger(run_name, runs_root=RUNS_ROOT)
        run_config = {
            "run_name": run_name,
            "mode": mode,
            "architecture": "ResNet18" if "ResNet" in run_name else "DenseNet121",
            "num_epochs": args.epochs,
            "seed": args.seed,
            "batch_size": 64,
            "num_workers": 0,
            "lr": lr,
            "weight_decay": 1e-4,
            "label_smoothing": smoothing,
            "advanced_augmentation": use_adv,   # RandAugment + RandomErasing
            "scheduler": "CosineAnnealingLR" if use_cosine else None,
            "early_stopping": {"patience": 4, "min_delta": 1e-4},
            "dataset": "CIFAR-10 (data/raw), fixed split seed=42 (45k/5k/10k)",
            "description": (  # 5W1H: what/why/when/where/who/how
                f"What: train {run_name} on CIFAR-10. "
                f"Why: LAB2 deliverable — pretrained transfer learning with "
                f"{'feature extraction' if mode == 'frozen' else ('head+tail finetune' if mode == 'finetune' else 'deep-feature LLRD SOTA')}. "
                f"How: AdamW (lr={lr}), CE{' (label smoothing 0.1)' if smoothing else ''}, "
                f"{'RandAugment+Erasing' if use_adv else 'standard ImageNet transforms'}, "
                f"early stopping patience 4. "
                f"Where: experiments/runs (checkpoints/logs/metrics). "
                f"Who: LAB2 team. When: reproducibility run — full state logged."),
        }

        model = builder(num_classes=10, device=device)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        optimizer = _build_optimizer(model, mode, lr, weight_decay=1e-4)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6) if use_cosine else None)
        writer = logger.tb_writer if args.tb else None

        res = train_model(
            model, (sota_train_loader if use_adv else train_loader), val_loader,
            criterion, optimizer, device, num_epochs=args.epochs, run_name=run_name,
            scheduler=scheduler, writer=writer, save_dir=str(logger.checkpoint_dir),
            early_stopping=True, patience=4, min_delta=1e-4,
            logger=logger,
            resume_from=(find_latest_run_dir(run_name, runs_root=RUNS_ROOT)
                         if (args.resume or args.force_resume) else None),
            resume_from_best=args.force_resume,
            seed=args.seed, config=run_config, progress_every=1,
            max_batches_per_epoch=args.max_batches,
        )
        if writer is not None:
            writer.close()
        logger.close()

        if not args.smoke:
            _update_registry(run_name, str(logger.run_dir))
        all_results[run_name] = res
        print(f"  -> best val acc {res['best_val_acc']:.2f}% @ epoch {res['best_epoch']} "
              f"(checkpoints in {logger.run_dir})")

    if not args.smoke:
        _write_combined_history(all_results)
        print("\nAll requested runs finished. Combined history -> experiments/results/training_history.json")
        print("Registry (run_name -> latest run dir) -> experiments/runs/registry.json")
    else:
        print("\nSmoke run finished (no registry / results updated). "
              f"Runs under {RUNS_ROOT}")


if __name__ == "__main__":
    main()
