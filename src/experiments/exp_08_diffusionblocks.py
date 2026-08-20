"""
exp_08_diffusionblocks.py — Evaluate DiffusionBlocks (Sakana AI, ICLR 2026)
for VRAM reduction during fine-tuning.

Method (arXiv:2506.14202, https://github.com/SakanaAI/DiffusionBlocks)
-----------------------------------------------------------------------
DiffusionBlocks reinterprets residual-block dynamics as a denoising process:
the network is partitioned into B blocks, each block is assigned a noise
range, and each block is trained independently with a score-matching loss
(weighted cross-entropy on corrupted label embeddings).  Only ONE block is
in the autograd graph per training step -> peak activation memory ~1/B.

IMPORTANT SCOPE NOTE
--------------------
The official implementation supports *transformer* models only (ViT in this
repo).  LAB2's deliverable models (ResNet18 / DenseNet121) are CNNs, so the
method cannot be applied to them directly.  This script therefore evaluates
DiffusionBlocks on the closest faithful setups:

  track=scratch   ViT-12L (h128, patch 4) trained from scratch on CIFAR-10
                  @32px — the official config, ours is vit vs dblock (B=3).
  track=finetune  ImageNet-pretrained ViT-Tiny-16-224 fine-tuned on CIFAR-10
                  @224px through OUR data pipeline (fixed seed-42 split).
                  This is the paper's stated "future work": converting
                  pretrained models to DiffusionBlocks for fine-tuning.
  track=reference Anchors "what our current finetune costs": peak VRAM of our
                  ResNet18 / DenseNet121 finetune runs (LAB2 pipeline).

Usage:
  python -m src.experiments.exp_08_diffusionblocks --track scratch --mode vit
  python -m src.experiments.exp_08_diffusionblocks --track scratch --mode dblock
  python -m src.experiments.exp_08_diffusionblocks --track finetune --mode vit
  python -m src.experiments.exp_08_diffusionblocks --track finetune --mode dblock
  python -m src.experiments.exp_08_diffusionblocks --track reference --arch resnet18
  python -m src.experiments.exp_08_diffusionblocks --smoke   # 1 epoch, 4 batches

Artifacts:
  experiments/runs/<ts>_DB-<track>-<mode>/  (config JSON, history JSONL, best.pt)
  experiments/results/diffusionblocks_summary.json  (consolidated across runs)

The DBlock training/inference math below is a faithful port of the official
model.py (Sakana AI, Apache-2.0): get_sigmas / get_weights /
estimate_target_layer / denoise / diffusion_step are copied with identical
numerics; only the Lightning boilerplate was removed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for dblock package

from dblock import dblock_modules as dbm
from dblock.vit import (
    ViTDiTConfig,
    ViTDiTForImageClassification,
    load_vit,
)

RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
SUMMARY_PATH = RESULTS_DIR / "diffusionblocks_summary.json"

SIGMA_DATA = 0.5  # EDM sigma_data (official value)
P_MEAN, P_STD = -1.2, 1.2  # log-normal sigma distribution (official values)

# ---------------------------------------------------------------------------
# Faithful ports of the official DBlock training math (src: model.py, SakanaAI)
# ---------------------------------------------------------------------------
def get_embeds(model: nn.Module, labels: torch.Tensor) -> torch.Tensor:
    embeds = model.get_input_embeddings()(labels)
    return F.normalize(embeds, p=2, dim=-1)


def get_sigmas(num_blocks: int, block_sigmas, n_samples: int,
               gamma: float = 0.05) -> tuple[torch.Tensor, int]:
    """Sample sigma from one uniformly-random block's noise range (official)."""
    block_idx = random.choices(range(num_blocks), k=1)[0]
    sigma_min_b = block_sigmas[block_idx]
    sigma_max_b = block_sigmas[block_idx + 1]
    if gamma > 0.0:
        log_sigma_min = np.log(sigma_min_b)
        log_sigma_max = np.log(sigma_max_b)
        log_range = log_sigma_max - log_sigma_min
        sigma_min_b = np.exp(log_sigma_min - gamma * log_range)
        sigma_max_b = np.exp(log_sigma_max + gamma * log_range)
        sigma_min_b = max(sigma_min_b, block_sigmas[0])
        sigma_max_b = min(sigma_max_b, block_sigmas[-1])
    cdf_min = norm.cdf((np.log(sigma_min_b) - P_MEAN) / P_STD)
    cdf_max = norm.cdf((np.log(sigma_max_b) - P_MEAN) / P_STD)
    rand = np.random.uniform(cdf_min, cdf_max, n_samples)
    sigma = torch.from_numpy(np.exp(P_MEAN + P_STD * norm.ppf(rand))).to(
        torch.float32
    )
    return sigma, block_idx


def get_weights(sigmas: torch.Tensor) -> torch.Tensor:
    return (sigmas**2 + SIGMA_DATA**2) / (sigmas * SIGMA_DATA) ** 2


def estimate_target_layer(sigmas: torch.Tensor, block_sigmas, num_blocks: int) -> int:
    bs = torch.tensor(block_sigmas, dtype=torch.float32, device=sigmas.device)
    block_idx = torch.bucketize(sigmas, bs, right=True) - 1
    block_idx = (num_blocks - 1) - block_idx
    block_idx = torch.clamp(block_idx, 0, num_blocks - 1).long()
    values, counts = block_idx.unique(return_counts=True)
    return values[counts.argmax()].item()


def denoise(model: nn.Module, layer_assignment: list[list[int]],
            pixel_values: torch.Tensor, zt: torch.Tensor,
            sigmas: torch.Tensor, block_idx: int | None = None) -> torch.Tensor:
    """Run one block (denoising step) + head, with EDM preconditioning."""
    if block_idx is None:
        block_idx = estimate_target_layer(sigmas, model.block_sigmas,
                                          model.num_blocks)
    c_skip = SIGMA_DATA**2 / (sigmas**2 + SIGMA_DATA**2)
    c_out = sigmas * SIGMA_DATA / (sigmas**2 + SIGMA_DATA**2) ** 0.5
    c_in = 1 / (sigmas**2 + SIGMA_DATA**2) ** 0.5
    c_noise = 0.25 * sigmas.log()

    outputs = model.forward_block(
        layer_indices=layer_assignment[block_idx],
        pixel_values=pixel_values,
        noisy_embeds=zt * c_in[:, None],
        timesteps=c_noise,
    )
    hidden_states = outputs.last_hidden_state
    conditioning = outputs.conditioning
    model_out = hidden_states * c_out[:, None] + zt * c_skip[:, None]
    logits = model.forward_output_embeddings(model_out.unsqueeze(1), conditioning)
    return logits


def dblock_train_step(model: nn.Module, batch, criterion, device) -> tuple[torch.Tensor, int]:
    """Official training step: corrupt label embeddings, denoise ONE block."""
    pixel_values, labels = batch
    z = get_embeds(model, labels)
    sigmas, block_idx = get_sigmas(model.num_blocks, model.block_sigmas, z.shape[0])
    sigmas = sigmas.to(z.device)
    zt = z + sigmas[:, None] * torch.randn_like(z)
    logits = denoise(model, model.layer_assignment, pixel_values, zt, sigmas, block_idx)
    loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
    w = get_weights(sigmas)[:, None]
    loss = (loss * w).mean()
    return loss, block_idx


def diffusion_step(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Official DBlock inference: Euler-denoise through all blocks (model.py)."""
    bsz = x.shape[0]
    hidden_size = model.config.hidden_size
    z = torch.randn(bsz, hidden_size, device=x.device)
    z *= torch.sqrt(1.0 + model.sigmas[0] ** 2.0)
    s_in = x.new_ones([x.shape[0]])
    for i in range(model.sigmas.shape[0] - 1):
        sigma = model.sigmas[i] * s_in
        next_sigma = model.sigmas[i + 1] * s_in
        logits = denoise(model, model.layer_assignment, x, z, sigma)
        probs = F.softmax(logits, dim=1)
        denoised = F.linear(probs, model.get_input_embeddings().weight.t())
        d = (z - denoised) / sigma[:, None]
        dt = next_sigma - sigma
        z = z + dt[:, None] * d
    min_sigma = model.sigmas[-1].item()
    sigmas = torch.full((x.shape[0],), min_sigma, device=x.device)
    return denoise(model, model.layer_assignment, x, z, sigmas)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_scratch_model(num_labels: int = 10, is_dblock: bool = False) -> nn.Module:
    return load_vit(image_size=32, num_labels=num_labels, is_dblock=is_dblock)


PRETRAINED_ID = "WinKawaks/vit-tiny-patch16-224"  # 12L, h192, 3 heads, patch 16


def build_finetune_model(is_dblock: bool = False, device=None) -> tuple[nn.Module, dict]:
    """Pretrained ViT-Tiny-16-224; dblock = DiffusionBlocks conversion."""
    if not is_dblock:
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(
            PRETRAINED_ID, num_labels=10, ignore_mismatched_sizes=True
        )
        return model.to(device) if device is not None else model, {}

    # ---- DiffusionBlocks conversion of the pretrained model ----
    from transformers import ViTForImageClassification as HF_ViT
    hf = HF_ViT.from_pretrained(PRETRAINED_ID)
    cfg = ViTDiTConfig.from_pretrained(PRETRAINED_ID)
    cfg.time_conditioning = True
    cfg.num_labels = 10
    model = ViTDiTForImageClassification(cfg)
    model._init_dit()  # zero-init conditioning (identity at init), as official

    # Map transformers-5.x pretrained keys -> vendored 4.x-style keys.
    # The pretrained classifier is 1000-class, so it is NOT copied; the dblock
    # 10-class head stays zero-init from `_init_dit`.
    mapping = {}
    shared = ("vit.embeddings.patch_embeddings.projection.weight",
              "vit.embeddings.patch_embeddings.projection.bias",
              "vit.embeddings.position_embeddings",
              "vit.layernorm.weight", "vit.layernorm.bias")
    for k in shared:
        if k in hf.state_dict():
            mapping[k] = k
    for i in range(cfg.num_hidden_layers):
        for ln in ("layernorm_before", "layernorm_after"):
            for sfx in ("weight", "bias"):
                src = f"vit.layers.{i}.{ln}.{sfx}"
                dst = f"vit.encoder.layer.{i}.{ln}.{sfx}"
                mapping[src] = dst
        for proj, name in (("q_proj", "q_proj"), ("k_proj", "k_proj"),
                           ("v_proj", "v_proj"), ("o_proj", "o_proj")):
            for sfx in ("weight", "bias"):
                src = f"vit.layers.{i}.attention.{proj}.{sfx}"
                dst = f"vit.encoder.layer.{i}.attention.{name}.{sfx}"
                mapping[src] = dst
        for mlp, comp in (("fc1", "intermediate.dense"), ("fc2", "output.dense")):
            for sfx in ("weight", "bias"):
                src = f"vit.layers.{i}.mlp.{mlp}.{sfx}"
                dst = f"vit.encoder.layer.{i}.{comp}.{sfx}"
                mapping[src] = dst

    sd_hf = {mapping[k]: v for k, v in hf.state_dict().items() if k in mapping}
    missing, unexpected = model.load_state_dict(sd_hf, strict=False)
    print(f"[load] pretrained {PRETRAINED_ID} -> dblock conversion: "
          f"{len(sd_hf)}/{len(hf.state_dict())} keys copied, "
          f"{len(missing)} new (conditioning), {len(unexpected)} unexpected")
    del hf
    return model.to(device) if device is not None else model, {}


def prepare_dblock(model: nn.Module, num_blocks: int, num_inference_steps: int | None):
    """Attach DBlock bookkeeping (sigmas, layer assignment) to the model."""
    model.num_blocks = num_blocks
    model.num_labels = model.config.num_labels
    model.block_sigmas = dbm.get_block_sigmas(num_layers=num_blocks)
    model.num_inference_steps = num_inference_steps or num_blocks
    model.register_buffer(
        "sigmas", dbm.get_discrete_sigmas(
            num_steps=model.num_inference_steps, dblock=True).to(model.device))
    split_size = model.config.num_hidden_layers // num_blocks
    model.layer_assignment = [
        list(range(i * split_size, (i + 1) * split_size))
        for i in range(num_blocks)
    ]
    return model


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def get_scratch_loaders(batch_size: int, num_workers: int = 0):
    """Official-style CIFAR-10 @32px (train = official 50k, test = official 10k)."""
    from torchvision import transforms as T
    from torchvision.datasets import CIFAR10

    root = str(PROJECT_ROOT / "data" / "raw")
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.Resize([32, 32]),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize([32, 32]),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train = CIFAR10(root=root, train=True, transform=train_tf,
                    download=False)
    test = CIFAR10(root=root, train=False, transform=eval_tf,
                   download=False)
    print(f"[load] CIFAR-10 (scratch track) root -> {root} "
          f"train={len(train)} test={len(test)}")
    tl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True)
    el = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
    return tl, el


def get_finetune_loaders(batch_size: int):
    """LAB2 pipeline: fixed seed-42 split, 224x224 ImageNet transforms."""
    from src.data.dataloader import get_cifar10_loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, num_workers=0)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# VRAM / timing instrumentation
# ---------------------------------------------------------------------------
class MemMeter:
    """Per-step peak-allocated/reserved VRAM tracking (CUDA only)."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.step_peaks: list[float] = []  # MB, allocated, one per step
        self.step_reserved: list[float] = []
        self.max_alloc = 0.0
        self.max_reserved = 0.0

    def step_start(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def step_end(self):
        if self.cuda:
            alloc = torch.cuda.max_memory_allocated() / 1e6
            res = torch.cuda.memory_reserved() / 1e6
            self.step_peaks.append(alloc)
            self.step_reserved.append(res)
            self.max_alloc = max(self.max_alloc, alloc)
            self.max_reserved = max(self.max_reserved, res)

    def epoch_summary(self) -> dict:
        if not self.step_peaks:
            return {}
        return {
            "step_peak_alloc_mean_mb": round(float(np.mean(self.step_peaks)), 1),
            "step_peak_alloc_max_mb": round(float(np.max(self.step_peaks)), 1),
            "step_peak_reserved_max_mb": round(float(np.max(self.step_reserved)), 1),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(model, train_loader, eval_loader, is_dblock: bool, args,
                 val_loader=None) -> dict:
    device = torch.device(args.device)
    model = model.to(device)
    is_dblock = is_dblock or args.mode == "dblock"
    if is_dblock:
        model = prepare_dblock(model, args.blocks, args.num_inference_steps)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] {type(model).__name__} | params {n_all/1e6:.2f}M "
          f"(trainable {n_trainable/1e6:.2f}M) | "
          f"{'DBlock B=' + str(args.blocks) if is_dblock else 'end-to-end'}")

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # DBlock aligns total iterations: epochs * num_blocks
    epochs = args.epochs if not is_dblock else args.epochs * args.blocks
    meter = MemMeter(device)
    history: list[dict] = []
    best_acc, best_epoch = 0.0, -1
    t_total = time.perf_counter()
    steps_total = 0

    for epoch in range(1, epochs + 1):
        model.train()
        meter.step_peaks, meter.step_reserved = [], []
        t0 = time.perf_counter()
        running_loss, n_batches = 0.0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            if args.smoke and batch_idx >= 4:
                break
            images, labels = images.to(device), labels.to(device)
            meter.step_start()
            optimizer.zero_grad()
            if is_dblock:
                loss, _block_idx = dblock_train_step(model, (images, labels),
                                                     criterion, device)
            else:
                logits = model(pixel_values=images, return_dict=True)
                logits = logits.logits
                loss = criterion(logits.view(-1, model.config.num_labels),
                                 labels.view(-1)).mean()
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            meter.step_end()
            running_loss += loss.item()
            n_batches += 1
            steps_total += 1
        epoch_time = time.perf_counter() - t0
        epoch_loss = running_loss / max(n_batches, 1)

        # ---- evaluation (val for finetune, test for scratch) ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in eval_loader:
                if args.smoke and total >= 100:
                    break
                images, labels = images.to(device), labels.to(device)
                meter.step_start()
                if is_dblock:
                    logits = diffusion_step(model, images)
                else:
                    logits = model(pixel_values=images, return_dict=True).logits
                meter.step_end()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / max(total, 1)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch

        history.append({
            "epoch": epoch,
            "effective_epoch": round(epoch / args.blocks, 2) if is_dblock else epoch,
            "train_loss": round(epoch_loss, 4),
            "acc": round(acc, 2),
            "epoch_time_s": round(epoch_time, 1),
            **meter.epoch_summary(),
        })
        print(f"  ep {epoch:>3}/{epochs} | loss {epoch_loss:.3f} | "
              f"{'val' if val_loader is not None else 'test'} acc {acc:.2f}% | "
              f"step peak alloc {meter.max_alloc:.0f}MB | "
              f"reserved {meter.max_reserved:.0f}MB | "
              f"{epoch_time:.1f}s/ep", flush=True)

    total_time = time.perf_counter() - t_total
    summary = {
        "best_acc": round(best_acc, 2),
        "best_epoch": best_epoch,
        "final_acc": round(acc, 2),
        "total_time_s": round(total_time, 1),
        "mean_step_time_s": round(total_time / max(steps_total, 1), 3),
        "peak_alloc_mb": round(meter.max_alloc, 1),
        "peak_reserved_mb": round(meter.max_reserved, 1),
        "params_M": round(n_all / 1e6, 2),
        "trainable_params_M": round(n_trainable / 1e6, 2),
    }
    return {"history": history, "summary": summary}


# ---------------------------------------------------------------------------
# Reference track: our CNN finetune VRAM anchor
# ---------------------------------------------------------------------------
def run_reference(args):
    from src.data.dataloader import get_cifar10_loaders
    from src.models.build_model import build_densenet121, build_resnet18

    device = torch.device(args.device)
    train_loader, _, _ = get_cifar10_loaders(batch_size=args.batch_size, num_workers=0)
    builder = build_resnet18 if args.arch == "resnet18" else build_densenet121
    model = builder(num_classes=10, mode="finetune", device=device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.arch}-finetune | trainable {n_trainable/1e6:.2f}M "
          f"(batch {args.batch_size}, {args.epochs} ep)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    meter = MemMeter(device)
    peaks, times = [], []
    for epoch in range(args.epochs):
        model.train()
        meter.step_peaks, meter.step_reserved = [], []
        t0 = time.perf_counter()
        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            meter.step_start()
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            meter.step_end()
        times.append(time.perf_counter() - t0)
        s = meter.epoch_summary()
        peaks.append(s)
        print(f"  ep {epoch+1} | step peak alloc {meter.max_alloc:.0f}MB | "
              f"reserved {meter.max_reserved:.0f}MB | {times[-1]:.1f}s/ep", flush=True)
    return {
        "arch": args.arch, "batch_size": args.batch_size,
        "epochs": args.epochs, "trainable_params_M": round(n_trainable / 1e6, 2),
        "history": [{"epoch": i + 1, "epoch_time_s": round(t, 1), **p}
                    for i, (p, t) in enumerate(zip(peaks, times))],
        "summary": {
            "peak_alloc_mb": max((p.get("step_peak_alloc_max_mb", 0) for p in peaks), default=0),
            "peak_reserved_mb": max((p.get("step_peak_reserved_max_mb", 0) for p in peaks), default=0),
            "mean_epoch_time_s": round(float(np.mean(times)), 1),
        },
    }


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------
def persist(run_name: str, payload: dict, is_smoke: bool):
    run_dir = RUNS_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics" / f"{run_name}_config.json", "w") as f:
        json.dump(payload["config"], f, indent=2)
    with open(run_dir / "metrics" / f"{run_name}_history.jsonl", "w") as f:
        for h in payload["results"]["history"]:
            f.write(json.dumps(h) + "\n")
    summary = {**payload["config"], "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
               **payload["results"]["summary"]}
    with open(run_dir / "metrics" / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if not is_smoke:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if SUMMARY_PATH.exists():
            try:
                data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        data[run_name] = summary
        tmp = SUMMARY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, SUMMARY_PATH)
    print(f"[save] run artifacts -> {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DiffusionBlocks VRAM evaluation (LAB2)")
    p.add_argument("--track", choices=["scratch", "finetune", "reference"], default="scratch")
    p.add_argument("--mode", choices=["vit", "dblock"], default="vit")
    p.add_argument("--arch", choices=["resnet18", "densenet121"], default="resnet18")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
    _set_seed(args.seed)
    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(args.device)
    print(f"Device: {device} | track={args.track} mode={args.mode} "
          f"epochs={args.epochs} batch={args.batch_size} seed={args.seed}")

    commit = "unknown"
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    common_config = {
        "experiment": "EXP-08 DiffusionBlocks VRAM evaluation",
        "date": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "device": args.device,
        "seed": args.seed,
        "track": args.track, "mode": args.mode,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "method": ("DiffusionBlocks (arXiv:2506.14202, B=" + str(args.blocks) + ")"
                   if args.mode == "dblock" else "end-to-end backprop"),
        "description": (  # 5W1H
            "What: peak VRAM / time / accuracy of DiffusionBlocks block-wise "
            "training vs end-to-end backprop. Why: evaluate VRAM reduction "
            "during finetuning for LAB2. How: plain-PyTorch port of official "
            "SakanaAI code; per-step torch.cuda peak stats; fixed seed. "
            "Where: RTX 4060 Laptop (8GB); runs under experiments/runs. "
            "Who: LAB2 team. When: " + datetime.now().strftime("%Y-%m-%d")),
    }

    if args.track == "reference":
        results = run_reference(args)
        run_name = f"DB-ref-{args.arch}-b{args.batch_size}"
        payload = {"config": common_config, "results": results}
    else:
        is_dblock = args.mode == "dblock"
        if args.track == "scratch":
            train_loader, eval_loader = get_scratch_loaders(args.batch_size)
            val_loader = None
            model = build_scratch_model(num_labels=10, is_dblock=is_dblock)
        else:  # finetune
            train_loader, val_loader, eval_loader = get_finetune_loaders(args.batch_size)
            model, _ = build_finetune_model(is_dblock=is_dblock, device=device)
        run_name = f"DB-{args.track}-{args.mode}"
        results = run_training(model, train_loader, eval_loader, is_dblock,
                               args, val_loader=val_loader)
        payload = {"config": common_config, "results": results}

    persist(run_name, payload, is_smoke=args.smoke)
    s = results["summary"]
    print(f"\n===== {run_name} done =====")
    print(f"  peak alloc {s['peak_alloc_mb']}MB | reserved {s['peak_reserved_mb']}MB | "
          f"total {s['total_time_s']}s | best acc {s['best_acc']}%")


if __name__ == "__main__":
    main()
