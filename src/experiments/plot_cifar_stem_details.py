"""
plot_cifar_stem_details.py — Comprehensive Plot Suite for CIFAR Stem Experiment.

Generates dedicated, high-resolution plots for CIFAR_STEM_EXPERIMENT.md comparing:
- ResNet18_224_Frozen
- ResNet18_224_Finetune
- ResNet18_32_NativeStem_Frozen
- ResNet18_32_NativeStem_Finetune

Saves PNG images into experiments/plots/.
"""

from __future__ import annotations

import os
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PLOT_DIR = "experiments/plots"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
})

def create_output_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)

# Dataset Specs from CIFAR_STEM_EXPERIMENT.md
variants = [
    "224x224 (Frozen)",
    "224x224 (Fine-tune)",
    "32x32 Native Stem (Frozen)",
    "32x32 Native Stem (Fine-tune)"
]
accuracies = [79.76, 91.73, 48.28, 72.93] # Test Acc (%)
losses = [0.5866, 0.2509, 1.4648, 0.7711]     # Test Loss
times_epoch = [73.97, 81.41, 42.71, 48.93]   # Time per Epoch (s)
total_times = [221.91, 244.22, 128.13, 146.80] # Total Time (3 epochs)
vram_mb = [620.2, 716.3, 532.8, 610.4]       # Peak VRAM (MB)
total_params = [11181642, 11181642, 11173962, 11173962]
trainable_params = [5130, 8398858, 6858, 8400586]

colors = ["#6baed6", "#1f77b4", "#fdae6b", "#e6550d"]

# -----------------------------------------------------------------------------
# 1. Accuracy Breakdown Plot
# -----------------------------------------------------------------------------
def plot_cifar_stem_accuracy():
    logger.info("Generating CIFAR Stem Plot 1: Accuracy Breakdown...")
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(variants, accuracies, color=colors, edgecolor="black", width=0.55)

    plt.ylabel("Test Classification Accuracy (%)", fontweight="bold")
    plt.title("CIFAR Stem Experiment: Test Accuracy Comparison across Resolution & Mode", fontsize=13, fontweight="bold")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1.5, f"{height:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Annotations
    plt.annotate("Spatial Feature Misalignment\n(Frozen 3x3 Conv1)", xy=(2, 48.28), xytext=(1.5, 25),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9, fontweight="bold", color="darkred")

    plt.annotate("Fine-tuning Recovers\nAccuracy (+24.65%)", xy=(3, 72.93), xytext=(2.7, 85),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9, fontweight="bold", color="darkgreen")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "cifar_stem_accuracy_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# 2. Test Loss Landscape Plot
# -----------------------------------------------------------------------------
def plot_cifar_stem_loss():
    logger.info("Generating CIFAR Stem Plot 2: Test Loss Landscape...")
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(variants, losses, color=colors, edgecolor="black", width=0.55)

    plt.ylabel("Cross-Entropy Loss (Lower is Better)", fontweight="bold")
    plt.title("CIFAR Stem Experiment: Test Loss Landscape Comparison", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.7)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.03, f"{height:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.annotate("High Loss Due to Broken\nPretrained Alignment", xy=(2, 1.4648), xytext=(1.2, 1.3),
                 arrowprops=dict(facecolor="crimson", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9, fontweight="bold", color="crimson")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "cifar_stem_loss_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# 3. Throughput & Speedup (FPS) Plot
# -----------------------------------------------------------------------------
def plot_cifar_stem_throughput():
    logger.info("Generating CIFAR Stem Plot 3: Throughput & Speedup...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Seconds per Epoch
    bars1 = axes[0].bar(variants, times_epoch, color=colors, edgecolor="black", width=0.55)
    axes[0].set_ylabel("Execution Time per Epoch (s)", fontweight="bold")
    axes[0].set_title("Training Latency per Epoch", fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 1.5, f"{height:.1f}s", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    axes[0].annotate("⚡ ~40% Speedup!", xy=(3, 48.93), xytext=(2.2, 70),
                     arrowprops=dict(facecolor="orange", shrink=0.08, width=1.5, headwidth=7),
                     fontsize=9.5, fontweight="bold", color="darkorange")

    # Panel B: Images per Second (FPS Throughput)
    # CIFAR-10 train set size = 50,000 images
    fps = [50000 / t for t in times_epoch]
    bars2 = axes[1].bar(variants, fps, color=colors, edgecolor="black", width=0.55)
    axes[1].set_ylabel("Throughput (Images / Sec - Higher is Better)", fontweight="bold")
    axes[1].set_title("Training Throughput (FPS)", fontweight="bold")
    axes[1].set_ylim(0, 1400)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 20, f"{height:.0f} FPS", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    axes[1].annotate("🚀 1.66x Higher Throughput", xy=(3, 1021.8), xytext=(2.0, 1200),
                     arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=7),
                     fontsize=9.5, fontweight="bold", color="darkgreen")

    plt.suptitle("CIFAR Stem Experiment: Execution Latency & Processing Throughput", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "cifar_stem_throughput_speedup.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# 4. Peak VRAM Memory Savings Plot
# -----------------------------------------------------------------------------
def plot_cifar_stem_vram():
    logger.info("Generating CIFAR Stem Plot 4: VRAM Memory Savings...")
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(variants, vram_mb, color=colors, edgecolor="black", width=0.55)

    plt.ylabel("Peak VRAM Allocated (MB - Lower is Better)", fontweight="bold")
    plt.title("CIFAR Stem Experiment: Peak GPU VRAM Memory Footprint", fontsize=13, fontweight="bold")
    plt.ylim(0, 850)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 12, f"{height:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.annotate("💾 105.9 MB (~15%) VRAM Saved\nNo 224x224 Upsampling Buffers!", xy=(3, 610.4), xytext=(1.8, 730),
                 arrowprops=dict(facecolor="royalblue", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9.5, fontweight="bold", color="darkblue")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "cifar_stem_vram_memory_footprint.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# 5. Parameter Breakdown Plot
# -----------------------------------------------------------------------------
def plot_cifar_stem_parameters():
    logger.info("Generating CIFAR Stem Plot 5: Parameter Breakdown...")
    plt.figure(figsize=(10, 5.5))

    x = np.arange(len(variants))
    width = 0.35

    t_params_m = [p / 1e6 for p in total_params]
    tr_params_m = [p / 1e6 for p in trainable_params]

    rects1 = plt.bar(x - width/2, t_params_m, width, label="Total Parameters (M)", color="#9ecae1", edgecolor="black")
    rects2 = plt.bar(x + width/2, tr_params_m, width, label="Trainable Parameters (M)", color="#3182bd", edgecolor="black")

    plt.ylabel("Parameters (Millions)", fontweight="bold")
    plt.title("CIFAR Stem Experiment: Total vs Trainable Parameter Count", fontsize=13, fontweight="bold")
    plt.xticks(x, variants)
    plt.ylim(0, 14)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in rects1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}M", ha="center", va="bottom", fontsize=8.5)
    for bar in rects2:
        yval = bar.get_height()
        label = f"{yval:.2f}M" if yval > 0.1 else f"{yval*1000:.1f}K"
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "cifar_stem_parameter_breakdown.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# 6. Accuracy vs Latency Scatter Tradeoff Plot
# -----------------------------------------------------------------------------
def plot_cifar_stem_scatter_tradeoff():
    logger.info("Generating CIFAR Stem Plot 6: Accuracy vs Latency Scatter Tradeoff...")
    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "*"]
    sizes = [150, 200, 150, 250]

    for var, lat, acc, vram, color, m, s in zip(variants, times_epoch, accuracies, vram_mb, colors, markers, sizes):
        plt.scatter(lat, acc, s=s, color=color, marker=m, label=f"{var} ({acc}%)", edgecolors="black", linewidth=1.5, zorder=5)
        plt.annotate(f"{var}\nAcc: {acc}% | {lat}s", (lat, acc), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    plt.xlabel("Latency per Epoch (Seconds - Lower is Better)", fontweight="bold")
    plt.ylabel("Test Classification Accuracy (% - Higher is Better)", fontweight="bold")
    plt.title("CIFAR Stem Tradeoff: Accuracy vs Training Latency", fontsize=13, fontweight="bold")
    plt.xlim(35, 90)
    plt.ylim(40, 100)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    plt.annotate("Optimal Edge Deployment\n(High Speed + High Acc)", xy=(48.93, 72.93), xytext=(55, 60),
                 arrowprops=dict(facecolor="orange", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9.5, fontweight="bold", color="darkorange")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "cifar_stem_tradeoff_scatter.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# 7. Master 4-Panel CIFAR Stem Infographic
# -----------------------------------------------------------------------------
def plot_cifar_stem_master_dashboard():
    logger.info("Generating CIFAR Stem Plot 7: Master 4-Panel Infographic...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0,0): Accuracy
    bars1 = axes[0, 0].bar(variants, accuracies, color=colors, edgecolor="black", width=0.55)
    axes[0, 0].set_title("A. Test Accuracy Comparison (%)", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars1:
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel (0,1): Test Loss
    bars2 = axes[0, 1].bar(variants, losses, color=colors, edgecolor="black", width=0.55)
    axes[0, 1].set_title("B. Test Loss Landscape (Cross-Entropy)", fontweight="bold")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_ylim(0, 1.7)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars2:
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel (1,0): Latency per Epoch
    bars3 = axes[1, 0].bar(variants, times_epoch, color=colors, edgecolor="black", width=0.55)
    axes[1, 0].set_title("C. Execution Latency per Epoch (Seconds)", fontweight="bold")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars3:
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel (1,1): Peak VRAM
    bars4 = axes[1, 1].bar(variants, vram_mb, color=colors, edgecolor="black", width=0.55)
    axes[1, 1].set_title("D. Peak GPU VRAM Footprint (MB)", fontweight="bold")
    axes[1, 1].set_ylabel("VRAM (MB)")
    axes[1, 1].set_ylim(0, 850)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars4:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{bar.get_height():.0f} MB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Rotate x labels slightly on subplots
    for ax in axes.flat:
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)

    plt.suptitle("CIFAR-10 Conv Stem Experiment: Native 32x32 Stem vs 224x224 Baseline Comprehensive Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "cifar_stem_master_dashboard.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

def generate_stem_plots():
    create_output_dir()
    plot_cifar_stem_accuracy()
    plot_cifar_stem_loss()
    plot_cifar_stem_throughput()
    plot_cifar_stem_vram()
    plot_cifar_stem_parameters()
    plot_cifar_stem_scatter_tradeoff()
    plot_cifar_stem_master_dashboard()
    logger.info("🎉 All CIFAR Stem experiment plots successfully generated!")

if __name__ == "__main__":
    generate_stem_plots()
