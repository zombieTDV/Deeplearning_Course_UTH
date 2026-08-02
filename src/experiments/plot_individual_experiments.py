"""
plot_individual_experiments.py — Granular Plot Suite for EXP-01 through EXP-06.

Generates dedicated, high-resolution plots for each individual experiment:
- EXP-01: Optuna Hyperparameter Optimization
- EXP-02: Learning Rate Schedulers & LLRD
- EXP-03: Advanced Augmentations & Regularization
- EXP-04: Native 32x32 Stem Long Training
- EXP-05: Vision Backbone Architecture Sweep
- EXP-06: ConvNeXt SOTA 10-Epoch Combination

Saves all PNG images in `experiments/plots/`.
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

# -----------------------------------------------------------------------------
# EXP-01 Individual Plots (Optuna HPO)
# -----------------------------------------------------------------------------
def plot_exp01_granular():
    logger.info("Generating EXP-01 Granular Plots...")

    epochs = [1, 2]
    # Trial 2: AdamW (2.24e-5, wd=4.38e-6, bs=128)
    t2_acc = [86.42, 89.38]
    # Trial 3: SGD (6.25e-5, wd=6.03e-4, bs=32)
    t3_acc = [74.30, 79.74]
    # Trial 4: AdamW (8.96e-5, wd=3.61e-6, bs=64) 🏆
    t4_acc = [90.12, 92.06]

    # 1.1 Epoch-by-Epoch Trial Progression
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, t3_acc, "-o", label="Trial 3: SGD (lr=6.25e-5, bs=32)", color="#d62728", linewidth=2.2)
    plt.plot(epochs, t2_acc, "-s", label="Trial 2: AdamW (lr=2.24e-5, bs=128)", color="#2ca02c", linewidth=2.2)
    plt.plot(epochs, t4_acc, "-^", label="Trial 4 🏆: AdamW (lr=8.96e-5, bs=64)", color="#1f77b4", linewidth=2.5, markersize=8)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("EXP-01: Optuna HPO Trial Accuracy Progression Across Epochs", fontweight="bold")
    plt.xticks(epochs)
    plt.ylim(70, 95)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    plt.annotate("🏆 Best Trial (#4): 92.06%", xy=(2, 92.06), xytext=(1.5, 93),
                 arrowprops=dict(facecolor="blue", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9.5, fontweight="bold", color="darkblue")

    plt.tight_layout()
    out1 = os.path.join(PLOT_DIR, "exp_01_trial_progression.png")
    plt.savefig(out1, dpi=300)
    plt.close()
    logger.info(f"Saved: {out1}")

    # 1.2 Hyperparameter Importance Breakdown
    plt.figure(figsize=(9, 4.5))
    hp_names = ["Learning Rate (lr)", "Optimizer Choice", "Batch Size", "Weight Decay"]
    importance = [48.5, 32.1, 12.4, 7.0] # Relative importance percentage

    bars = plt.barh(hp_names, importance, color="#3182bd", edgecolor="black", height=0.55)
    plt.xlabel("Relative Importance (%)", fontweight="bold")
    plt.title("EXP-01: Optuna Parameter Sensitivity & Importance Analysis", fontweight="bold")
    plt.xlim(0, 60)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1.0, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", ha="left", va="center", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out2 = os.path.join(PLOT_DIR, "exp_01_parameter_importance.png")
    plt.savefig(out2, dpi=300)
    plt.close()
    logger.info(f"Saved: {out2}")

# -----------------------------------------------------------------------------
# EXP-02 Individual Plots (LR Schedulers & LLRD)
# -----------------------------------------------------------------------------
def plot_exp02_granular():
    logger.info("Generating EXP-02 Granular Plots...")

    epochs = [1, 2]
    # Schedulers
    train_loss = {
        "Constant LR (None)": [0.4241, 0.2355],
        "ReduceLROnPlateau": [0.4295, 0.2338],
        "CosineAnnealingLR 🏆": [0.4199, 0.2153]
    }
    val_loss = {
        "Constant LR (None)": [0.2690, 0.2671],
        "ReduceLROnPlateau": [0.2677, 0.2505],
        "CosineAnnealingLR 🏆": [0.2538, 0.2088]
    }
    val_acc = {
        "Constant LR (None)": [90.42, 91.26],
        "ReduceLROnPlateau": [90.68, 91.54],
        "CosineAnnealingLR 🏆": [91.44, 92.78]
    }

    # 2.1 Side-by-Side Train vs Val Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Constant LR (None)": "#1f77b4", "ReduceLROnPlateau": "#ff7f0e", "CosineAnnealingLR 🏆": "#2ca02c"}

    for name, t_loss in train_loss.items():
        axes[0].plot(epochs, t_loss, "-o", label=name, color=colors[name], linewidth=2.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Cross-Entropy Loss")
    axes[0].set_title("Training Loss Convergence", fontweight="bold")
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    for name, v_loss in val_loss.items():
        axes[1].plot(epochs, v_loss, "-s", label=name, color=colors[name], linewidth=2.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Cross-Entropy Loss")
    axes[1].set_title("Validation Loss Convergence (Lowest: 0.2088)", fontweight="bold")
    axes[1].set_xticks(epochs)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle("EXP-02: Training vs Validation Loss Curves Across Schedulers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out1 = os.path.join(PLOT_DIR, "exp_02_loss_detailed_curves.png")
    plt.savefig(out1, dpi=300)
    plt.close()
    logger.info(f"Saved: {out1}")

    # 2.2 Layer-wise Learning Rate Decay (LLRD) Profile
    plt.figure(figsize=(9, 4.5))
    layers = ["FC Head\n(New Params)", "Layer 4\n(Deep Features)", "Layer 1-3\n(Backbone)"]
    lr_rates = [1e-3, 1e-4, 0.0]

    bars = plt.bar(layers, lr_rates, color=["#e6550d", "#fdae6b", "#cccccc"], edgecolor="black", width=0.45)
    plt.yscale("log")
    plt.ylabel("Learning Rate (Log Scale)", fontweight="bold")
    plt.title("EXP-02: Layer-wise Learning Rate Decay (LLRD) Profile", fontweight="bold")
    plt.ylim(1e-5, 5e-3)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar, lr in zip(bars, lr_rates):
        label = f"{lr:.0e}" if lr > 0 else "Frozen (0)"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3 if lr > 0 else 2e-5, label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out2 = os.path.join(PLOT_DIR, "exp_02_llrd_learning_rate_profile.png")
    plt.savefig(out2, dpi=300)
    plt.close()
    logger.info(f"Saved: {out2}")

# -----------------------------------------------------------------------------
# EXP-03 Individual Plots (Advanced Augmentations)
# -----------------------------------------------------------------------------
def plot_exp03_granular():
    logger.info("Generating EXP-03 Granular Plots...")

    # 3.1 Loss & Accuracy Trajectory under Regularization
    epochs = [1, 2]
    train_loss = [0.9808, 0.8028]
    val_loss = [0.7494, 0.7027]
    train_acc = [80.35, 88.40]
    val_acc = [91.34, 92.72]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, "--o", label="Train Loss (Label Smoothed)", color="#d62728", linewidth=2)
    axes[0].plot(epochs, val_loss, "-s", label="Val Loss", color="#1f77b4", linewidth=2.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss (Label Smoothing = 0.1)", fontweight="bold")
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, "--o", label="Train Accuracy (%)", color="#ff7f0e", linewidth=2)
    axes[1].plot(epochs, val_acc, "-^", label="Val Accuracy (%)", color="#2ca02c", linewidth=2.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Trajectory (Zero Overfitting)", fontweight="bold")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(75, 95)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle("EXP-03: RandAugment + RandomErasing + Label Smoothing Impact", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out1 = os.path.join(PLOT_DIR, "exp_03_regularization_trajectory.png")
    plt.savefig(out1, dpi=300)
    plt.close()
    logger.info(f"Saved: {out1}")

    # 3.2 Overfitting Gap Comparison (Baseline vs Advanced Augs)
    plt.figure(figsize=(8, 4.5))
    pipelines = ["Standard Baseline\n(Crop + Flip)", "Advanced RandAugment\n(+ Label Smoothing)"]
    train_accs = [91.82, 88.40]
    val_accs   = [92.36, 92.72]

    x = np.arange(len(pipelines))
    width = 0.35

    plt.bar(x - width/2, train_accs, width, label="Train Accuracy (%)", color="#aec7e8", edgecolor="black")
    plt.bar(x + width/2, val_accs, width, label="Val Accuracy (%)", color="#1f77b4", edgecolor="black")

    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.title("EXP-03: Generalization Gap Comparison", fontweight="bold")
    plt.xticks(x, pipelines)
    plt.ylim(80, 96)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.annotate("Gap = -4.32%\n(Val Acc > Train Acc!)", xy=(1, 92.72), xytext=(0.7, 94.5),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9, fontweight="bold", color="darkgreen")

    plt.tight_layout()
    out2 = os.path.join(PLOT_DIR, "exp_03_generalization_gap.png")
    plt.savefig(out2, dpi=300)
    plt.close()
    logger.info(f"Saved: {out2}")

# -----------------------------------------------------------------------------
# EXP-04 Individual Plots (Native 32x32 Stem Training)
# -----------------------------------------------------------------------------
def plot_exp04_granular():
    logger.info("Generating EXP-04 Granular Plots...")

    epochs = [1, 2, 3]
    train_loss = [1.3732, 1.0540, 0.9057]
    val_loss   = [0.9377, 0.7620, 0.6514]
    train_acc  = [50.53, 62.10, 68.19]
    val_acc    = [67.76, 73.40, 76.98]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 4.1 Loss Convergence Over Epochs
    axes[0].plot(epochs, train_loss, "--o", label="Train Loss", color="#d62728", linewidth=2)
    axes[0].plot(epochs, val_loss, "-s", label="Val Loss", color="#1f77b4", linewidth=2.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Native 32x32 Stem Loss Convergence", fontweight="bold")
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 4.2 Accuracy Progression Over Epochs
    axes[1].plot(epochs, train_acc, "--o", label="Train Accuracy (%)", color="#ff7f0e", linewidth=2)
    axes[1].plot(epochs, val_acc, "-^", label="Val Accuracy (%)", color="#2ca02c", linewidth=2.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Native 32x32 Stem Accuracy Growth (Reaches 76.98%)", fontweight="bold")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(45, 85)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle("EXP-04: Native 32x32 Conv Stem Multi-Epoch Fine-Tuning Trajectory", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "exp_04_native_stem_trajectory.png")
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")

# -----------------------------------------------------------------------------
# EXP-05 Individual Plots (Vision Architecture Sweep)
# -----------------------------------------------------------------------------
def plot_exp05_granular():
    logger.info("Generating EXP-05 Granular Plots...")

    archs = ["EfficientNet-B0", "DenseNet121", "ResNet18", "ConvNeXt-Tiny 🏆"]
    epochs = [1, 2]

    # Val Loss curves
    val_loss_data = {
        "EfficientNet-B0": [0.5101, 0.4602],
        "DenseNet121": [0.3086, 0.2810],
        "ResNet18": [0.2867, 0.2450],
        "ConvNeXt-Tiny 🏆": [0.1200, 0.1085],
    }

    # Val Accuracy curves
    val_acc_data = {
        "EfficientNet-B0": [82.54, 84.44],
        "DenseNet121": [89.16, 90.70],
        "ResNet18": [90.12, 92.36],
        "ConvNeXt-Tiny 🏆": [95.94, 96.42],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"EfficientNet-B0": "#d62728", "DenseNet121": "#ff7f0e", "ResNet18": "#1f77b4", "ConvNeXt-Tiny 🏆": "#9467bd"}

    # 5.1 Validation Loss Trajectories
    for arch, v_loss in val_loss_data.items():
        axes[0].plot(epochs, v_loss, "-o", label=arch, color=colors[arch], linewidth=2.2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss Comparison (ConvNeXt = 0.1085)", fontweight="bold")
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 5.2 Validation Accuracy Trajectories
    for arch, v_acc in val_acc_data.items():
        axes[1].plot(epochs, v_acc, "-s", label=arch, color=colors[arch], linewidth=2.2)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("Validation Accuracy Comparison (ConvNeXt = 96.42%)", fontweight="bold")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(80, 98)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle("EXP-05: Epoch-by-Epoch Performance Trajectories across Modern Vision Backbones", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "exp_05_architecture_trajectories.png")
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")

# -----------------------------------------------------------------------------
# EXP-06 Individual Plots (ConvNeXt SOTA 10-Epoch Combination)
# -----------------------------------------------------------------------------
def plot_exp06_granular():
    logger.info("Generating EXP-06 Granular Plots...")

    epochs = np.arange(1, 11)
    val_acc = [95.76, 96.20, 96.65, 96.90, 97.00, 97.25, 97.40, 97.55, 97.62, 97.66]

    # 6.1 SOTA Recipe Component Waterfall / Gain Breakdown
    plt.figure(figsize=(9, 5))
    stages = [
        "ResNet18\nBaseline",
        "Optuna HPO\nWinner",
        "Cosine + LLRD\nScheduler",
        "ConvNeXt-Tiny\nArchitecture",
        "EXP-06 SOTA\nCombo 🏆"
    ]
    stage_accs = [92.36, 92.06, 92.78, 96.42, 97.66]
    stage_colors = ["#aec7e8", "#17becf", "#2ca02c", "#9467bd", "#ffd700"]

    bars = plt.bar(stages, stage_accs, color=stage_colors, edgecolor="black", width=0.55)
    plt.ylabel("Validation Accuracy (%)", fontweight="bold")
    plt.title("EXP-06: Incremental Performance Waterfall (From 92.36% to 97.66%)", fontweight="bold")
    plt.ylim(88, 100)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f"{height:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.annotate("🏆 All-Time Record: 97.66%", xy=(4, 97.66), xytext=(2.8, 98.8),
                 arrowprops=dict(facecolor="gold", edgecolor="black", shrink=0.08, width=2, headwidth=8),
                 fontsize=10, fontweight="bold", color="darkgreen")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "exp_06_sota_waterfall_gain.png")
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")

def generate_all_granular():
    create_output_dir()
    plot_exp01_granular()
    plot_exp02_granular()
    plot_exp03_granular()
    plot_exp04_granular()
    plot_exp05_granular()
    plot_exp06_granular()
    logger.info("🎉 All granular experiment plots (EXP-01 to EXP-06) generated successfully!")

if __name__ == "__main__":
    generate_all_granular()
