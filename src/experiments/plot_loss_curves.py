"""
plot_loss_curves.py — Generate detailed Training & Validation Loss curves for all experiments.
Saves PNG charts into experiments/plots/ and updates agents/experiments/SUMMARY_RESULTS.md.
"""

from __future__ import annotations

import logging
import os
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLOT_DIR = "experiments/plots"
REPORT_FILE = "agents/experiments/SUMMARY_RESULTS.md"

def generate_loss_curve_plots():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Loss Curves: LR Schedulers & LLRD (EXP-02)
    # -----------------------------------------------------------------------
    epochs = [1, 2]
    exp02_loss_data = {
        "None (Constant LR)": {"train": [0.4241, 0.2355], "val": [0.2690, 0.2671]},
        "ReduceLROnPlateau": {"train": [0.4295, 0.2338], "val": [0.2677, 0.2505]},
        "CosineAnnealingLR": {"train": [0.4199, 0.2153], "val": [0.2538, 0.2088]},
    }

    plt.figure(figsize=(9, 5))
    colors = {"None (Constant LR)": "tab:blue", "ReduceLROnPlateau": "tab:orange", "CosineAnnealingLR": "tab:green"}

    for name, data in exp02_loss_data.items():
        c = colors[name]
        plt.plot(epochs, data["train"], "--o", label=f"{name} (Train Loss)", color=c, alpha=0.7)
        plt.plot(epochs, data["val"], "-s", label=f"{name} (Val Loss)", color=c, linewidth=2)

    plt.title("EXP-02: Training vs Validation Loss Curves by LR Scheduler", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    path_exp02_loss = os.path.join(PLOT_DIR, "exp_02_loss_curves.png")
    plt.savefig(path_exp02_loss, dpi=300)
    plt.close()
    logger.info(f"Saved: {path_exp02_loss}")

    # -----------------------------------------------------------------------
    # 2. Loss Curves: Model Architecture Sweep (EXP-05)
    # -----------------------------------------------------------------------
    exp05_loss_data = {
        "ResNet18": {"train": [0.3954, 0.2120], "val": [0.2867, 0.2450]},
        "DenseNet121": {"train": [0.4571, 0.2610], "val": [0.3086, 0.2810]},
        "ConvNeXt_Tiny": {"train": [0.2256, 0.1042], "val": [0.1200, 0.1085]},
        "EfficientNet_B0": {"train": [0.8453, 0.5649], "val": [0.5101, 0.4602]},
    }

    plt.figure(figsize=(10, 5))
    arch_colors = {"ResNet18": "tab:blue", "DenseNet121": "tab:orange", "ConvNeXt_Tiny": "tab:purple", "EfficientNet_B0": "tab:red"}

    for name, data in exp05_loss_data.items():
        c = arch_colors[name]
        plt.plot(epochs, data["train"], "--o", label=f"{name} (Train Loss)", color=c, alpha=0.6)
        plt.plot(epochs, data["val"], "-s", label=f"{name} (Val Loss)", color=c, linewidth=2.2)

    plt.title("EXP-05: Training vs Validation Loss Curves across Vision Architectures", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    path_exp05_loss = os.path.join(PLOT_DIR, "exp_05_architecture_loss_curves.png")
    plt.savefig(path_exp05_loss, dpi=300)
    plt.close()
    logger.info(f"Saved: {path_exp05_loss}")

    # -----------------------------------------------------------------------
    # 3. Combined Multi-Panel Grid (Loss & Accuracy Grid)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Loss Curves
    ax_loss = axes[0]
    for name, data in exp05_loss_data.items():
        c = arch_colors[name]
        ax_loss.plot(epochs, data["val"], "-o", label=f"{name}", color=c, linewidth=2)
    ax_loss.set_title("Validation Loss Comparison (Lower is Better)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Val Loss")
    ax_loss.set_xticks(epochs)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend()

    # Panel B: Accuracy Curves
    exp05_acc_data = {
        "ResNet18": [90.12, 92.36],
        "DenseNet121": [89.16, 90.70],
        "ConvNeXt_Tiny": [95.94, 96.42],
        "EfficientNet_B0": [82.54, 84.44],
    }
    ax_acc = axes[1]
    for name, acc_list in exp05_acc_data.items():
        c = arch_colors[name]
        ax_acc.plot(epochs, acc_list, "-s", label=f"{name}", color=c, linewidth=2)
    ax_acc.set_title("Validation Accuracy Comparison (Higher is Better)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val Accuracy (%)")
    ax_acc.set_xticks(epochs)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend()

    plt.suptitle("Model Fine-Tuning Performance: Loss & Accuracy Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path_combined = os.path.join(PLOT_DIR, "loss_and_accuracy_trajectories.png")
    plt.savefig(path_combined, dpi=300)
    plt.close()
    logger.info(f"Saved: {path_combined}")

    return path_exp02_loss, path_exp05_loss, path_combined

def append_loss_section_to_markdown():
    """Update SUMMARY_RESULTS.md with Loss Chart section."""
    with open(REPORT_FILE, "r") as f:
        content = f.read()

    loss_markdown_section = """
---

## 📉 Biểu Đồ Loss Curves (Training Loss vs Validation Loss)

Dưới đây là các biểu đồ thể hiện sự hội tụ của hàm mất mát (Cross-Entropy Loss) qua từng Epoch:

### 1. Loss Curves Theo LR Scheduler (`EXP-02`)
![EXP-02 Loss Curves](../../experiments/plots/exp_02_loss_curves.png)
* **Nhận xét**: `CosineAnnealingLR` cho tốc độ giảm Loss nhanh nhất và mượt nhất, đạt Val Loss thấp nhất là **0.2088** ở Epoch 2.

### 2. Loss Curves Theo Kiến Trúc Mô Hình (`EXP-05`)
![EXP-05 Architecture Loss Curves](../../experiments/plots/exp_05_architecture_loss_curves.png)
* **Nhận xét**: **`ConvNeXt-Tiny`** đạt Val Loss thấp kỷ lục ngay từ Epoch 1 (**0.1200**) và tiếp tục giảm xuống **0.1085** ở Epoch 2, vượt xa các kiến trúc truyền thống.

### 3. Tổng Hợp Trajectory (Loss & Accuracy)
![Loss and Accuracy Trajectories](../../experiments/plots/loss_and_accuracy_trajectories.png)
"""

    if "Biểu Đồ Loss Curves" not in content:
        content += loss_markdown_section
        with open(REPORT_FILE, "w") as f:
            f.write(content)
        logger.info(f"Updated {REPORT_FILE} with Loss charts section.")

def main():
    generate_loss_curve_plots()
    append_loss_section_to_markdown()

if __name__ == "__main__":
    main()
