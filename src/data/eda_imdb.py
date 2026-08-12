"""IMDB Dataset EDA and Visualization Script.

Performs Exploratory Data Analysis (EDA) on the IMDB sentiment dataset:
1. Label distribution & class balance check (50/50 ratio).
2. Review length distribution (word counts & character lengths).
3. Saves plots to `experiments/plots/`:
   - `imdb_label_distribution.png` (Label Balance Bar Chart)
   - `imdb_review_length_distribution.png` (Review Length Histogram & Boxplot)
4. Saves structured EDA statistics to `experiments/results/imdb_dataset_eda.json`.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def run_imdb_eda() -> dict[str, Any]:
    print("Loading [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) dataset...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    print(f"Train samples: {len(train_ds):,}")
    print(f"Test samples:  {len(test_ds):,}")

    train_labels = list(train_ds["label"])
    test_labels = list(test_ds["label"])

    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    test_pos = sum(test_labels)
    test_neg = len(test_labels) - test_pos

    train_word_counts = [len(text.split()) for text in train_ds["text"]]

    # 1. Plot Label Distribution
    plots_dir = Path("experiments/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ["Negative (0)", "Positive (1)"]
    train_counts = [train_neg, train_pos]
    test_counts = [test_neg, test_pos]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, train_counts, width, label="Train Split (25k)", color="#4C72B0")
    ax.bar(x + width / 2, test_counts, width, label="Test Split (25k)", color="#55A868")

    ax.set_ylabel("Number of Samples")
    ax.set_title("IMDB Dataset Class Balance (50/50 Ratio)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 15000)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(categories)):
        ax.text(x[i] - width / 2, train_counts[i] + 200, f"{train_counts[i]:,}", ha="center")
        ax.text(x[i] + width / 2, test_counts[i] + 200, f"{test_counts[i]:,}", ha="center")

    plt.tight_layout()
    label_plot_path = plots_dir / "imdb_label_distribution.png"
    plt.savefig(label_plot_path, dpi=150)
    plt.close()
    print(f"Saved Label Distribution Plot to: {label_plot_path}")

    # 2. Plot Review Length Distribution (Word Count)
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_hist.hist(train_word_counts, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("IMDB Review Length Distribution (Word Counts)")
    ax_hist.axvline(
        np.median(train_word_counts),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {np.median(train_word_counts):.0f} words",
    )
    ax_hist.axvline(
        np.percentile(train_word_counts, 95),
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"95th Pct: {np.percentile(train_word_counts, 95):.0f} words",
    )
    ax_hist.legend()
    ax_hist.grid(axis="y", linestyle="--", alpha=0.5)

    ax_box.boxplot(train_word_counts, orientation="horizontal", patch_artist=True, boxprops=dict(facecolor="#4C72B0"))
    ax_box.set_xlabel("Review Length (Word Count)")
    ax_box.set_yticks([])

    plt.tight_layout()
    length_plot_path = plots_dir / "imdb_review_length_distribution.png"
    plt.savefig(length_plot_path, dpi=150)
    plt.close()
    print(f"Saved Review Length Plot to: {length_plot_path}")

    # 3. Save Structured EDA JSON
    eda_stats = {
        "dataset_name": "stanfordnlp/imdb",
        "splits": {
            "train": {"total": len(train_ds), "negative": train_neg, "positive": train_pos},
            "test": {"total": len(test_ds), "negative": test_neg, "positive": test_pos},
        },
        "review_length_word_count_stats": {
            "mean": float(np.mean(train_word_counts)),
            "median": float(np.median(train_word_counts)),
            "std": float(np.std(train_word_counts)),
            "pct_95": float(np.percentile(train_word_counts, 95)),
            "max": int(np.max(train_word_counts)),
            "min": int(np.min(train_word_counts)),
        },
        "plots": {
            "label_distribution": str(label_plot_path),
            "review_length_distribution": str(length_plot_path),
        },
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    eda_json_path = results_dir / "imdb_dataset_eda.json"

    with open(eda_json_path, "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, indent=2)

    print(f"Saved EDA Statistics JSON to: {eda_json_path}")
    return eda_stats


if __name__ == "__main__":
    run_imdb_eda()
