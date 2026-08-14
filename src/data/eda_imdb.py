"""IMDB Dataset Visual EDA Utilities."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from src.data.prepare_imdb import prepare_imdb


class IMDBDatasetEDA:
    """High-level class for dataset exploration and visual EDA."""

    def __init__(self, max_length: int = 512, processed_dir: str | None = None) -> None:
        kwargs: dict[str, Any] = {"max_length": max_length}
        if processed_dir:
            kwargs["processed_dir"] = processed_dir
        self.ds, self.tokenizer, self.meta = prepare_imdb(**kwargs)

    def plot_dataset_overview(self, sample_size: int = 1000) -> None:
        """Plot label distribution and untruncated token length distribution figures."""
        train_labels = np.array(self.ds["train"]["label"])
        val_labels = np.array(self.ds["val"]["label"])
        test_labels = np.array(self.ds["test"]["label"])

        split_names = ["Train (22.5k)", "Val (2.5k)", "Test (25k)"]
        neg_counts = [int(np.sum(train_labels == 0)), int(np.sum(val_labels == 0)), int(np.sum(test_labels == 0))]
        pos_counts = [int(np.sum(train_labels == 1)), int(np.sum(val_labels == 1)), int(np.sum(test_labels == 1))]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        x = np.arange(len(split_names))
        width = 0.35

        ax1.bar(x - width/2, neg_counts, width, label="Negative (0)", color="#e74c3c", edgecolor="black")
        ax1.bar(x + width/2, pos_counts, width, label="Positive (1)", color="#2ecc71", edgecolor="black")
        ax1.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
        ax1.set_title("Exact Class Balance Across All 3 Splits (50/50)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(split_names, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        overall_pos = sum(pos_counts)
        overall_neg = sum(neg_counts)
        ax2.pie([overall_neg, overall_pos], labels=["Negative (50.0%)", "Positive (50.0%)"], autopct="%1.2f%%", colors=["#e74c3c", "#2ecc71"], startangle=90, wedgeprops=dict(width=0.4, edgecolor="w"))
        ax2.set_title("Overall Dataset Label Ratio (50,000 Total Reviews)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.show()

        # Figure 2: Review Length Distributions
        raw_train_sample = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(sample_size))
        word_counts = [len(x["text"].split()) for x in raw_train_sample]
        token_lengths = [len(self.tokenizer.encode(x["text"], truncation=False)) for x in raw_train_sample]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        ax1.hist(word_counts, bins=50, color="#8e44ad", edgecolor="black", alpha=0.75)
        ax1.axvline(np.mean(word_counts), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(word_counts):.0f} words")
        ax1.axvline(np.median(word_counts), color="yellow", linestyle="-.", linewidth=2, label=f"Median: {np.median(word_counts):.0f} words")
        ax1.set_title("Review Word Count Distribution (Raw Text)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Word Count per Review", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.hist(token_lengths, bins=40, color="#3498db", edgecolor="black", alpha=0.8)
        ax2.axvline(np.mean(token_lengths), color="orange", linestyle="--", linewidth=2, label=f"Mean: {np.mean(token_lengths):.0f} tokens")
        ax2.axvline(256, color="red", linestyle="--", linewidth=2, label="256 Tokens Cutoff")
        ax2.axvline(512, color="green", linestyle="--", linewidth=2, label="512 Tokens Cutoff")
        ax2.set_title("DistilBERT Untruncated Token Length Distribution", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Untruncated Token Count per Review", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

    def print_sample_reviews(self) -> None:
        """Print sample positive and negative reviews."""
        raw_train_sample = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(10))
        pos_review = next(x["text"] for x in raw_train_sample if x["label"] == 1)
        neg_review = next(x["text"] for x in raw_train_sample if x["label"] == 0)

        print("--- Sample Positive Review ---")
        print(f'"{pos_review[:250]}..."\n')
        print("--- Sample Negative Review ---")
        print(f'"{neg_review[:250]}..."\n')
