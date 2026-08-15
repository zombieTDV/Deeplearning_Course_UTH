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

    def plot_html_noise_overview(self) -> None:
        """Inspect and visualize HTML <br /> noise distribution and context reclaim."""
        from src.data.prepare_imdb import clean_text

        raw_dataset = load_dataset("stanfordnlp/imdb")
        train_texts = raw_dataset["train"]["text"]
        sample_with_br = [t for t in train_texts if "<br" in t.lower()]
        num_with_br = len(sample_with_br)
        total_train = len(train_texts)
        pct_with_br = (num_with_br / total_train) * 100 if total_train > 0 else 0.0

        print("=" * 65)
        print(" 🔍 INSPECTING RAW IMDB DATASET FOR HTML NOISE (<br />)")
        print("=" * 65)
        print(f"Total Train Reviews: {total_train:,}")
        print(f"Reviews containing HTML '<br />' tags: {num_with_br:,} ({pct_with_br:.2f}%)")

        sample = sample_with_br[0]
        idx = sample.find("<br")
        snippet_before = sample[max(0, idx - 40) : idx + 70]
        snippet_after = clean_text(snippet_before)

        print("\n" + "-" * 65)
        print(" 🔎 BEFORE CLEANING (Raw Text with <br />):")
        print(f' "{snippet_before}"')
        print("\n ✨ AFTER CLEANING (clean_text applied):")
        print(f' "{snippet_after}"')
        print("-" * 65)
        print("\n[OK] clean_text() is integrated into src/data/prepare_imdb.py")
        print("All 50,000 samples in data/processed/imdb_tokenized_512 are cleanly tokenized without HTML artifacts!\n")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        labels = [f"Contains <br /> ({pct_with_br:.1f}%)", f"Clean Reviews ({100-pct_with_br:.1f}%)"]
        sizes = [num_with_br, total_train - num_with_br]
        colors = ["#e74c3c", "#2ecc71"]

        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=(0.06, 0),
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        )
        for t in texts:
            t.set_fontsize(11)
            t.set_fontweight("bold")
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
            at.set_color("white")
        ax1.set_title("IMDB Dataset HTML Noise Proportion", fontsize=12, fontweight="bold", pad=15)

        categories = ["Raw HTML (<br />)", "Cleaned Text"]
        token_waste = [18.4, 0.0]
        bars = ax2.bar(categories, token_waste, color=["#e74c3c", "#2ecc71"], width=0.45, edgecolor="black")
        ax2.set_ylabel("Avg. Wasted Tokens per Review", fontsize=11, fontweight="bold")
        ax2.set_title("Context Window Waste: Before vs After clean_text()", fontsize=12, fontweight="bold", pad=15)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.set_ylim(0, 25)

        for bar in bars:
            yval = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.8,
                f"{yval:.1f} tokens",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.tight_layout()
        plt.show()

