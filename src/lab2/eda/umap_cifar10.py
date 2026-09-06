"""
umap_cifar10.py — UMAP EDA on raw CIFAR-10 pixel features.

Runs UMAP dimensionality reduction on the raw flattened pixel vectors
(32×32×3 = 3072-d) of the CIFAR-10 test set and saves:
  - experiments/plots/umap_cifar10_raw_pixels.png  — scatter plot (300 dpi)
  - experiments/results/umap_cifar10_embeddings.npy — (N, 2) float32 array
  - experiments/results/umap_cifar10_labels.npy     — (N,) int64 label array

Usage:
    python src/eda/umap_cifar10.py [--data-root ./data/raw]
                                   [--n-neighbors 15]
                                   [--min-dist 0.1]
                                   [--n-samples 10000]
                                   [--seed 42]
                                   [--out-dir experiments]

Run from the project root so that src/ imports resolve correctly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path (allows `python src/eda/umap_cifar10.py`)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lab2.eval.evaluate_model import CIFAR10_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CIFAR-10 class colour palette (one colour per class, consistent with ROC charts)
# ---------------------------------------------------------------------------
_CLASS_COLORS = [
    "#1f77b4",  # airplane
    "#ff7f0e",  # automobile
    "#2ca02c",  # bird
    "#d62728",  # cat
    "#9467bd",  # deer
    "#8c564b",  # dog
    "#e377c2",  # frog
    "#7f7f7f",  # horse
    "#bcbd22",  # ship
    "#17becf",  # truck
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_cifar10_test_pixels(data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X is float32 normalised pixels, y is int64 labels."""
    import torchvision
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),   # → [0,1] float32, (3,32,32)
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=True, transform=transform
    )
    # Stack all images into one array: (N, 3072)
    X_list, y_list = [], []
    for img_tensor, label in dataset:
        X_list.append(img_tensor.numpy().reshape(-1))
        y_list.append(label)
    X = np.array(X_list, dtype=np.float32)   # (10000, 3072)
    y = np.array(y_list, dtype=np.int64)     # (10000,)
    return X, y


def _run_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Fit UMAP and return 2-D embeddings (N, 2)."""
    import umap  # lazy import — installed as umap-learn

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(X)


def _plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    title: str = "UMAP of CIFAR-10 Test Set (Raw 32×32×3 Pixels)",
) -> None:
    """Render a publication-quality UMAP scatter plot and save to *save_path*."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    for class_idx, (class_name, color) in enumerate(zip(CIFAR10_CLASSES, _CLASS_COLORS)):
        mask = labels == class_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=color,
            s=16,
            alpha=0.6,
            linewidths=0,
            label=class_name,
            rasterized=True,
        )

    legend = ax.legend(
        title="CIFAR-10 Class",
        loc="upper right",
        markerscale=4,
        fontsize=9,
        title_fontsize=10,
        facecolor="#1c1c1c",
        edgecolor="#555555",
        labelcolor="white",
    )
    legend.get_title().set_color("white")

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=14)
    ax.set_xlabel("UMAP Dimension 1", fontsize=10, color="#aaaaaa")
    ax.set_ylabel("UMAP Dimension 2", fontsize=10, color="#aaaaaa")
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    n = len(labels)
    ax.annotate(
        f"n = {n:,} test samples  |  raw 32×32×3 pixels (3072-d → 2-d)",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=7.5,
        color="#888888",
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP plot saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="UMAP EDA on raw CIFAR-10 pixels")
    parser.add_argument("--data-root",   default="./data/raw",     help="CIFAR-10 download directory")
    parser.add_argument("--n-neighbors", type=int,   default=15,   help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist",    type=float, default=0.1,  help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--n-samples",   type=int,   default=10000, help="Max samples to embed (default: all 10k test)")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    parser.add_argument("--out-dir",     default="experiments",    help="Root output directory")
    args = parser.parse_args(argv)

    out_dir = PROJECT_ROOT / args.out_dir
    plots_dir   = out_dir / "plots"
    results_dir = out_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw pixels
    logger.info("Loading CIFAR-10 test set from %s …", args.data_root)
    X, y = _load_cifar10_test_pixels(args.data_root)
    logger.info("Loaded %d samples, shape %s", len(X), X.shape)

    # 2. Optional subsample
    if args.n_samples and args.n_samples < len(X):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X), size=args.n_samples, replace=False)
        X, y = X[idx], y[idx]
        logger.info("Subsampled to %d samples", args.n_samples)

    # 3. UMAP
    logger.info(
        "Running UMAP (n_neighbors=%d, min_dist=%.2f, seed=%d) …",
        args.n_neighbors, args.min_dist, args.seed,
    )
    t0 = perf_counter()
    embeddings = _run_umap(X, n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed)
    elapsed = perf_counter() - t0
    logger.info("UMAP complete in %.1f s → embeddings shape %s", elapsed, embeddings.shape)

    # 4. Save embeddings & labels
    emb_path = results_dir / "umap_cifar10_embeddings.npy"
    lbl_path = results_dir / "umap_cifar10_labels.npy"
    np.save(emb_path, embeddings.astype(np.float32))
    np.save(lbl_path, y)
    logger.info("Embeddings → %s", emb_path)
    logger.info("Labels     → %s", lbl_path)

    # 5. Plot
    plot_path = plots_dir / "umap_cifar10_raw_pixels.png"
    _plot_umap(
        embeddings, y, save_path=plot_path,
        title=(
            f"UMAP of CIFAR-10 Test Set (Raw 32×32×3 Pixels)\n"
            f"n_neighbors={args.n_neighbors}  min_dist={args.min_dist}  seed={args.seed}"
        ),
    )
    logger.info("✓ Done. Plot → %s", plot_path)


if __name__ == "__main__":
    main()


