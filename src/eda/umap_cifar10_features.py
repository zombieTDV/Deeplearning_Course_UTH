"""
umap_cifar10_features.py — UMAP on deep latent features extracted by trained models.

Extracts the penultimate feature representations (pre-classification MLP layer)
from our best model variants on the CIFAR-10 test set (10,000 samples) and runs
UMAP dimensionality reduction to visualize learned semantic class separation.

Features extracted:
  - ResNet18 variants:   512-d feature vector (after global avgpool, before fc)
  - DenseNet121 variants: 1024-d feature vector (after global avgpool, before classifier)
  - SOTA Ensemble:       1536-d concatenated latent representation [ResNet18-sota (512) + DenseNet121-sota (1024)]

Outputs:
  - experiments/plots/umap_cifar10_features_<model_name>.png
  - experiments/plots/umap_cifar10_features_comparison.png (Multi-panel dashboard)
  - experiments/results/umap_embeddings_<model_name>.npy
  - experiments/results/umap_labels_<model_name>.npy
  - experiments/results/umap_features_summary.json (Silhouette, Davies-Bouldin, CH scores)

Usage:
    # Run UMAP on best SOTA models (ResNet18-sota & DenseNet121-sota)
    python src/eda/umap_cifar10_features.py

    # Run specific model variants
    python src/eda/umap_cifar10_features.py --models ResNet18-sota DenseNet121-sota SOTA-Ensemble

    # Run on all 6 variants + Ensemble + generate comparison dashboard
    python src/eda/umap_cifar10_features.py --all

    # Custom UMAP hyperparameters
    python src/eda/umap_cifar10_features.py --n-neighbors 15 --min-dist 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_eval_transform  # noqa: E402
from src.eval.evaluate_model import CIFAR10_CLASSES  # noqa: E402
from src.models.build_model import (  # noqa: E402
    build_densenet121,
    build_densenet121_full_sota,
    build_resnet18,
    build_resnet18_full_sota,
)
from src.utils.checkpoint_utils import find_best_checkpoint, load_model_weights  # noqa: E402

logger = logging.getLogger("umap_cifar10_features")

# ---------------------------------------------------------------------------
# CIFAR-10 class colour palette (consistent with raw UMAP & ROC charts)
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

# Model builders mapping
MODEL_BUILDERS: dict[str, Callable[[torch.device], nn.Module]] = {
    "ResNet18-sota": lambda dev: build_resnet18_full_sota(num_classes=10, device=dev),
    "DenseNet121-sota": lambda dev: build_densenet121_full_sota(num_classes=10, device=dev),
    "ResNet18-finetune": lambda dev: build_resnet18(num_classes=10, mode="finetune", device=dev),
    "DenseNet121-finetune": lambda dev: build_densenet121(num_classes=10, mode="finetune", device=dev),
    "ResNet18-frozen": lambda dev: build_resnet18(num_classes=10, mode="frozen", device=dev),
    "DenseNet121-frozen": lambda dev: build_densenet121(num_classes=10, mode="frozen", device=dev),
}


# ---------------------------------------------------------------------------
# Feature Extraction Helpers
# ---------------------------------------------------------------------------
def build_test_loader(
    data_root: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """CIFAR-10 test DataLoader with the ImageNet-compatible 224x224 eval transform."""
    import torchvision

    transform = get_eval_transform()
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def extract_model_features(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate layer feature embeddings (N, D) and true labels (N)."""
    model.eval()
    model.to(device)

    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    # Register forward hook on the input to the classifier layer
    captured_batch: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, inputs: tuple[torch.Tensor, ...], outputs: torch.Tensor) -> None:
        # inputs[0] is the feature tensor passed into the classifier
        captured_batch.append(inputs[0].detach())

    target_layer = model.fc if hasattr(model, "fc") else model.classifier
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            captured_batch.clear()
            _ = model(images)
            if captured_batch:
                feat = captured_batch[0]
                if feat.dim() > 2:
                    feat = torch.flatten(feat, 1)
                features_list.append(feat.cpu())
            labels_list.append(targets.clone())
    finally:
        handle.remove()

    X = torch.cat(features_list, dim=0).numpy().astype(np.float32)
    y = torch.cat(labels_list, dim=0).numpy().astype(np.int64)
    return X, y


def load_model_for_umap(model_name: str, device: torch.device) -> nn.Module:
    """Build and load checkpoint weights for *model_name*."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name '{model_name}'. Options: {list(MODEL_BUILDERS.keys())}")

    builder = MODEL_BUILDERS[model_name]
    model = builder(device)

    ckpt_path = find_best_checkpoint(model_name)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for '{model_name}'.")

    model = load_model_weights(model, ckpt_path, device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# UMAP & Clustering Metrics
# ---------------------------------------------------------------------------
def run_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
) -> np.ndarray:
    """Fit UMAP on feature matrix X and return 2D embeddings (N, 2)."""
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(X)


def compute_cluster_metrics(X_emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute standard cluster quality metrics on 2D UMAP space."""
    sil = float(silhouette_score(X_emb, labels))
    db = float(davies_bouldin_score(X_emb, labels))
    ch = float(calinski_harabasz_score(X_emb, labels))
    return {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_index": round(ch, 2),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_umap_single(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    feature_dim: int,
    metrics: dict[str, float],
    save_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> None:
    """Render publication-quality single-model UMAP scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    for class_idx, (class_name, color) in enumerate(zip(CIFAR10_CLASSES, _CLASS_COLORS)):
        mask = labels == class_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=color,
            s=18,
            alpha=0.7,
            linewidths=0,
            label=class_name,
            rasterized=True,
        )

    legend = ax.legend(
        title="CIFAR-10 Class",
        loc="upper right",
        markerscale=3.5,
        fontsize=9.5,
        title_fontsize=10.5,
        facecolor="#1c1c1c",
        edgecolor="#555555",
        labelcolor="white",
    )
    legend.get_title().set_color("white")

    title = (
        f"UMAP of Learned Latent Features — {model_name}\n"
        f"Penultimate Layer ({feature_dim}-d → 2-d)  |  "
        f"Silhouette: {metrics['silhouette_score']:.3f}  |  "
        f"Davies-Bouldin: {metrics['davies_bouldin_index']:.3f}"
    )
    ax.set_title(title, fontsize=13.5, fontweight="bold", color="white", pad=14)
    ax.set_xlabel("UMAP Dimension 1", fontsize=10.5, color="#aaaaaa")
    ax.set_ylabel("UMAP Dimension 2", fontsize=10.5, color="#aaaaaa")
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    n = len(labels)
    ax.annotate(
        f"n = {n:,} CIFAR-10 test samples  |  n_neighbors={n_neighbors}  min_dist={min_dist}  seed={seed}",
        xy=(0.01, 0.015),
        xycoords="axes fraction",
        fontsize=8,
        color="#888888",
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", save_path)


def plot_umap_comparison_dashboard(
    runs_data: list[dict],
    save_path: Path,
) -> None:
    """Generate multi-panel comparison dashboard of Raw Pixels vs Models."""
    n_plots = len(runs_data)
    if n_plots == 0:
        return

    cols = min(n_plots, 4)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows))
    fig.patch.set_facecolor("#0e0e0e")

    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, run in enumerate(runs_data):
        ax = axes[idx]
        ax.set_facecolor("#0e0e0e")
        emb = run["embeddings"]
        lbls = run["labels"]
        name = run["name"]
        fdim = run["feature_dim"]
        metrics = run.get("metrics", {})

        for class_idx, (class_name, color) in enumerate(zip(CIFAR10_CLASSES, _CLASS_COLORS)):
            mask = lbls == class_idx
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=color,
                s=10,
                alpha=0.65,
                linewidths=0,
                label=class_name if idx == 0 else None,
                rasterized=True,
            )

        sil = metrics.get("silhouette_score", None)
        sil_str = f" | Sil: {sil:.3f}" if sil is not None else ""
        ax.set_title(f"{name} ({fdim}-d){sil_str}", fontsize=11, fontweight="bold", color="white", pad=8)
        ax.tick_params(colors="#555555", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    # Global legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        markerscale=3,
        fontsize=9,
        facecolor="#1c1c1c",
        edgecolor="#555555",
        labelcolor="white",
    )

    plt.suptitle(
        "Evolution of Feature Separability in UMAP Space: Raw Pixels vs Deep Model Representations",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=1.03,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison dashboard saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="UMAP on model penultimate features")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DenseNet121-sota", "ResNet18-sota", "SOTA-Ensemble"],
        help="Model variants to extract features from and visualize",
    )
    parser.add_argument("--all", action="store_true", help="Run all 6 variants + SOTA-Ensemble + Raw Pixels")
    parser.add_argument("--data-root", default="./data/raw", help="CIFAR-10 dataset directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--metric", default="euclidean", help="UMAP distance metric (euclidean/cosine)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", default="experiments", help="Output directory root")
    args = parser.parse_args(argv)

    out_dir = PROJECT_ROOT / args.out_dir
    plots_dir = out_dir / "plots"
    results_dir = out_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device for feature extraction: %s", device)

    # Determine models to evaluate
    if args.all:
        target_models = [
            "ResNet18-frozen",
            "DenseNet121-frozen",
            "ResNet18-finetune",
            "DenseNet121-finetune",
            "ResNet18-sota",
            "DenseNet121-sota",
            "SOTA-Ensemble",
        ]
    else:
        target_models = args.models

    # Build test DataLoader
    logger.info("Loading CIFAR-10 test set from %s …", args.data_root)
    test_loader = build_test_loader(args.data_root, batch_size=args.batch_size)
    logger.info("Test DataLoader initialized with %d samples", len(test_loader.dataset))

    extracted_features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summary_report: dict[str, dict] = {}
    dashboard_runs: list[dict] = []

    # 0. Check if raw pixels embeddings already exist for comparison
    raw_emb_path = results_dir / "umap_cifar10_embeddings.npy"
    raw_lbl_path = results_dir / "umap_cifar10_labels.npy"
    if raw_emb_path.exists() and raw_lbl_path.exists():
        raw_emb = np.load(raw_emb_path)
        raw_lbl = np.load(raw_lbl_path)
        raw_metrics = compute_cluster_metrics(raw_emb, raw_lbl)
        summary_report["Raw-Pixels-3072d"] = {
            "feature_dim": 3072,
            "metrics": raw_metrics,
        }
        dashboard_runs.append({
            "name": "Raw Pixels",
            "feature_dim": 3072,
            "embeddings": raw_emb,
            "labels": raw_lbl,
            "metrics": raw_metrics,
        })
        logger.info("Included existing Raw Pixels UMAP in summary dashboard (Sil: %.3f)", raw_metrics["silhouette_score"])

    # 1. Feature Extraction Pass
    for model_name in target_models:
        if model_name == "SOTA-Ensemble":
            continue  # Derived from ResNet18-sota + DenseNet121-sota

        logger.info(">>> Loading model: %s", model_name)
        try:
            model = load_model_for_umap(model_name, device)
        except Exception as e:
            logger.error("Failed to load %s: %s — skipping", model_name, e)
            continue

        logger.info("Extracting penultimate features for %s …", model_name)
        t0 = perf_counter()
        X, y = extract_model_features(model, model_name, test_loader, device)
        elapsed = perf_counter() - t0
        logger.info("Extracted %s features shape %s in %.2f s", model_name, X.shape, elapsed)
        extracted_features[model_name] = (X, y)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Handle SOTA-Ensemble concatenation if requested
    if "SOTA-Ensemble" in target_models:
        if "ResNet18-sota" not in extracted_features:
            rn = load_model_for_umap("ResNet18-sota", device)
            extracted_features["ResNet18-sota"] = extract_model_features(rn, "ResNet18-sota", test_loader, device)
            del rn

        if "DenseNet121-sota" not in extracted_features:
            dn = load_model_for_umap("DenseNet121-sota", device)
            extracted_features["DenseNet121-sota"] = extract_model_features(dn, "DenseNet121-sota", test_loader, device)
            del dn

        X_rn, y_rn = extracted_features["ResNet18-sota"]
        X_dn, _ = extracted_features["DenseNet121-sota"]
        X_ens = np.concatenate([X_rn, X_dn], axis=1)  # 512 + 1024 = 1536
        extracted_features["SOTA-Ensemble"] = (X_ens, y_rn)
        logger.info("Synthesized SOTA-Ensemble concatenated features shape %s", X_ens.shape)

    # 2. UMAP Dimensionality Reduction & Visualization Pass
    for model_name in target_models:
        if model_name not in extracted_features:
            continue
        X, y = extracted_features[model_name]

        fdim = X.shape[1]
        logger.info(
            "=== Running UMAP for %s (dim=%d, n_neighbors=%d, min_dist=%.2f, seed=%d) ===",
            model_name, fdim, args.n_neighbors, args.min_dist, args.seed,
        )
        t0 = perf_counter()
        embeddings = run_umap(
            X,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            seed=args.seed,
        )
        elapsed = perf_counter() - t0
        logger.info("UMAP for %s complete in %.1f s", model_name, elapsed)

        # Compute cluster metrics
        metrics = compute_cluster_metrics(embeddings, y)
        logger.info(
            "[%s] Silhouette: %.4f | Davies-Bouldin: %.4f | Calinski-Harabasz: %.2f",
            model_name, metrics["silhouette_score"], metrics["davies_bouldin_index"], metrics["calinski_harabasz_index"],
        )

        summary_report[model_name] = {
            "feature_dim": fdim,
            "metrics": metrics,
            "umap_time_seconds": round(elapsed, 2),
        }

        # Save embeddings & labels
        emb_path = results_dir / f"umap_embeddings_{model_name}.npy"
        lbl_path = results_dir / f"umap_labels_{model_name}.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        np.save(lbl_path, y)
        logger.info("Saved embeddings → %s", emb_path)

        # Save single scatter plot
        plot_path = plots_dir / f"umap_cifar10_features_{model_name}.png"
        plot_umap_single(
            embeddings,
            y,
            model_name=model_name,
            feature_dim=fdim,
            metrics=metrics,
            save_path=plot_path,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
        )

        dashboard_runs.append({
            "name": model_name,
            "feature_dim": fdim,
            "embeddings": embeddings,
            "labels": y,
            "metrics": metrics,
        })

    # 3. Generate Multi-panel Comparison Dashboard
    if len(dashboard_runs) > 1:
        comp_plot_path = plots_dir / "umap_cifar10_features_comparison.png"
        plot_umap_comparison_dashboard(dashboard_runs, comp_plot_path)

    # 4. Persist Summary JSON
    summary_json_path = results_dir / "umap_features_summary.json"
    summary_json_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    logger.info("Summary report saved → %s", summary_json_path)

    print("\n" + "=" * 80)
    print("UMAP FEATURE SEPARATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model / Feature Source':28s} {'Dim':6s} {'Silhouette ↑':14s} {'Davies-Bouldin ↓':18s} {'Calinski-Harabasz ↑':20s}")
    print("-" * 80)
    for name, data in summary_report.items():
        dim_str = str(data["feature_dim"])
        m = data["metrics"]
        sil = f"{m['silhouette_score']:.4f}"
        db = f"{m['davies_bouldin_index']:.4f}"
        ch = f"{m['calinski_harabasz_index']:.2f}"
        print(f"{name:28s} {dim_str:6s} {sil:14s} {db:18s} {ch:20s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
