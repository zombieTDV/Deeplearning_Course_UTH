"""Evaluate the Exercise 2 finetuned model on the held-out IMDB test set.

Runs inference exactly once on the sealed test split (golden rule 4, §10),
reports accuracy, ROC-AUC, confusion matrix, and per-class precision/recall/F1 with 5W1H
context, saves `experiments/results/imdb_sentiment_eval.json`, a normalized confusion
matrix plot, and an ROC curve plot.

Usage::

    python -m src.eval.evaluate_model --checkpoint experiments/runs/<ts>_<run>/checkpoints/<run>_best.pt
    python -m src.eval.evaluate_model    # automatically resolves latest best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# Prevent PyTorch c10 ApproximateClock non-monotonic CPU frequency assertion crash on Linux
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.preprocessing import label_binarize

from src.data.prepare_imdb import prepare_imdb
from src.utils.checkpoint_utils import safe_load_checkpoint
from src.utils.resource_monitor import ResourceMonitor, cleanup_vram


def _resolve_checkpoint(checkpoint: str | None, run_root: str) -> tuple[Path, dict[str, Any] | None]:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        return path, None

    run_root_path = Path(run_root)
    if not run_root_path.exists():
        raise SystemExit(f"Run root directory not found: {run_root}")

    # Find the newest timestamped run directory under experiments/runs/
    matches = [p for p in run_root_path.glob("*_*") if p.is_dir() and (p / "checkpoints").exists()]
    if not matches:
        raise SystemExit(f"No run directory found under {run_root}")

    run_dir = sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)[0]

    from src.utils.checkpoint_utils import resolve_run_files
    files = resolve_run_files(run_dir)
    target = files["swa"] or files["best"] or files["last"]
    if target and target.exists():
        cfg = None
        metrics_cfg = list((run_dir / "metrics").glob("*_config.json")) if (run_dir / "metrics").exists() else []
        if metrics_cfg:
            try:
                with open(metrics_cfg[0], encoding="utf-8") as f:
                    cfg = json.load(f).get("config")
            except (json.JSONDecodeError, OSError):
                cfg = None
        return target, cfg
    raise SystemExit(f"Best checkpoint not found under {run_dir / 'checkpoints'}")



@torch.no_grad()
def predict(
    model: Any,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_probs_pos: list[float] = []
    n = len(dataset)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_ids = dataset["input_ids"][i:end].to(device)
        batch_mask = dataset["attention_mask"][i:end].to(device)
        out = model(input_ids=batch_ids, attention_mask=batch_mask)
        logits = out.logits.detach().cpu().numpy()
        probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_logits.append(logits)
        all_probs_pos.extend(probs.tolist())
    logits = np.concatenate(all_logits, axis=0)
    preds = np.argmax(logits, axis=-1)
    return preds, logits, all_probs_pos


def evaluate(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    batch_size: int = 32,
    max_samples: int | None = None,
) -> dict[str, Any]:
    cleanup_vram()
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = safe_load_checkpoint(checkpoint_path, device="cpu")

    run_cfg = ckpt.get("config", {})
    model_cfg = run_cfg.get("model") or cfg.get("model") or {}
    model_name = model_cfg.get("name", "distilbert-base-uncased")
    data_cfg = run_cfg.get("data") or run_cfg.get("dataset") or cfg.get("data") or cfg.get("dataset") or {}
    max_length = data_cfg.get("max_length", 256)
    processed_dir = data_cfg.get("processed_dir", "data/processed/imdb_tokenized")
    lora_cfg = run_cfg.get("lora")

    from src.models.model_builder import build_model
    model = build_model(
        model_name=model_name,
        num_labels=model_cfg.get("num_labels", 2),
        id2label={int(k): v for k, v in model_cfg.get("id2label", {"0": "neg", "1": "pos"}).items()} if model_cfg.get("id2label") else None,
        label2id={k: int(v) for k, v in model_cfg.get("label2id", {"neg": 0, "pos": 1}).items()} if model_cfg.get("label2id") else None,
        lora_config_dict=lora_cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    monitor = ResourceMonitor()
    monitor.sample()

    ds, _, _ = prepare_imdb(
        dataset_id="stanfordnlp/imdb",
        model_name=model_name,
        max_length=max_length,
        processed_dir=processed_dir,
    )
    test: Dataset = ds["test"]
    if max_samples and max_samples < len(test):
        test = test.select(range(max_samples))

    y_true = [int(lbl) for lbl in test["label"]]
    preds, _, probs_pos = predict(model, test, device, batch_size)

    acc = accuracy_score(y_true, preds)
    auc_score = float(roc_auc_score(y_true, probs_pos))
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    per_class = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], average=None, zero_division=0
    )
    report = classification_report(
        y_true, preds, labels=[0, 1], target_names=["neg", "pos"], output_dict=True, zero_division=0
    )
    monitor.sample()

    result = {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "num_test_samples": len(test),
        "accuracy": float(acc),
        "roc_auc": auc_score,
        "confusion_matrix": {
            "neg_neg": int(cm[0, 0]),
            "neg_pos": int(cm[0, 1]),
            "pos_neg": int(cm[1, 0]),
            "pos_pos": int(cm[1, 1]),
        },
        "per_class": {
            "neg": {
                "precision": float(per_class[0][0]),
                "recall": float(per_class[1][0]),
                "f1": float(per_class[2][0]),
                "support": int(per_class[3][0]),
            },
            "pos": {
                "precision": float(per_class[0][1]),
                "recall": float(per_class[1][1]),
                "f1": float(per_class[2][1]),
                "support": int(per_class[3][1]),
            },
        },
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "vram": monitor.summary(),
    }

    print("\n" + "=" * 65)
    print(" EVALUATION RESULTS (IMDB Sealed Test Split)")
    print("=" * 65)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"ROC-AUC:   {auc_score:.4f}")
    print(f"Macro F1:  {result['f1_macro']:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(f"Per-class P/R/F1: {result['per_class']}")
    print("=" * 65)

    # Persist results with 5W1H metadata.
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "imdb_sentiment_eval.json"
    payload = {
        "metadata_5w1h": {
            "what": "Finetuned DistilBERT test-set evaluation (accuracy, ROC-AUC, confusion matrix, per-class P/R/F1)",
            "why": "Generalization benchmark on sealed IMDB test split vs Exercise 1 zero-shot floor",
            "when": datetime.now().isoformat(),
            "where": f"{out_file} | executed on {device}",
            "who": "bush-le + AI agent — coursework submission",
            "how": "single held-out evaluation pass, softmax probability output, no TTA",
        },
        "result": result,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved evaluation results to: {out_file}")

    # Publication-Grade Confusion Matrix Plot
    plots_dir = Path("experiments/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], ["Negative (0)", "Positive (1)"], fontsize=11)
    ax.set_yticks([0, 1], ["Negative (0)", "Positive (1)"], fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        f"DistilBERT Finetuned Confusion Matrix\nAccuracy: {acc * 100:.2f}% | Macro F1: {result['f1_macro']:.4f}",
        fontsize=12,
        pad=12,
    )

    for i in range(2):
        for j in range(2):
            count_str = f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)"
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, count_str, ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plot_path = plots_dir / "imdb_finetuned_confusion_matrix.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion-matrix plot to: {plot_path}")

    # Publication-Grade ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, probs_pos)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Finetuned DistilBERT (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    roc_plot_path = plots_dir / "imdb_finetuned_roc_curve.png"
    plt.savefig(roc_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve plot to: {roc_plot_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate finetuned IMDB sentiment model")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to <run>_best.pt")
    parser.add_argument("--run-root", default="experiments/runs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None, help="cap for quick smoke tests")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    ckpt_path, saved_cfg = _resolve_checkpoint(args.checkpoint, args.run_root)

    if saved_cfg is not None and not args.config:
        cfg = saved_cfg
    else:
        import yaml
        config_file = args.config or "configs/config_imdb_sentiment.yaml"
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    evaluate(ckpt_path, cfg, batch_size=args.batch_size, max_samples=args.max_samples)



if __name__ == "__main__":
    main()


def compute_model_metrics(
    y_true: np.ndarray | list[int],
    probs: np.ndarray,
    num_classes: int = 2,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for a single model.

    Computes:
        - Overall Top-1 Accuracy (%)
        - Empirical Cross-Entropy Loss
        - Macro-averaged Precision, Recall, and F1 (%)
        - Weighted-averaged F1 (%)
        - One-vs-Rest (OvR) Macro & Micro ROC-AUC
        - Per-class accuracy (%)
        - Confusion matrix

    Args:
        y_true: Ground-truth integer class labels of shape (N,).
        probs: Predicted class probability distribution of shape (N, num_classes) or (N,).
        num_classes: Number of target classes (default 2 for binary classification).

    Returns:
        Dictionary mapping metric names to computed float/list values.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(probs, dtype=float)
    if p_arr.ndim == 1:
        p_arr = np.column_stack([1.0 - p_arr, p_arr])

    num_classes = p_arr.shape[1] if p_arr.ndim == 2 else num_classes
    preds = np.argmax(p_arr, axis=1)

    acc = float(np.mean(preds == y_arr) * 100.0)
    true_probs = np.clip(p_arr[np.arange(len(y_arr)), y_arr], 1e-15, 1.0)
    loss = float(-np.mean(np.log(true_probs)))

    macro_prec = float(precision_score(y_arr, preds, average="macro", zero_division=0) * 100.0)
    macro_rec = float(recall_score(y_arr, preds, average="macro", zero_division=0) * 100.0)
    macro_f1 = float(f1_score(y_arr, preds, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_arr, preds, average="weighted", zero_division=0) * 100.0)

    y_bin = label_binarize(y_arr, classes=list(range(num_classes)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    try:
        macro_auc = float(roc_auc_score(y_bin, p_arr, multi_class="ovr", average="macro"))
        micro_auc = float(roc_auc_score(y_bin, p_arr, multi_class="ovr", average="micro"))
    except Exception:
        macro_auc, micro_auc = float("nan"), float("nan")

    cm = sk_cm(y_arr, preds, labels=list(range(num_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    per_class = (cm.diagonal() / row_sums.squeeze() * 100.0).tolist()

    return {
        "test_loss": loss,
        "test_acc": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "per_class_acc": per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_comprehensive_benchmark(
    models_probs_dict: dict[str, np.ndarray],
    y_true: np.ndarray | list[int],
    num_classes: int = 2,
) -> dict[str, dict[str, Any]]:
    """Compute comprehensive benchmark metrics across multiple model variants.

    Args:
        models_probs_dict: Dictionary mapping model variant names to probability arrays.
        y_true: Ground-truth target labels.
        num_classes: Total class count.

    Returns:
        Nested dictionary mapping model names to their comprehensive metric dictionaries.
    """
    benchmark_dict: dict[str, dict[str, Any]] = {}
    for name, probs in models_probs_dict.items():
        benchmark_dict[name] = compute_model_metrics(y_true, probs, num_classes=num_classes)
    return benchmark_dict


def format_ascii_metrics_table(
    benchmark_metrics: dict[str, dict[str, Any]],
) -> str:
    """Format benchmark metrics into an aligned, publication-grade ASCII table.

    Includes:
        Model, Test Loss, Accuracy (%), Macro Precision (%), Macro Recall (%),
        Macro F1 (%), Weighted F1 (%), Macro ROC-AUC, Micro ROC-AUC.
    """
    sep = (
        "+" + "-" * 32 + "+" + "-" * 11 + "+" + "-" * 11 + "+"
        + "-" * 13 + "+" + "-" * 12 + "+" + "-" * 11 + "+"
        + "-" * 13 + "+" + "-" * 11 + "+" + "-" * 11 + "+"
    )
    header = (
        f"| {'Model':<30s} | {'Test Loss':<9s} | {'Accuracy':<9s} | "
        f"{'Macro Prec':<11s} | {'Macro Rec':<10s} | {'Macro F1':<9s} | "
        f"{'Weighted F1':<11s} | {'Macro AUC':<9s} | {'Micro AUC':<9s} |"
    )

    lines: list[str] = [sep, header, sep]
    for name, res in benchmark_metrics.items():
        tl = res.get("test_loss", float("nan"))
        ta = res.get("test_acc", float("nan"))
        mp = res.get("macro_precision", float("nan"))
        mr = res.get("macro_recall", float("nan"))
        mf1 = res.get("macro_f1", float("nan"))
        wf1 = res.get("weighted_f1", float("nan"))
        mauc = res.get("macro_auc", float("nan"))
        uauc = res.get("micro_auc", float("nan"))

        lines.append(
            f"| {name:<30s} | {tl:>9.4f} | {ta:>8.2f}% | "
            f"{mp:>10.2f}% | {mr:>9.2f}% | {mf1:>8.2f}% | "
            f"{wf1:>10.2f}% | {mauc:>9.4f} | {uauc:>9.4f} |"
        )
    lines.append(sep)
    return "\n".join(lines)


def plot_metrics_heatmap_table(
    benchmark_metrics: dict[str, dict[str, Any]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (13, 6),
    title: str = "Comprehensive Classification Benchmark Metrics (IMDB Test Set)",
) -> plt.Figure:
    """Render a Seaborn color-coded summary table heatmap across all evaluated models.

    Column metrics are normalized to visually highlight rank performance (1.0 = best),
    with individual cells annotated with exact formatted score values.

    Args:
        benchmark_metrics: Dictionary mapping model names to metric dictionaries.
        save_path: Optional file path to persist the generated figure.
        figsize: Figure dimensions (width, height).
        title: Plot super title.

    Returns:
        The generated Matplotlib Figure instance.
    """
    models = list(benchmark_metrics.keys())
    metric_cols = [
        "Test Loss", "Accuracy (%)", "Macro Prec (%)", "Macro Rec (%)",
        "Macro F1 (%)", "Weighted F1 (%)", "Macro AUC", "Micro AUC",
    ]
    keys = [
        "test_loss", "test_acc", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "macro_auc", "micro_auc",
    ]

    raw_vals = np.array(
        [[benchmark_metrics[m].get(k, 0.0) for k in keys] for m in models],
        dtype=float,
    )

    annot_matrix = []
    for m in models:
        res = benchmark_metrics[m]
        row = [
            f"{res.get('test_loss', 0.0):.4f}",
            f"{res.get('test_acc', 0.0):.2f}%",
            f"{res.get('macro_precision', 0.0):.2f}%",
            f"{res.get('macro_recall', 0.0):.2f}%",
            f"{res.get('macro_f1', 0.0):.2f}%",
            f"{res.get('weighted_f1', 0.0):.2f}%",
            f"{res.get('macro_auc', 0.0):.4f}",
            f"{res.get('micro_auc', 0.0):.4f}",
        ]
        annot_matrix.append(row)
    annot_matrix = np.array(annot_matrix)

    # Column-wise normalization (higher is better, test_loss inverted)
    norm_vals = np.zeros_like(raw_vals, dtype=float)
    for c in range(raw_vals.shape[1]):
        col = raw_vals[:, c]
        rng = col.max() - col.min()
        denom = rng if rng > 1e-12 else 1.0
        if keys[c] == "test_loss":
            norm_vals[:, c] = (col.max() - col) / denom
        else:
            norm_vals[:, c] = (col - col.min()) / denom

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        norm_vals,
        annot=annot_matrix,
        fmt="",
        cmap="YlGnBu",
        xticklabels=metric_cols,
        yticklabels=models,
        cbar=True,
        cbar_kws={"label": "Normalized Performance Rank (Column-wise: 1.0 = Peak SOTA)"},
        linewidths=1.2,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 9.5, "fontweight": "bold"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=300, bbox_inches="tight")

    return fig

