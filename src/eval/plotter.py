"""Unified Visualization & Plotter Module for IMDB Pipeline Artifacts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.checkpoint_utils import resolve_run_files


class IMDBPlotter:
    """Centralized plotter class encapsulating all matplotlib figures and image artifact displays."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path(".").resolve()
        if self.project_root.name == "notebooks":
            self.project_root = self.project_root.parent

    def _load_finetuned_accuracy(self) -> float | None:
        """Finetuned test accuracy (%) from the committed eval artifact, if present."""
        eval_path = self.project_root / "experiments" / "results" / "imdb_sentiment_eval.json"
        if not eval_path.exists():
            return None
        try:
            with open(eval_path, encoding="utf-8") as f:
                acc = json.load(f)["result"]["accuracy"]
            return float(acc) * 100
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            return None

    def plot_baseline_benchmark(self, include_ex2: bool = False) -> None:
        """Plot Exercise 1 Baseline Accuracy, F1, and ROC-AUC benchmark charts."""
        baseline_path = self.project_root / "experiments" / "results" / "baseline_imdb_sentiment.json"
        if not baseline_path.exists():
            print("Baseline JSON artifact not found at", baseline_path)
            return

        with open(baseline_path, encoding="utf-8") as f:
            b_data = json.load(f)["evaluation"]

        maj_acc = b_data.get("majority_class_accuracy", 0.50) * 100
        zs_acc = b_data.get("zero_shot_accuracy", 0.8907) * 100
        roc_auc = b_data.get("zero_shot_roc_auc", 0.9587)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        # Panel 1: Pure Zero-Shot Baseline Performance vs Majority Floor
        if include_ex2:
            finetuned_acc = self._load_finetuned_accuracy()
            if finetuned_acc is not None:
                categories = ["Random / Majority Floor", "Zero-Shot Baseline (Ex 1)", "Finetuned Target (Ex 2)"]
                accuracies = [maj_acc, zs_acc, finetuned_acc]
                colors = ["#95a5a6", "#3498db", "#2ecc71"]
            else:
                # No finetuned eval artifact yet — show only the baseline bars.
                categories = ["Random / Majority Floor", "Zero-Shot Baseline (Ex 1)"]
                accuracies = [maj_acc, zs_acc]
                colors = ["#95a5a6", "#3498db"]
        else:
            categories = ["Random / Majority Floor", "Zero-Shot Baseline (Ex 1)"]
            accuracies = [maj_acc, zs_acc]
            colors = ["#95a5a6", "#3498db"]

        bars = ax1.bar(categories, accuracies, color=colors, width=0.55, edgecolor="black", linewidth=1)
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=11)
        ax1.set_title("Zero-Shot Baseline vs Majority Floor Accuracy", fontweight="bold", fontsize=12)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

        # Panel 2: Evaluation Metrics
        metrics = ["Zero-Shot Accuracy", "Zero-Shot ROC-AUC"]
        scores = [zs_acc / 100.0, roc_auc]
        colors2 = ["#3498db", "#9b59b6"]

        bars2 = ax2.barh(metrics[::-1], scores[::-1], color=colors2[::-1], height=0.45, edgecolor="black", linewidth=1)
        ax2.set_xlim(0, 1.1)
        ax2.set_xlabel("Score (0.0 to 1.0)", fontweight="bold", fontsize=11)
        ax2.set_title("Zero-Shot Model Performance Metrics", fontweight="bold", fontsize=12)
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        for bar, score in zip(bars2, scores[::-1], strict=False):
            xval = bar.get_width()
            ax2.text(xval + 0.005, bar.get_y() + bar.get_height()/2.0, f"{score:.4f}", ha="left", va="center", fontweight="bold", fontsize=10)

        plt.tight_layout()
        plt.show()

    def display_roc_curve(self, finetuned: bool = False) -> None:
        """Display saved ROC Curve PNG artifact."""
        plot_name = "imdb_finetuned_roc_curve.png" if finetuned else "baseline_zero_shot_roc_curve.png"
        roc_plot_path = self.project_root / "experiments" / "plots" / plot_name
        if roc_plot_path.exists():
            try:
                from IPython.display import Image, display

                display(Image(filename=str(roc_plot_path)))
            except ImportError:
                pass
        else:
            print("ROC Curve plot artifact not found at", roc_plot_path)

    def display_confusion_matrix(self, finetuned: bool = False) -> None:
        """Display saved Confusion Matrix PNG artifact."""
        plot_name = "imdb_finetuned_confusion_matrix.png" if finetuned else "baseline_zero_shot_confusion_matrix.png"
        cm_plot_path = self.project_root / "experiments" / "plots" / plot_name
        if cm_plot_path.exists():
            try:
                from IPython.display import Image, display

                display(Image(filename=str(cm_plot_path)))
            except ImportError:
                pass
        else:
            print("Confusion Matrix plot artifact not found at", cm_plot_path)

    def plot_training_curves(self, run_root: str | Path | None = None) -> None:
        """Plot dual-panel Loss and Validation Accuracy/Macro F1 curves from newest run."""
        root_path = Path(run_root) if run_root else (self.project_root / "experiments" / "runs")
        matches = [p for p in root_path.glob("*_*") if p.is_dir() and (p / "checkpoints").exists()]
        if not matches:
            print("No fine-tuning run directory found under", root_path)
            return

        run_dir = sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)[0]
        files = resolve_run_files(run_dir)
        history_jsonl = files["history"]
        history = []
        if history_jsonl and history_jsonl.exists():
            with open(history_jsonl, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))

        if history:
            epochs = [r["epoch"] for r in history if "eval_loss" in r]
            train_losses = [r["train_loss"] for r in history if "eval_loss" in r]
            eval_losses = [r["eval_loss"] for r in history if "eval_loss" in r]
            eval_accuracies = [r.get("eval_accuracy", 0) * 100 for r in history if "eval_loss" in r]
            eval_f1s = [r.get("eval_f1", 0) for r in history if "eval_loss" in r]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(epochs, train_losses, "o-", label="Train Loss", color="#e74c3c", linewidth=2)
            ax1.plot(epochs, eval_losses, "s-", label="Validation Loss", color="#3498db", linewidth=2)
            ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Epoch", fontweight="bold")
            ax1.set_ylabel("Cross-Entropy Loss", fontweight="bold")
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend()

            ax2.plot(epochs, eval_accuracies, "^-", label="Validation Accuracy (%)", color="#2ecc71", linewidth=2)
            ax2.set_title("Validation Accuracy & F1 Trajectory", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch", fontweight="bold")
            ax2.set_ylabel("Accuracy (%)", fontweight="bold", color="#2ecc71")
            ax2.tick_params(axis="y", labelcolor="#2ecc71")
            ax2.grid(True, linestyle="--", alpha=0.6)

            ax2_f1 = ax2.twinx()
            ax2_f1.plot(epochs, eval_f1s, "d--", label="Validation Macro F1", color="#e67e22", linewidth=2)
            ax2_f1.set_ylabel("Macro F1 Score", fontweight="bold", color="#e67e22")
            ax2_f1.tick_params(axis="y", labelcolor="#e67e22")

            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_f1.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

            plt.tight_layout()
            plt.show()

    def display_metrics_summary_heatmap(self) -> None:
        """Display saved Seaborn Metrics Summary Heatmap PNG artifact."""
        heatmap_path = self.project_root / "experiments" / "plots" / "metrics_summary_heatmap.png"
        if heatmap_path.exists():
            try:
                from IPython.display import Image, display

                display(Image(filename=str(heatmap_path)))
            except ImportError:
                pass
        else:
            print("Metrics summary heatmap plot artifact not found at", heatmap_path)

    def display_zero_shot_baseline_artifacts(self) -> None:
        """Helper to display Exercise 1 Baseline benchmark, ROC curve, baseline confusion matrix & metrics summary heatmap."""
        self.plot_baseline_benchmark(include_ex2=False)
        self.display_roc_curve(finetuned=False)
        self.display_confusion_matrix(finetuned=False)
        self.display_metrics_summary_heatmap()

    def display_all_baseline_artifacts(self) -> None:
        """Legacy helper alias for zero-shot baseline artifacts."""
        self.display_zero_shot_baseline_artifacts()
