"""High-level Training Runner and Metric Curves Plotter."""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.checkpoint_utils import resolve_run_files


class IMDBTrainer:
    """High-level class for executing training presets and plotting training curves."""

    PRESETS = {
        "EXP-BASE": {"config": "configs/config_imdb_sentiment_baseline.yaml", "run_name": "baseline_zero_shot"},
        "EXP-00": {"config": "configs/config_imdb_sentiment.yaml", "run_name": "distilbert-finetune-exp00"},
        "EXP-01": {"config": "configs/config_imdb_sentiment_tuned.yaml", "run_name": "distilbert-finetune-exp01"},
        "EXP-02": {"config": "configs/config_imdb_sentiment_tuned.yaml", "run_name": "distilbert-finetune-exp02"},
        "EXP-03": {"config": "configs/config_imdb_sentiment_tuned.yaml", "run_name": "distilbert-finetune-exp03"},
        "EXP-04": {"config": "configs/config_imdb_sentiment_tuned.yaml", "run_name": "distilbert-finetune-exp04"},
        "EXP-05": {"config": "configs/config_imdb_sentiment_tuned.yaml", "run_name": "distilbert-finetune-exp05"},
        "EXP-06": {"config": "configs/config_imdb_sentiment_512.yaml", "run_name": "distilbert-finetune-512"},
        "EXP-07": {"config": "configs/config_imdb_sentiment_512_hyper.yaml", "run_name": "distilbert-finetune-512-hyper"},
        "EXP-LORA": {"config": "configs/config_imdb_sentiment_lora.yaml", "run_name": "distilbert-finetune-lora"},
    }


    def __init__(self, preset_name: str = "EXP-06", project_root: str | Path | None = None) -> None:
        self.preset_name = preset_name
        self.project_root = Path(project_root).resolve() if project_root else Path(".").resolve()
        if self.project_root.name == "notebooks":
            self.project_root = self.project_root.parent

        selected = self.PRESETS.get(preset_name, self.PRESETS["EXP-06"])
        self.config_path = selected["config"]
        self.run_name = selected["run_name"]

    def run_training(self) -> None:
        """Execute the training process for selected experiment preset."""
        print("=" * 65)
        print(f" LAUNCHING TRAINING PRESET: {self.preset_name}")
        print(f" Config File: {self.config_path}")
        print(f" Run Name:    {self.run_name}")
        print("=" * 65)

        cmd = [
            sys.executable,
            "-m",
            "src.training.imdb_sentiment_train",
            "--config",
            self.config_path,
            "--run-name",
            self.run_name,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        process = subprocess.Popen(cmd, cwd=str(self.project_root), env=env)
        process.communicate()

        if process.returncode == 0:
            print(f"\n[SUCCESS] Training for {self.preset_name} completed successfully!")
        else:
            print(f"\n[ERROR] Training process exited with code {process.returncode}")

    @staticmethod
    def plot_latest_training_curves(run_root: str | Path = "experiments/runs") -> None:
        """Plot dual-axis Loss and Accuracy/Macro F1 curves for the newest run."""
        root_path = Path(run_root)
        matches = [p for p in root_path.glob("*_*") if p.is_dir() and (p / "checkpoints").exists()]
        if not matches:
            print("No run directory found under", run_root)
            return

        run_dir = sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)[0]
        print("Latest run directory:", run_dir)

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
