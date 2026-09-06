"""Data-Centric Confident Learning & Label Error Auditor via Cleanlab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.lab3.data.prepare_imdb import clean_text
from src.lab3.models.model_builder import build_model
from src.lab3.utils.checkpoint_utils import resolve_run_files, safe_load_checkpoint
from src.lab3.utils.resource_monitor import cleanup_vram


class IMDBCleanlabAuditor:
    """High-level Data-Centric AI Auditor using Confident Learning (Cleanlab).

    Detects label noise, mislabeled examples, and anomalous reviews in the
    IMDB training split without violating the sealed benchmark test split.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        processed_dir: str = "data/lab3/processed/imdb_tokenized_512",
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if project_root is None:
            cwd = Path(".").resolve()
            self.project_root = cwd.parent if cwd.name == "notebooks" else cwd
        else:
            self.project_root = Path(project_root)

        self.processed_dir = self.project_root / processed_dir
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _resolve_highest_accuracy_checkpoint(self) -> tuple[Path, float, str]:
        """Find the checkpoint with the highest validation accuracy across all runs in experiments/runs."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            return self.checkpoint_path, 1.0, "explicit_user_override"

        runs_dir = self.project_root / "experiments" / "lab3" / "runs"
        matches = [p for p in runs_dir.glob("*_*") if p.is_dir() and (p / "checkpoints").exists()]
        if not matches:
            raise FileNotFoundError(f"No run directory found under {runs_dir}")

        best_ckpt: Path | None = None
        best_acc: float = -1.0
        best_run_name: str = ""

        for run_dir in matches:
            files = resolve_run_files(run_dir)
            ckpt = files.get("swa") or files.get("best") or files.get("last")
            if not ckpt or not ckpt.exists():
                continue

            acc: float = 0.0
            history_files = list(run_dir.glob("metrics/*_history.jsonl"))
            if history_files:
                try:
                    with open(history_files[0], encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            eval_acc = row.get("eval_accuracy", 0.0)
                            if eval_acc and eval_acc > acc:
                                acc = eval_acc
                except Exception:
                    pass

            if acc > best_acc:
                best_acc = acc
                best_ckpt = ckpt
                best_run_name = run_dir.name

        if best_ckpt is None or not best_ckpt.exists():
            # Fallback to newest run if no history file found
            latest_dir = sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)[0]
            files = resolve_run_files(latest_dir)
            best_ckpt = files["swa"] or files["best"] or files["last"]
            best_run_name = latest_dir.name
            best_acc = 0.0

        return best_ckpt, best_acc, best_run_name

    def _load_latest_model(self) -> tuple[torch.nn.Module, AutoTokenizer, dict[str, Any]]:
        """Resolve highest-accuracy checkpoint and initialize model with LoRA/Full-FT support."""
        ckpt_path, best_acc, run_name = self._resolve_highest_accuracy_checkpoint()

        if not ckpt_path or not ckpt_path.exists():
            raise FileNotFoundError(f"No valid checkpoint found: {ckpt_path}")

        print(f"[CLEANLAB] 🏆 Using Highest-Accuracy Checkpoint: {ckpt_path.name} from {run_name} (Val Acc: {best_acc:.2%})")

        ckpt = safe_load_checkpoint(ckpt_path, device="cpu")
        run_cfg = ckpt.get("config", {})
        model_cfg = run_cfg.get("model", {})

        run_data_cfg = run_cfg.get("data") or {}
        run_max_length = int(run_data_cfg.get("max_length") or 512)
        if run_max_length < 512:
            print(
                f"[WARN] Checkpoint from {run_name} was trained with max_length={run_max_length} "
                f"(not 512) — audit tokenization will match this model but may diverge from the "
                f"head-tail 512 convention used for the denoised export."
            )

        model_name = model_cfg.get("name", "distilbert-base-uncased")
        state = ckpt["model_state_dict"]
        is_lora = bool(run_cfg.get("lora")) or any(
            k.startswith("base_model.model.") or k.startswith("lora_") for k in state
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = build_model(
            model_name=model_name,
            num_labels=2,
            classifier_dropout=0.0,
            use_lora=is_lora,
            lora_config_dict=run_cfg.get("lora"),
        )
        model.load_state_dict(state)
        model.to(self.device).eval()

        return model, tokenizer, run_cfg

    @torch.no_grad()
    def compute_train_probabilities(
        self,
        batch_size: int = 32,
        max_samples: int | None = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Run batch inference on train split to compute predicted probabilities.

        Features smart cache synchronization: automatically validates cache against the highest-accuracy checkpoint.
        """
        cache_file = self.project_root / "experiments" / "lab3" / "results" / "imdb_train_pred_probs.npy"
        cache_meta = self.project_root / "experiments" / "lab3" / "results" / "imdb_train_cache_meta.json"

        raw_full = load_dataset("stanfordnlp/imdb", split="train")
        raw = raw_full.train_test_split(test_size=0.1, seed=42)["train"]
        if max_samples and max_samples < len(raw):
            raw = raw.shuffle(seed=42).select(range(max_samples))

        labels = np.array(raw["label"])
        texts = list(raw["text"])

        # Check highest-accuracy model checkpoint path and modification time
        ckpt_path, best_acc, run_name = self._resolve_highest_accuracy_checkpoint()
        current_ckpt_str = str(ckpt_path) if ckpt_path else ""
        current_ckpt_mtime = ckpt_path.stat().st_mtime if (ckpt_path and ckpt_path.exists()) else 0

        # Smart Cache check: Validate cache against highest-accuracy model checkpoint
        if use_cache and cache_file.exists() and cache_meta.exists() and max_samples is None:
            try:
                with open(cache_meta, encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    meta.get("checkpoint_path") == current_ckpt_str
                    and meta.get("checkpoint_mtime") == current_ckpt_mtime
                    and meta.get("total_samples") == len(labels)
                    and meta.get("truncation") == "head_tail"
                ):
                    print(f"[CACHE] Loading precomputed probabilities from {cache_file}")
                    print(f"[CACHE] Synced with model: {Path(current_ckpt_str).name} ({Path(current_ckpt_str).parent.parent.name})")
                    probs = np.load(cache_file)
                    if len(probs) == len(labels):
                        return probs, labels, texts
                else:
                    print(f"[CLEANLAB] 🔄 Model checkpoint changed or upgraded! Automatically invalidating cache and recomputing with highest accuracy model ({run_name})...")
            except Exception:
                pass

        from src.lab3.data.prepare_imdb import head_tail_tokenize

        cleanup_vram()
        model, tokenizer, run_cfg = self._load_latest_model()

        # Tokenize with the exact head-tail scheme the model was trained on,
        # using the checkpoint's own max_length (512 for the denoised preset).
        max_length = int((run_cfg.get("data") or {}).get("max_length") or 512)
        if max_length < 256:
            max_length = 512  # head-tail truncation only applies at >= 256 tokens

        all_probs: list[np.ndarray] = []
        n = len(texts)
        print(
            f"[CLEANLAB] Computing predicted probabilities across {n:,} samples on "
            f"{self.device} (max_length={max_length}, head-tail truncation)..."
        )

        for i in range(0, n, batch_size):
            batch_texts = [clean_text(t) for t in texts[i : min(i + batch_size, n)]]
            enc_dict = head_tail_tokenize(batch_texts, tokenizer, max_length=max_length, head_ratio=0.25)
            enc = {
                "input_ids": torch.tensor(enc_dict["input_ids"], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor(enc_dict["attention_mask"], dtype=torch.long).to(self.device),
            }

            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)

        pred_probs = np.concatenate(all_probs, axis=0)

        if max_samples is None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, pred_probs)
            with open(cache_meta, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_samples": n,
                        "device": str(self.device),
                        "checkpoint_path": current_ckpt_str,
                        "checkpoint_mtime": current_ckpt_mtime,
                        "truncation": "head_tail",
                    },
                    f,
                    indent=2,
                )

        return pred_probs, labels, texts

    def compute_oof_probabilities(
        self,
        n_splits: int = 5,
        epochs: int = 2,
        batch_size: int = 16,
        lr: float = 3.0e-4,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Compute True 5-Fold Cross-Validation Out-Of-Fold (OOF) predicted probabilities.

        Zero-Memorization Guarantee:
        Splits the 22,500 train reviews into K stratified folds. For each fold, a fresh
        LoRA adapter is trained on (K-1)/K of the data and predicts softmax probabilities on
        the held-out 1/K validation fold. Samples are NEVER evaluated by a model that saw them during training.
        """
        from datetime import datetime

        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader

        from src.lab3.data.prepare_imdb import prepare_imdb

        cache_file = self.project_root / "experiments" / "lab3" / "results" / "imdb_oof_train_pred_probs.npy"
        cache_meta = self.project_root / "experiments" / "lab3" / "results" / "imdb_oof_train_cache_meta.json"

        base_ds, tokenizer, _ = prepare_imdb(
            dataset_id="stanfordnlp/imdb",
            model_name="distilbert-base-uncased",
            max_length=512,
            processed_dir=str(self.processed_dir),
        )

        train_ds = base_ds["train"]
        labels = np.array(train_ds["label"])

        # Load raw texts if available
        raw_full = load_dataset("stanfordnlp/imdb", split="train")
        raw_train = raw_full.train_test_split(test_size=0.1, seed=42)["train"]
        texts = list(raw_train["text"])
        n_samples = len(labels)

        if use_cache and cache_file.exists() and cache_meta.exists():
            try:
                with open(cache_meta, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("n_splits") == n_splits and meta.get("total_samples") == n_samples:
                    print(f"[OOF-CACHE] Loading precomputed 5-Fold Out-Of-Fold probabilities from {cache_file}")
                    oof_probs = np.load(cache_file)
                    if len(oof_probs) == n_samples:
                        return oof_probs, labels, texts
            except Exception:
                pass

        print("\n" + "=" * 65)
        print(f" 🚀 INITIATING {n_splits}-FOLD OUT-OF-FOLD (OOF) CONFIDENT LEARNING AUDIT")
        print("=" * 65)
        print(f" Total Train Samples:     {n_samples:,}")
        print(f" Folds:                   {n_splits} ({n_samples // n_splits:,} samples per validation slice)")
        print(f" LoRA Training Budget:    {epochs} epochs per fold | lr={lr}")
        print(" Zero-Memorization Rule:  100% Guaranteed via Out-of-Fold Cross-Validation")
        print("=" * 65 + "\n")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros((n_samples, 2), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), labels)):
            print(f"\n--- [FOLD {fold + 1}/{n_splits}] Training on {len(train_idx):,} samples, Evaluating {len(val_idx):,} held-out samples ---")
            cleanup_vram()

            fold_train = train_ds.select(train_idx)
            fold_val = train_ds.select(val_idx)

            train_loader = DataLoader(fold_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(fold_val, batch_size=batch_size * 2, shuffle=False)

            fold_model = build_model(
                model_name="distilbert-base-uncased",
                num_labels=2,
                classifier_dropout=0.20,
                use_lora=True,
                lora_config_dict={"r": 32, "lora_alpha": 64, "target_modules": ["q_lin", "k_lin", "v_lin", "out_lin"], "lora_dropout": 0.10},
            ).to(self.device)

            use_amp = self.device.type == "cuda"
            optimizer = torch.optim.AdamW(fold_model.parameters(), lr=lr, weight_decay=0.01)
            scaler = torch.amp.GradScaler("cuda") if use_amp else None

            fold_model.train()
            for epoch in range(epochs):
                running_loss = 0.0
                step_count = 0
                for batch in train_loader:
                    input_ids = batch["input_ids"].clone().detach().to(self.device) if isinstance(batch["input_ids"], torch.Tensor) else torch.tensor(batch["input_ids"]).to(self.device)
                    attention_mask = batch["attention_mask"].clone().detach().to(self.device) if isinstance(batch["attention_mask"], torch.Tensor) else torch.tensor(batch["attention_mask"]).to(self.device)
                    batch_labels = batch["label"].clone().detach().to(self.device) if isinstance(batch["label"], torch.Tensor) else torch.tensor(batch["label"]).to(self.device)

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                            loss = outputs.loss
                    else:
                        outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                        loss = outputs.loss

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    optimizer.zero_grad()

                    running_loss += loss.item()
                    step_count += 1

                avg_loss = running_loss / max(1, step_count)
                print(f"  [Fold {fold + 1} | Epoch {epoch + 1}/{epochs}] Train Loss: {avg_loss:.4f}")

            # Predict on held-out validation fold
            fold_model.eval()
            val_preds: list[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].clone().detach().to(self.device) if isinstance(batch["input_ids"], torch.Tensor) else torch.tensor(batch["input_ids"]).to(self.device)
                    attention_mask = batch["attention_mask"].clone().detach().to(self.device) if isinstance(batch["attention_mask"], torch.Tensor) else torch.tensor(batch["attention_mask"]).to(self.device)

                    if self.device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            logits = fold_model(input_ids=input_ids, attention_mask=attention_mask).logits
                    else:
                        logits = fold_model(input_ids=input_ids, attention_mask=attention_mask).logits

                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    val_preds.append(probs)

            oof_probs[val_idx] = np.concatenate(val_preds, axis=0)
            print(f"[FOLD {fold + 1}/{n_splits} DONE] Out-Of-Fold Softmax probabilities captured.")

            del fold_model, optimizer, scaler
            cleanup_vram()

        # Persist OOF probabilities cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, oof_probs)
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({
                "n_splits": n_splits,
                "epochs": epochs,
                "total_samples": n_samples,
                "date": datetime.now().isoformat(),
            }, f, indent=2)

        print(f"\n[OOF AUDIT COMPLETE] 5-Fold Out-Of-Fold probabilities saved to: {cache_file}")
        return oof_probs, labels, texts

    def audit_label_errors(
        self,
        pred_probs: np.ndarray,
        labels: np.ndarray,
        texts: list[str],
        output_json: str | Path = "experiments/lab3/results/cleanlab_label_issues.json",
    ) -> dict[str, Any]:
        """Perform Confident Learning audit via Cleanlab to detect label issues."""
        try:
            from cleanlab.filter import find_label_issues
        except ImportError as exc:
            raise ImportError("Please install cleanlab: pip install cleanlab") from exc

        print("[CLEANLAB] Executing Confident Learning analysis...")
        issue_indices = find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
            filter_by="prune_by_noise_rate",
        )

        num_issues = len(issue_indices)
        total = len(labels)
        noise_rate = (num_issues / total) * 100 if total > 0 else 0.0

        issues_details: list[dict[str, Any]] = []
        fp_count = 0  # Labeled 0 (neg) but model is confident it's pos (1)
        fn_count = 0  # Labeled 1 (pos) but model is confident it's neg (0)

        for idx in issue_indices:
            idx = int(idx)
            given_lbl = int(labels[idx])
            prob_neg = float(pred_probs[idx][0])
            prob_pos = float(pred_probs[idx][1])
            suggested_lbl = 1 if prob_pos > prob_neg else 0
            confidence = max(prob_pos, prob_neg) * 100

            if given_lbl == 0 and suggested_lbl == 1:
                fp_count += 1
            elif given_lbl == 1 and suggested_lbl == 0:
                fn_count += 1

            snippet = texts[idx][:250].replace("\n", " ").strip()
            issues_details.append(
                {
                    "dataset_index": idx,
                    "given_label": "Negative (0)" if given_lbl == 0 else "Positive (1)",
                    "suggested_label": "Positive (1)" if suggested_lbl == 1 else "Negative (0)",
                    "confidence_pct": round(confidence, 2),
                    "prob_positive": round(prob_pos, 4),
                    "prob_negative": round(prob_neg, 4),
                    "text_excerpt": snippet + ("..." if len(texts[idx]) > 250 else ""),
                }
            )

        results = {
            "total_samples_audited": total,
            "total_label_issues_found": num_issues,
            "estimated_noise_rate_pct": round(noise_rate, 2),
            "mislabeled_as_negative_count": fp_count,
            "mislabeled_as_positive_count": fn_count,
            "issue_indices": [int(i) for i in issue_indices],
            "top_issues": issues_details,
        }

        out_path = self.project_root / output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"[CLEANLAB] Audit complete: Found {num_issues:,} label issues ({noise_rate:.2f}%)")
        print(f"[CLEANLAB] Saved audit report to: {out_path}")
        return results

    def print_top_issues(self, audit_results: dict[str, Any], top_k: int = 3) -> None:
        """Pretty-print the top confident mislabeled reviews."""
        issues = audit_results.get("top_issues", [])[:top_k]
        print("\n" + "=" * 75)
        print(f" 🚨 TOP {len(issues)} CONFIDENT LABEL ERRORS DISCOVERED BY CLEANLAB")
        print("=" * 75)

        for i, item in enumerate(issues, 1):
            print(f"\n[{i}] Sample Index: #{item['dataset_index']}")
            print(f"    • Given Label in IMDB:    {item['given_label']} ❌ (Stanford Annotator Error)")
            print(f"    • Model Suggested Label:  {item['suggested_label']} ✅ (Confidence: {item['confidence_pct']}%)")
            print(f"    • Probabilities:          Pos={item['prob_positive']:.4f}, Neg={item['prob_negative']:.4f}")
            print(f"    • Review Text Excerpt:    \"{item['text_excerpt']}\"")
        print("-" * 75)

    def plot_audit_overview(
        self,
        audit_results: dict[str, Any],
        save_path: str | Path = "experiments/lab3/plots/cleanlab_label_audit.png",
    ) -> None:
        """Render a 3-panel dashboard visualizing Cleanlab label audit findings."""
        total = audit_results["total_samples_audited"]
        num_issues = audit_results["total_label_issues_found"]
        clean_count = total - num_issues
        fp = audit_results["mislabeled_as_negative_count"]
        fn = audit_results["mislabeled_as_positive_count"]
        top_issues = audit_results.get("top_issues", [])
        confidences = [item["confidence_pct"] for item in top_issues] if top_issues else [95.0]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.8))

        # Panel 1: Donut Chart - Clean vs Noisy Labels
        labels = [f"Clean Labels\n({clean_count:,})", f"Label Errors\n({num_issues:,})"]
        sizes = [clean_count, num_issues]
        colors = ["#2ecc71", "#e74c3c"]

        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=labels,
            autopct="%1.2f%%",
            startangle=140,
            colors=colors,
            explode=(0, 0.08) if num_issues > 0 else (0, 0),
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        )
        for t in texts:
            t.set_fontsize(11)
            t.set_fontweight("bold")
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
            at.set_color("white")
        ax1.set_title("Training Set Label Integrity", fontsize=12, fontweight="bold", pad=15)

        # Panel 2: Error Direction Breakdown
        categories = ["Mislabeled as Neg\n(True: Positive)", "Mislabeled as Pos\n(True: Negative)"]
        counts = [fp, fn]
        bars = ax2.bar(categories, counts, color=["#3498db", "#e67e22"], width=0.45, edgecolor="black")
        ax2.set_ylabel("Number of Identified Issues", fontsize=11, fontweight="bold")
        ax2.set_title("Label Error Direction Breakdown", fontsize=12, fontweight="bold", pad=15)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        max_h = max(counts) if counts and max(counts) > 0 else 10
        ax2.set_ylim(0, max_h * 1.3)
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + (max_h * 0.03),
                f"{h:,} reviews",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Panel 3: Confidence Distribution of Detected Errors
        if len(confidences) > 1:
            ax3.hist(confidences, bins=min(15, len(confidences)), color="#9b59b6", edgecolor="black", alpha=0.85)
        else:
            ax3.bar(["Avg. Confidence"], [np.mean(confidences) if confidences else 90.0], color="#9b59b6", width=0.4, edgecolor="black")
        ax3.set_xlabel("Model Confidence (%)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Number of Error Cases", fontsize=11, fontweight="bold")
        ax3.set_title("Confident Learning Detection Strength", fontsize=12, fontweight="bold", pad=15)
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        out = self.project_root / save_path
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()

    def run_audit_and_visualize(
        self,
        top_k: int = 5,
        max_samples: int | None = None,
        use_cache: bool = True,
        use_oof: bool = False,
        n_splits: int = 5,
        epochs: int = 2,
    ) -> dict[str, Any]:
        """Convenient one-line API for Jupyter Notebooks.

        Args:
            top_k (int): Number of top confident label errors to display.
            max_samples (int | None): Number of samples to scan. None = Full 22,500 train dataset.
            use_cache (bool): Whether to use cached probabilities if available.
            use_oof (bool): If True, executes 5-Fold Out-Of-Fold Cross-Validation for 100% zero-memorization confidence.
            n_splits (int): Number of cross-validation folds (default: 5).
            epochs (int): Number of epochs per fold (default: 2).
        """
        if use_oof:
            pred_probs, labels, texts = self.compute_oof_probabilities(
                n_splits=n_splits,
                epochs=epochs,
                use_cache=use_cache,
            )
        else:
            pred_probs, labels, texts = self.compute_train_probabilities(max_samples=max_samples, use_cache=use_cache)

        results = self.audit_label_errors(pred_probs, labels, texts)
        self.print_top_issues(results, top_k=top_k)
        self.plot_audit_overview(results)
        return results

    def export_denoised_dataset(
        self,
        issue_indices: list[int] | dict[str, Any],
        output_dir: str | Path = "data/lab3/processed/imdb_denoised_512",
    ) -> Path:
        """Filter out identified label issues from train set and save tokenized clean splits to disk."""
        import json

        from src.lab3.data.prepare_imdb import prepare_imdb

        out_path = self.project_root / output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        if isinstance(issue_indices, dict):
            raw_issues = issue_indices.get("issue_indices", [])
            idx_set = set(int(x) for x in raw_issues)
        else:
            idx_set = set(int(x) for x in issue_indices)

        base_ds, _, _ = prepare_imdb(
            dataset_id="stanfordnlp/imdb",
            model_name="distilbert-base-uncased",
            max_length=512,
            processed_dir=str(self.processed_dir),
        )

        train_len = len(base_ds["train"])
        removed_count = len(idx_set)
        clean_indices = [i for i in range(train_len) if i not in idx_set]

        clean_train = base_ds["train"].select(clean_indices)
        clean_val = base_ds["val"]
        clean_test = base_ds["test"]

        clean_train.save_to_disk(str(out_path / "train"))
        clean_val.save_to_disk(str(out_path / "val"))
        clean_test.save_to_disk(str(out_path / "test"))

        meta_info = {
            "dataset_id": "stanfordnlp/imdb",
            "model_name": "distilbert-base-uncased",
            "max_length": 512,
            "val_size": 0.1,
            "seed": 42,
            "splits": {
                "train": len(clean_train),
                "val": len(clean_val),
                "test": len(clean_test),
            },
            "denoised": True,
            "removed_noise_samples": removed_count,
        }
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)

        print("\n" + "=" * 65)
        print(" 🧹 CLEANLAB DENOISED DATASET EXPORTED SUCCESSFULLY")
        print("=" * 65)
        print(f"Original Train Samples: {train_len:,}")
        print(f"Removed Noisy Labels:   {removed_count:,} ({removed_count/train_len*100:.2f}%)")
        print(f"Clean Train Samples:    {len(clean_train):,} ({len(clean_train)/train_len*100:.2f}%)")
        print(f"Val & Test Splits:      100% Preserved ({len(clean_val):,} val / {len(clean_test):,} test)")
        print(f"Saved To Directory:     {out_path}")
        print("=" * 65 + "\n")

        return out_path
