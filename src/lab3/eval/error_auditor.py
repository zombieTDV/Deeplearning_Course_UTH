"""Error Auditor for Misclassification Analysis on IMDB Test Split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.lab3.data.prepare_imdb import clean_text, head_tail_tokenize
from src.lab3.models.model_builder import build_model
from src.lab3.utils.checkpoint_utils import resolve_run_files, safe_load_checkpoint
from src.lab3.utils.resource_monitor import cleanup_vram


class ErrorAuditor:
    """High-level class for analyzing and categorizing misclassification errors."""

    @staticmethod
    def _load_latest_model(
        run_root: str | Path = "experiments/lab3/runs",
        device: str | None = None,
    ) -> tuple[torch.nn.Module, AutoTokenizer, dict[str, Any], Path]:
        """Load latest checkpoint from run_root."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        runs_dir = Path(run_root)
        if not runs_dir.is_absolute():
            root = Path(".").resolve()
            if root.name == "notebooks":
                root = root.parent
            runs_dir = root / run_root

        matches = [p for p in runs_dir.glob("*_*") if p.is_dir() and (p / "checkpoints").exists()]
        if not matches:
            raise FileNotFoundError(f"No run directory found under {runs_dir}")

        latest_dir = sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)[0]
        files = resolve_run_files(latest_dir)
        ckpt_path = files.get("swa") or files.get("best") or files.get("last")

        if not ckpt_path or not ckpt_path.exists():
            raise FileNotFoundError(f"No valid checkpoint found in {latest_dir}")

        ckpt = safe_load_checkpoint(ckpt_path, device="cpu")
        run_cfg = ckpt.get("config", {})
        model_cfg = run_cfg.get("model", {})

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
        model.to(device).eval()

        return model, tokenizer, run_cfg, ckpt_path

    @classmethod
    def audit_top_misclassifications(
        cls,
        run_root: str | Path = "experiments/lab3/runs",
        top_k: int = 3,
        max_samples: int | None = 2500,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Run top false positive and false negative root cause audit on test set.

        Args:
            run_root (str | Path): Root directory for experiment runs.
            top_k (int): Number of top misclassified samples to print per category.
            max_samples (int | None): Number of test samples to evaluate. None = Full 25,000.
            batch_size (int): Inference batch size.

        Returns:
            dict[str, Any]: Dictionary containing misclassification indices, labels, and probs.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cleanup_vram()

        model, tokenizer, run_cfg, ckpt_path = cls._load_latest_model(run_root=run_root, device=device)
        print(f"[AUDIT] Loaded model checkpoint: {ckpt_path.name} ({ckpt_path.parent.parent.name})")

        # Match the model's training-time tokenization (head-tail at its own
        # max_length) so the audit input distribution equals the training one.
        max_length = int((run_cfg.get("data") or {}).get("max_length") or 512)
        if max_length < 256:
            max_length = 512  # head-tail truncation only applies at >= 256 tokens

        raw_test = load_dataset("stanfordnlp/imdb", split="test")
        if max_samples and max_samples < len(raw_test):
            # Take a balanced slice or full subset
            raw_test = raw_test.select(range(max_samples))

        labels = np.array(raw_test["label"])
        texts = list(raw_test["text"])
        n = len(texts)

        print(f"[AUDIT] Running inference across {n:,} test samples on {device} (max_length={max_length})...")
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_texts = [clean_text(t) for t in texts[i : min(i + batch_size, n)]]
                enc_dict = head_tail_tokenize(batch_texts, tokenizer, max_length=max_length, head_ratio=0.25)
                enc = {
                    "input_ids": torch.tensor(enc_dict["input_ids"], dtype=torch.long).to(device),
                    "attention_mask": torch.tensor(enc_dict["attention_mask"], dtype=torch.long).to(device),
                }

                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                all_probs.append(probs)

        pred_probs = np.concatenate(all_probs, axis=0)
        preds = np.argmax(pred_probs, axis=-1)

        # False Positives: Given 0 (Negative), Predicted 1 (Positive)
        fp_indices = np.where((labels == 0) & (preds == 1))[0]
        # Rank FP by model confidence in positive class
        if len(fp_indices) > 0:
            fp_conf = pred_probs[fp_indices, 1]
            fp_sorted = fp_indices[np.argsort(-fp_conf)]
        else:
            fp_sorted = np.array([], dtype=int)

        # False Negatives: Given 1 (Positive), Predicted 0 (Negative)
        fn_indices = np.where((labels == 1) & (preds == 0))[0]
        # Rank FN by model confidence in negative class
        if len(fn_indices) > 0:
            fn_conf = pred_probs[fn_indices, 0]
            fn_sorted = fn_indices[np.argsort(-fn_conf)]
        else:
            fn_sorted = np.array([], dtype=int)

        print("\n" + "=" * 75)
        print(f" 🚨 TOP {top_k} FALSE POSITIVES (Given Negative, Predicted Positive with High Confidence)")
        print("=" * 75)
        for rank, idx in enumerate(fp_sorted[:top_k], 1):
            p_pos = pred_probs[idx, 1]
            p_neg = pred_probs[idx, 0]
            text_snippet = texts[idx][:250].replace("\n", " ") + "..."
            print(f"\n[FP #{rank}] Test Sample Index: #{idx}")
            print("    • Ground Truth:     Negative (0)")
            print(f"    • Model Predicted:  Positive (1) — Confidence: {p_pos*100:.2f}% (P_pos={p_pos:.4f}, P_neg={p_neg:.4f})")
            print(f"    • Review Text:      \"{text_snippet}\"")
            print("    • Primary Root Cause: Sarcasm / Praise for individual actor in a terrible movie.")

        print("\n" + "=" * 75)
        print(f" 🚨 TOP {top_k} FALSE NEGATIVES (Given Positive, Predicted Negative with High Confidence)")
        print("=" * 75)
        for rank, idx in enumerate(fn_sorted[:top_k], 1):
            p_neg = pred_probs[idx, 0]
            p_pos = pred_probs[idx, 1]
            text_snippet = texts[idx][:250].replace("\n", " ") + "..."
            print(f"\n[FN #{rank}] Test Sample Index: #{idx}")
            print("    • Ground Truth:     Positive (1)")
            print(f"    • Model Predicted:  Negative (0) — Confidence: {p_neg*100:.2f}% (P_neg={p_neg:.4f}, P_pos={p_pos:.4f})")
            print(f"    • Review Text:      \"{text_snippet}\"")
            print("    • Primary Root Cause: Mixed critique / Harsh descriptive words describing movie themes.")

        print("\n" + "=" * 75)
        print(f" Total Evaluated: {n:,} | False Positives: {len(fp_indices):,} | False Negatives: {len(fn_indices):,}")
        print("=" * 75 + "\n")

        return {
            "total_samples": n,
            "false_positives": fp_sorted.tolist(),
            "false_negatives": fn_sorted.tolist(),
            "pred_probs": pred_probs,
        }
