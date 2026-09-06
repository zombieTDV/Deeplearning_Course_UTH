"""Interactive Sentiment Predictor and Inference Engine."""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.lab3.models.model_builder import build_model
from src.lab3.utils.checkpoint_utils import (
    average_checkpoints,
    resolve_run_files,
    safe_load_checkpoint,
)


class SentimentPredictor:
    """High-level interactive sentiment inference predictor."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        ckpt = safe_load_checkpoint(checkpoint_path, device="cpu")
        run_cfg = ckpt.get("config", {})
        model_cfg = run_cfg.get("model", {})
        data_cfg = run_cfg.get("data", {})

        self.model_name = model_cfg.get("name", "distilbert-base-uncased")
        self.max_length = data_cfg.get("max_length", 512)
        clf_dropout = (run_cfg.get("training") or {}).get("classifier_dropout", 0.15)
        # Rebuild the exact training-time architecture. The checkpoint state dict
        # tells us whether the model was trained with LoRA (PEFT prefixed keys or
        # a stored lora config) — loading a plain model then would mismatch keys.
        state = ckpt["model_state_dict"]
        is_lora = bool(run_cfg.get("lora")) or any(
            k.startswith("base_model.model.") or k.startswith("lora_") for k in state
        )
        id2label = model_cfg.get("id2label")
        label2id = model_cfg.get("label2id")
        if id2label is not None:
            id2label = {int(k): v for k, v in id2label.items()}
        if label2id is not None:
            label2id = {k: int(v) for k, v in label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = build_model(
            model_name=self.model_name,
            num_labels=2,
            classifier_dropout=clf_dropout,
            id2label=id2label,
            label2id=label2id,
            use_lora=is_lora,
            lora_config_dict=run_cfg.get("lora"),
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @classmethod
    def from_latest_run(cls, run_root: str | Path = "experiments/lab3/runs", enable_swa: bool = True) -> "SentimentPredictor":
        """Factory method to load predictor from the newest run directory in run_root."""
        root_path = Path(run_root)
        matches = [p for p in root_path.glob("*_*") if p.is_dir() and (p / "checkpoints").exists()]
        if not matches:
            raise RuntimeError(f"No run directory found under {run_root}")

        run_dir = sorted(matches, key=lambda p: (p.name, p.stat().st_mtime), reverse=True)[0]
        files = resolve_run_files(run_dir)
        best_pt = files["best"]
        last_pt = files["last"]
        run_name = run_dir.name.split("_", 2)[-1] if len(run_dir.name.split("_")) >= 3 else run_dir.name
        swa_pt = files["swa"] or (run_dir / "checkpoints" / f"{run_name}_swa.pt")

        if enable_swa and best_pt and best_pt.exists() and last_pt and last_pt.exists() and not swa_pt.exists():
            average_checkpoints([best_pt, last_pt], swa_pt)

        target = swa_pt if (enable_swa and swa_pt and swa_pt.exists()) else (best_pt or last_pt)
        if not target or not target.exists():
            raise RuntimeError(f"No valid checkpoint found in {run_dir}")

        return cls(target)

    def predict(self, review_text: str) -> dict[str, Any]:
        """Predict sentiment label, confidence %, and probabilities for input review text."""
        from src.lab3.data.prepare_imdb import clean_text, head_tail_tokenize

        cleaned = clean_text(review_text)
        enc_dict = head_tail_tokenize([cleaned], self.tokenizer, max_length=self.max_length, head_ratio=0.25)
        enc = {
            "input_ids": torch.tensor(enc_dict["input_ids"], dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor(enc_dict["attention_mask"], dtype=torch.long).to(self.device),
        }
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(logits.argmax(-1))
            label = "POSITIVE 😊" if pred_id == 1 else "NEGATIVE 😞"
            confidence = probs[pred_id].item() * 100

        res = {
            "review": review_text,
            "label": label,
            "confidence": confidence,
            "prob_pos": probs[1].item(),
            "prob_neg": probs[0].item(),
        }
        print(f"Input Review: \"{review_text}\"")
        print(f"Prediction:   {label} (Confidence: {confidence:.2f}%)")
        print(f"Probabilities: Pos={probs[1]:.4f}, Neg={probs[0]:.4f}\n")
        return res
