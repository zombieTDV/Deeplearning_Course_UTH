"""Interactive Sentiment Predictor and Inference Engine."""

from typing import Any
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.checkpoint_utils import safe_load_checkpoint, latest_run_dir, average_checkpoints, resolve_run_files


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
        lora_cfg = run_cfg.get("lora")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        from src.models.model_builder import build_model
        self.model = build_model(
            model_name=self.model_name,
            num_labels=model_cfg.get("num_labels", 2),
            classifier_dropout=clf_dropout,
            id2label={int(k): v for k, v in model_cfg.get("id2label", {"0": "neg", "1": "pos"}).items()} if model_cfg.get("id2label") else None,
            label2id={k: int(v) for k, v in model_cfg.get("label2id", {"neg": 0, "pos": 1}).items()} if model_cfg.get("label2id") else None,
            lora_config_dict=lora_cfg,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def from_latest_run(cls, run_root: str | Path = "experiments/runs", enable_swa: bool = True) -> "SentimentPredictor":
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
        enc = self.tokenizer(review_text, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
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
