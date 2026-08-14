"""Exercise 1 — Zero-Shot Sentiment Analysis Baseline.

This script implements Exercise 1 of Practice 3:
1. Load a pre-trained sentiment analysis model from the Hugging Face Hub using pipeline.
2. Tokenize sample sentences and display token IDs and tokens.
3. Perform sentiment analysis on sample inputs.
4. Optionally evaluate zero-shot sentiment prediction accuracy on the IMDB test set
   and compare it against the majority-class baseline floor.
5. Auto-persist results to `experiments/results/baseline_imdb_sentiment.json` with 5W1H metadata.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across torch, cuda, and Python stdlib."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demonstrate_tokenization_and_pipeline(
    sample_text: str, model_name: str, device_id: int
) -> dict[str, Any]:
    """Demonstrate tokenization (Exercise 1 Step 3) and pipeline inference (Step 4)."""
    print("\n" + "=" * 60)
    print(" EXERCISE 1: TOKENIZATION & ZERO-SHOT SENTIMENT PIPELINE DEMO")
    print("=" * 60)
    print(f"Sample Input: \"{sample_text}\"")

    # 1. Tokenizer Demonstration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    encoding = tokenizer(sample_text)

    print("\n--- Tokenizer Breakdown ---")
    print(f"Tokens:       {tokens}")
    print(f"Token IDs:    {token_ids}")
    print(f"Input IDs:    {encoding['input_ids']}")
    print(f"Attention:    {encoding['attention_mask']}")

    # 2. Pipeline Inference Demonstration
    print("\n--- Pipeline Sentiment Analysis ---")
    clf = pipeline("sentiment-analysis", model=model_name, device=device_id)
    start_time = time.time()
    result = clf(sample_text)[0]
    latency_ms = (time.time() - start_time) * 1000.0

    print(f"Model ID:     {model_name}")
    print(f"Predicted:    Label='{result['label']}', Score={result['score']:.4f}")
    print(f"Latency:      {latency_ms:.2f} ms")
    print("=" * 60 + "\n")

    return {
        "sample_text": sample_text,
        "tokens": tokens,
        "token_ids": token_ids,
        "predicted_label": result["label"],
        "confidence_score": float(result["score"]),
        "latency_ms": latency_ms,
    }


def evaluate_zero_shot_imdb(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    data_dir: Path | None = None,
    device_id: int = 0,
    output_json: Path | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Alias function for evaluate_imdb_baseline for framework compatibility."""
    return evaluate_imdb_baseline(model_name=model_name, batch_size=batch_size, max_samples=max_samples, device_id=device_id)


def evaluate_imdb_baseline(
    model_name: str,
    batch_size: int = 32,
    max_samples: int | None = None,
    device_id: int = 0,
) -> dict[str, Any]:

    """Evaluate zero-shot pipeline on IMDB test split alongside majority-class baseline floor."""
    print("Loading IMDB test dataset via Hugging Face `datasets`...")
    dataset = load_dataset("stanfordnlp/imdb", split="test")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Subsampled test set to {max_samples} reviews for fast evaluation.")

    print(f"Total test samples to evaluate: {len(dataset)}")

    # Compute majority-class baseline floor
    labels = list(dataset["label"])
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    majority_class_acc = max(pos_count, neg_count) / float(len(labels))
    print(f"Majority-class Baseline Accuracy (Floor): {majority_class_acc * 100.0:.2f}%")

    # Load zero-shot sentiment pipeline with top_k=None to obtain raw prediction scores/probabilities
    clf = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device_id,
        truncation=True,
        max_length=512,
        top_k=None,
    )

    # Print Formatted Model Parameter Architecture Summary for Ex 1
    if hasattr(clf, "model"):
        all_params = sum(p.numel() for p in clf.model.parameters())
        print("\n" + "=" * 65)
        print(" MODEL PARAMETER ARCHITECTURE SUMMARY (EXERCISE 1 BASELINE)")
        print("=" * 65)
        print(f" Pretrained Pipeline Model: {model_name}")
        print(" Evaluation Mode:           Zero-Shot Pretrained Baseline (No Training)")
        print(f" Total Parameters:          {all_params:,}")
        print(" Trainable Parameters:      0 (0.00% - Fully Frozen Pretrained)")
        print(f" Pretrained Weights:        {all_params:,} (100.00%)")
        print("=" * 65 + "\n")

    print(f"Evaluating zero-shot model '{model_name}' on IMDB test set...")

    texts = dataset["text"]
    correct = 0
    y_true = []
    y_score_pos = []
    start_eval_time = time.time()

    # Batch evaluation
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        results = clf(batch_texts)

        for res_list, target in zip(results, batch_labels, strict=False):
            # res_list contains scores for both classes, e.g. [{'label': 'POSITIVE', 'score': 0.99}, ...]
            pos_score = 0.0
            pred_label = 0
            max_s = -1.0
            for item in res_list:
                lbl_str = str(item["label"]).upper()
                s = float(item["score"])
                if "POS" in lbl_str or lbl_str == "LABEL_1":
                    pos_score = s
                if s > max_s:
                    max_s = s
                    pred_label = 1 if ("POS" in lbl_str or lbl_str == "LABEL_1") else 0

            y_true.append(target)
            y_score_pos.append(pos_score)
            if pred_label == target:
                correct += 1

    total_eval_time = time.time() - start_eval_time
    zero_shot_acc = correct / float(len(dataset))

    # Compute ROC-AUC Score and Curve
    from sklearn.metrics import roc_auc_score, roc_curve
    roc_auc = roc_auc_score(y_true, y_score_pos)
    fpr, tpr, thresholds = roc_curve(y_true, y_score_pos)

    print(f"\nZero-Shot Accuracy: {zero_shot_acc * 100.0:.2f}%")
    print(f"Zero-Shot ROC-AUC Score: {roc_auc:.4f}")
    print(f"Total Evaluation Time: {total_eval_time:.2f} s")

    # Generate and Save ROC Curve Plot
    plots_dir = Path("experiments/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "baseline_zero_shot_roc_curve.png"

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Zero-Shot ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Floor (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve — Zero-Shot Baseline ({model_name})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved ROC Curve plot to: {plot_path}")

    # Write PyTorch TensorBoard Logs per PYTORCH_FRAMEWORK_RULES.md
    from torch.utils.tensorboard import SummaryWriter
    tb_dir = Path("experiments/runs/baseline_zero_shot/tensorboard")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_scalar("Baseline/Zero_Shot_Accuracy", zero_shot_acc, 0)
    writer.add_scalar("Baseline/Majority_Floor_Accuracy", majority_class_acc, 0)
    writer.add_scalar("Baseline/Zero_Shot_ROC_AUC", roc_auc, 0)
    writer.add_scalar("Baseline/Eval_Time_Seconds", total_eval_time, 0)
    writer.close()
    print(f"PyTorch TensorBoard scalar metrics logged to: {tb_dir}")

    return {
        "num_test_samples": len(dataset),
        "majority_class_accuracy": majority_class_acc,
        "zero_shot_accuracy": zero_shot_acc,
        "zero_shot_roc_auc": roc_auc,
        "eval_time_seconds": total_eval_time,
        "roc_curve_plot": str(plot_path),
        "tensorboard_log_dir": str(tb_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise 1 — Hugging Face Zero-Shot Sentiment Analysis Baseline"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="I absolutely loved this movie! The acting was superb and the plot kept me hooked.",
        help="Sample text sentence to tokenize and evaluate.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Pre-trained Hugging Face sentiment model ID.",
    )
    parser.add_argument(
        "--eval-imdb",
        action="store_true",
        help="Run zero-shot evaluation on the IMDB test set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for IMDB evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate from IMDB test set (for quick testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Running on Device: {'GPU (cuda)' if device_id >= 0 else 'CPU'}")

    # 1. Exercise 1 Tokenization & Prediction Demo
    demo_res = demonstrate_tokenization_and_pipeline(
        sample_text=args.samples,
        model_name=args.model_name,
        device_id=device_id,
    )

    eval_res = {}
    if args.eval_imdb:
        eval_res = evaluate_imdb_baseline(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device_id=device_id,
        )

    # 2. 5W1H Metadata & Result Auto-Persistence
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "baseline_imdb_sentiment.json"

    payload = {
        "metadata_5w1h": {
            "who": "Learner & AI Coding Agent",
            "what": "Exercise 1 Zero-Shot Sentiment Analysis Baseline",
            "when": datetime.now().isoformat(),
            "where": f"Repo branch feature/ex1-sentiment-baseline ({'GPU' if device_id >= 0 else 'CPU'})",
            "why": "Establish baseline accuracy floor on IMDB for Practice 3 Exercise 1 & 2 comparison",
            "how": f"Hugging Face pipeline with pretrained model '{args.model_name}'",
        },
        "environment": {
            "model_name": args.model_name,
            "device": "cuda" if device_id >= 0 else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            if torch.cuda.is_available()
            else 0.0,
        },
        "demo_sample": demo_res,
        "evaluation": eval_res if eval_res else "Not run (pass --eval-imdb to run full evaluation)",
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Persisted baseline results with 5W1H metadata to: {out_file}")


if __name__ == "__main__":
    main()
