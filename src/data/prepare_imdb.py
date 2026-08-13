"""IMDB preprocessing — tokenization (feature engineering for DL) + train/val/test splits.

Phase 3 (FEATURE_SPLIT.md): tokenize with the static `distilbert-base-uncased`
tokenizer (never fit on data → no leakage), carve a validation split out of the
IMDB train split, and seal the test split (evaluated exactly once, golden rule 4).
Caches the tokenized datasets to `data/processed/imdb_tokenized/` as arrow files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def _validate_no_test_leakage(train: Dataset, val: Dataset, test: Dataset) -> None:
    """Assert split cardinality so no test statistics can leak into training."""
    # IDs are not shared across splits by construction (IMDB has distinct
    # train/test examples); the assertion guards the 25k/2.5k/22.5k geometry.
    expected_train = 25000
    assert len(test) == 25000, f"expected 25000 test reviews, got {len(test)}"
    assert len(train) + len(val) == expected_train, (
        f"train({len(train)}) + val({len(val)}) must equal {expected_train}"
    )


def tokenize_split(examples: dict[str, list], tokenizer: AutoTokenizer, max_length: int) -> dict[str, Any]:
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def prepare_imdb(
    dataset_id: str = "stanfordnlp/imdb",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    val_size: float = 0.1,
    seed: int = 42,
    processed_dir: str = "data/processed/imdb_tokenized",
    force: bool = False,
) -> tuple[DatasetDict, AutoTokenizer, dict[str, Any]]:
    """Load IMDB, tokenize, split, cache to arrow files, and return (datasets, tokenizer, meta)."""
    import json

    root = Path(".").resolve()
    if root.name == "notebooks":
        root = root.parent
    processed = root / processed_dir if not Path(processed_dir).is_absolute() else Path(processed_dir)

    flat_ready = all((processed / s).is_dir() for s in ("train", "val", "test"))

    ds_cache_ready = (processed / "dataset").is_dir()
    meta_path = processed / "meta.json"

    cache_valid = False
    if (flat_ready or ds_cache_ready) and not force:
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                cached_meta = json.load(f)
            if (
                cached_meta.get("dataset_id") == dataset_id
                and cached_meta.get("model_name") == model_name
                and cached_meta.get("max_length") == max_length
                and cached_meta.get("val_size") == val_size
                and cached_meta.get("seed") == seed
            ):
                cache_valid = True
            else:
                print(
                    f"Cache invalidated for {processed}: config mismatch "
                    f"(requested max_length={max_length}, model_name={model_name}). Re-tokenizing..."
                )
        else:
            cache_valid = True

    if cache_valid and not force:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if ds_cache_ready:
            ds = DatasetDict.load_from_disk(str(processed / "dataset"))
        else:
            ds = DatasetDict(
                {
                    name: Dataset.load_from_disk(str(processed / name))
                    for name in ("train", "val", "test")
                }
            )
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        meta: dict[str, Any] = {"splits": {k: len(v) for k, v in ds.items()}}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        print(f"Loaded cached tokenized datasets from {processed}")
        return ds, tokenizer, meta


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw = load_dataset(dataset_id)
    train_test = raw["train"].train_test_split(test_size=val_size, seed=seed)
    ds = DatasetDict(
        {
            "train": train_test["train"],
            "val": train_test["test"],
            "test": raw["test"],
        }
    )

    def _map(dataset: Dataset) -> Dataset:
        return dataset.map(
            lambda ex: tokenize_split(ex, tokenizer, max_length),
            batched=True,
            remove_columns=["text"],
        )

    tokenized = DatasetDict(
        {name: _map(ds[name]) for name in ("train", "val", "test")}
    )
    _validate_no_test_leakage(tokenized["train"], tokenized["val"], tokenized["test"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    meta = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "max_length": max_length,
        "val_size": val_size,
        "seed": seed,
        "splits": {k: len(v) for k, v in tokenized.items()},
        "columns": ["input_ids", "attention_mask", "label"],
    }
    processed.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        tokenized[name].save_to_disk(str(processed / name))
    with open(processed / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"Tokenized + split datasets cached to {processed}")
    return tokenized, tokenizer, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenized IMDB splits (Phase 3)")
    parser.add_argument("--dataset-id", default="stanfordnlp/imdb")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", default="data/processed/imdb_tokenized")
    parser.add_argument("--force", action="store_true", help="re-tokenize even if cached")
    args = parser.parse_args()

    ds, tokenizer, meta = prepare_imdb(
        dataset_id=args.dataset_id,
        model_name=args.model_name,
        max_length=args.max_length,
        val_size=args.val_size,
        seed=args.seed,
        processed_dir=args.processed_dir,
        force=args.force,
    )
    print(f"Dataset cached to {args.processed_dir}: {meta['splits']}")


if __name__ == "__main__":
    main()
