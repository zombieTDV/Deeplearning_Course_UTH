"""Exercise 2 — Finetune `distilbert-base-uncased` on IMDB with the HF Trainer.

Script-only training CLI (LOGGING_CHECKPOINT_RULES.md §1). Wraps the HF
`Trainer` + `TrainingArguments` (Exercise 2 steps 5-6) and persists full-state
checkpoints (``<run>_best.pt``, ``<run>_last.pt``), a log file, config, and
JSONL history into ``experiments/runs/<ts>_<run>/`` with resume support.

Usage::

    python -m src.training.imdb_sentiment_train --epochs 3 --seed 42 --tb
    python -m src.training.imdb_sentiment_train --smoke
    python -m src.training.imdb_sentiment_train --resume
    python -m src.training.imdb_sentiment_train --force-resume
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.data.prepare_imdb import prepare_imdb
from src.models import (
    build_model,
)
from src.models import (
    get_llrd_optimizer_grouped_parameters as _get_llrd_optimizer_grouped_parameters,
)
from src.utils.checkpoint_utils import (
    checkpoint_paths,
    latest_run_dir,
    next_run_dir,
    safe_load_checkpoint,
    save_checkpoint,
    update_registry,
)
from src.utils.resource_monitor import ResourceMonitor, cleanup_vram
from src.utils.run_logger import RunLogger


def _warmup_steps(config: dict[str, Any], ds: DatasetDict) -> int:
    """Convert config `warmup_ratio` to absolute warmup steps."""
    t = config["training"]
    n_train = len(ds["train"])
    eff_batch = t["batch_size"] * t["gradient_accumulation_steps"]
    steps_per_epoch = max(1, int(np.ceil(n_train / eff_batch)))
    total_steps = steps_per_epoch * int(t["epochs"])
    return max(1, int(total_steps * t["warmup_ratio"]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "eval_accuracy": accuracy_score(labels, preds),
        "eval_f1": f1_score(labels, preds, average="binary"),
    }


def _prepare_datasets(cfg: dict[str, Any], smoke: bool) -> tuple[DatasetDict, AutoTokenizer]:
    d = cfg["data"]
    ds, tokenizer, meta = prepare_imdb(
        dataset_id=d["name"],
        model_name=cfg["model"]["name"],
        max_length=d["max_length"],
        val_size=d["val_size"],
        seed=d["seed"],
        processed_dir=d["processed_dir"],
    )
    cfg["data"]["tokenized_splits"] = meta.get("splits", {k: len(v) for k, v in ds.items()})
    if smoke:
        ds = DatasetDict(
            {
                "train": ds["train"].select(range(64)),
                "val": ds["val"].select(range(16)),
                "test": ds["test"].select(range(16)),
            }
        )
        print(
            f"SMOKE TEST: {len(ds['train'])} train / {len(ds['val'])} val / "
            f"{len(ds['test'])} test samples"
        )
    return ds, tokenizer


def _full_state(
    trainer: Trainer,
    config: dict[str, Any],
    history: list[dict],
    best: dict | None,
    early_stop_triggered: bool = False,
) -> dict[str, Any]:
    return {
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
        "epoch": int(trainer.state.epoch),
        "global_step": int(trainer.state.global_step),
        "best_metrics": best or {},
        "history": history,
        "config": config,
        "rng": {
            "torch_seed": int(torch.initial_seed()),
            "numpy_seed": int(np.random.get_state()[1][0]),
        },
        "early_stop_triggered": early_stop_triggered,
        "commit": _git_commit(),
    }


class FullStateCallback(TrainerCallback):
    """Write ``_last.pt`` every evaluation and ``_best.pt`` on best eval metric (rules §3-§4)."""

    def __init__(
        self,
        run_dir: Path,
        run_name: str,
        config: dict[str, Any],
        logger: RunLogger,
        metric_name: str = "eval_accuracy",
        greater_is_better: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.run_name = run_name
        self.config = config
        self.logger = logger
        self.metric_name = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
        self.greater_is_better = greater_is_better
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.trainer: Trainer | None = None
        self.latest_train_loss: float | None = None
        self.early_stop_triggered: bool = False

    def attach(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs and "loss" in logs:
            self.latest_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        if not metrics:
            return
        row = {
            "epoch": float(state.epoch),
            "global_step": int(state.global_step),
            "train_loss": self.latest_train_loss,
            "eval_loss": metrics.get("eval_loss"),
            "eval_accuracy": metrics.get("eval_accuracy"),
            "eval_f1": metrics.get("eval_f1"),
        }
        self.history.append(row)
        self.logger.epoch_summary(int(state.epoch), {k: v for k, v in row.items() if v is not None})

        # Append row to history.jsonl
        metrics_dir = self.run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        history_jsonl = metrics_dir / f"{self.run_name}_history.jsonl"
        with open(history_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        current_val = row.get(self.metric_name)
        is_new_best = False
        if current_val is not None:
            if self.best is None:
                is_new_best = True
            else:
                best_val = self.best.get(self.metric_name)
                if best_val is None:
                    is_new_best = True
                elif self.greater_is_better and current_val > best_val:
                    is_new_best = True
                elif not self.greater_is_better and current_val < best_val:
                    is_new_best = True

        if is_new_best:
            self.best = row

        best_path, last_path = checkpoint_paths(self.run_dir, self.run_name)
        save_checkpoint(
            last_path,
            _full_state(self.trainer, self.config, self.history, None, early_stop_triggered=self.early_stop_triggered),
        )
        if self.best is not None:
            save_checkpoint(
                best_path,
                _full_state(self.trainer, self.config, self.history, self.best, early_stop_triggered=self.early_stop_triggered),
            )
        if self.trainer and self.trainer.model:
            self.trainer.model.zero_grad(set_to_none=True)
        cleanup_vram()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        # Single reliable capture point for the early-stop flag: the callback
        # ordering guarantees on_evaluate runs before EarlyStoppingCallback
        # flips control.should_training_stop, so we record the terminal state
        # here (the post-train checkpoint save below picks it up).
        if control.should_training_stop:
            self.early_stop_triggered = True


def _train(
    config: dict[str, Any],
    args: argparse.Namespace,
    resume_from: str | None,
    force_resume: bool,
) -> dict[str, Any]:
    cleanup_vram()
    t = config["training"]
    run_name = t["run_name"]
    run_dir = Path(resume_from) if resume_from else next_run_dir(t["run_root"], run_name)

    logger = RunLogger(run_dir, run_name)
    logger.info(f"run dir: {run_dir}")
    monitor = ResourceMonitor(target_gb=t.get("vram_budget_gb", 3.5))
    monitor.start_background()

    set_seed(t["seed"])
    logger.info(f"starting training: {json.dumps(t, default=str)}")

    ds, tokenizer = _prepare_datasets(config, smoke=args.smoke)

    # Configure classifier dropout (CLI override or config default)
    clf_dropout = args.classifier_dropout if args.classifier_dropout is not None else t.get("classifier_dropout", 0.20)
    model = build_model(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        classifier_dropout=clf_dropout,
        id2label={int(k): v for k, v in config["model"]["id2label"].items()},
        label2id={k: int(v) for k, v in config["model"]["label2id"].items()},
        freeze_layers=args.freeze_layers,
        lora_config_dict=config.get("lora"),
    )

    # Print Formatted Model Parameter Architecture Summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    frozen_params = all_params - trainable_params
    adaptation_method = f"LoRA PEFT (r={config['lora']['r']}, alpha={config['lora']['lora_alpha']})" if config.get("lora") else "Full Fine-Tuning"

    print("\n" + "=" * 65)
    print(" MODEL PARAMETER ARCHITECTURE SUMMARY")
    print("=" * 65)
    print(f" Backbone Architecture:   {config['model']['name']}")
    print(f" Adaptation Mode:         {adaptation_method}")
    print(f" Total Parameters:        {all_params:,}")
    print(f" Trainable Parameters:    {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f" Frozen Parameters:       {frozen_params:,} ({100 * frozen_params / all_params:.2f}%)")
    print("=" * 65 + "\n")

    # Freeze bottom transformer layers if specified

    if args.freeze_layers > 0:
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for i in range(min(args.freeze_layers, len(model.distilbert.transformer.layer))):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Froze bottom {args.freeze_layers} layers: {trainable:,} / {total:,} trainable params")

    scheduler_type = args.lr_scheduler_type if args.lr_scheduler_type is not None else t.get("lr_scheduler_type", "cosine")
    label_smoothing = args.label_smoothing_factor if args.label_smoothing_factor is not None else t.get("label_smoothing_factor", 0.0)

    metric_name = args.metric_for_best_model if args.metric_for_best_model is not None else t.get("metric_for_best_model", "accuracy")
    eval_metric_name = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
    if args.greater_is_better is not None:
        greater_is_better = args.greater_is_better
    else:
        greater_is_better = False if "loss" in metric_name else True

    train_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_name,
        num_train_epochs=t.get("epochs", 4),
        per_device_train_batch_size=t.get("batch_size", 8),
        per_device_eval_batch_size=t.get("batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        learning_rate=t.get("lr", 2.0e-5),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_steps=_warmup_steps(config, ds),
        lr_scheduler_type=scheduler_type,
        label_smoothing_factor=label_smoothing,
        max_grad_norm=args.max_grad_norm,
        fp16=t.get("fp16", torch.cuda.is_available()),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 300),
        logging_steps=t.get("logging_steps", 50),
        disable_tqdm=not getattr(args, "show_tqdm", False),
        save_strategy="no",
        dataloader_num_workers=0,
        seed=t.get("seed", 42),
        report_to=["tensorboard"] if args.tb else ["none"],
    )

    callback = FullStateCallback(
        run_dir,
        run_name,
        config,
        logger,
        metric_name=eval_metric_name,
        greater_is_better=greater_is_better,
    )
    callbacks: list[Any] = [callback]
    patience = args.early_stopping_patience if args.early_stopping_patience > 0 else t.get("early_stopping_patience", 3)
    if patience > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        logger.info(f"Enabled EarlyStoppingCallback(patience={patience})")

    # Layer-wise Learning Rate Decay (LLRD) setup
    custom_optimizer = None
    use_llrd = getattr(args, "llrd", False) or t.get("use_llrd", False)
    if use_llrd:
        decay_factor = getattr(args, "decay_factor", 0.8) or t.get("decay_factor", 0.8)
        grouped_params = _get_llrd_optimizer_grouped_parameters(
            model=model,
            base_lr=t["lr"],
            weight_decay=t["weight_decay"],
            decay_factor=decay_factor,
        )
        custom_optimizer = torch.optim.AdamW(grouped_params, lr=t["lr"])
        logger.info(f"Enabled LLRD with decay_factor={decay_factor}")

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": ds["train"],
        "eval_dataset": ds["val"],
        "processing_class": tokenizer,
        "compute_metrics": _compute_metrics,
        "callbacks": callbacks,
    }
    if custom_optimizer is not None:
        trainer_kwargs["optimizers"] = (custom_optimizer, None)

    trainer = Trainer(**trainer_kwargs)

    callback.attach(trainer)

    remaining_epochs = t["epochs"]
    if resume_from:
        best_path, last_path = checkpoint_paths(run_dir, run_name)
        ckpt_path = best_path if force_resume else last_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        ckpt = safe_load_checkpoint(ckpt_path, device="cpu")

        if not force_resume and ckpt.get("early_stop_triggered", False):
            logger.info(f"Run '{run_name}' was previously stopped early at epoch {ckpt.get('epoch')}. `--resume` halted.")
            print(f"Run '{run_name}' stopped early at epoch {ckpt.get('epoch')}. `--resume` halted. Use `--force-resume` to rewind from best checkpoint.")
            return {
                "run_dir": str(run_dir),
                "run_name": run_name,
                "history": list(ckpt.get("history", [])),
                "best": dict(ckpt.get("best_metrics", {})),
                "vram": monitor.summary(),
            }

        done_epochs = int(ckpt.get("epoch", 0))
        remaining_epochs = max(0, t["epochs"] - done_epochs)
        train_args.num_train_epochs = remaining_epochs

        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.create_optimizer_and_scheduler(num_training_steps=-1)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            trainer.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        rng_info = ckpt.get("rng", {})
        if "torch_seed" in rng_info:
            set_seed(int(rng_info["torch_seed"]))

        callback.history = list(ckpt.get("history", []))
        callback.best = dict(ckpt.get("best_metrics", {})) or None
        if force_resume:
            callback.early_stop_triggered = False
        logger.info(f"resumed from {ckpt_path} at epoch {done_epochs} (remaining {remaining_epochs})")

    trainer.train()
    # Ensure a final _last.pt reflects the last state / best metrics.
    best_path, last_path = checkpoint_paths(run_dir, run_name)
    save_checkpoint(
        last_path,
        _full_state(trainer, config, callback.history, callback.best, early_stop_triggered=callback.early_stop_triggered),
    )
    if callback.best is not None:
        save_checkpoint(
            best_path,
            _full_state(trainer, config, callback.history, callback.best, early_stop_triggered=callback.early_stop_triggered),
        )
    update_registry(t["run_root"], run_name, run_dir)

    metrics_path = run_dir / "metrics" / f"{run_name}_config.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "vram": monitor.summary()}, f, indent=2)
    logger.info(monitor.report())

    return {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "history": callback.history,
        "best": callback.best,
        "vram": monitor.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise 2 — finetune distilbert-base-uncased on IMDB"
    )
    parser.add_argument("--config", default="configs/config_imdb_sentiment.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--run-name", default=None, help="override run_name in config")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="patience for EarlyStoppingCallback")
    parser.add_argument("--metric-for-best-model", default=None, help="metric for best model (e.g. accuracy, f1, loss)")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=None,
                        help="whether a higher metric value is better (default: auto — False for loss, True otherwise)")
    parser.add_argument("--lr-scheduler-type", default=None, help="learning rate scheduler type (linear, cosine, etc.)")
    parser.add_argument("--weight-decay", type=float, default=None, help="override weight decay")
    parser.add_argument("--classifier-dropout", type=float, default=None, help="override classifier dropout")
    parser.add_argument("--label-smoothing-factor", type=float, default=None, help="label smoothing factor (e.g. 0.10)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="maximum gradient norm for clipping")
    parser.add_argument("--freeze-layers", type=int, default=0, help="number of bottom transformer layers to freeze")
    parser.add_argument("--llrd", action="store_true", help="enable Layer-wise Learning Rate Decay (LLRD)")
    parser.add_argument("--decay-factor", type=float, default=0.8, help="LLRD decay factor per layer (default 0.8)")
    parser.add_argument("--resume", action="store_true", help="resume from the latest run's _last.pt")
    parser.add_argument("--force-resume", action="store_true", help="rewind from _best.pt")
    parser.add_argument("--tb", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--show-tqdm", action="store_true", help="show raw tqdm progress bars (default disabled for clean notebook outputs)")
    parser.add_argument("--smoke", action="store_true", help="run on a tiny subset")
    parser.add_argument("--baseline", action="store_true", help="run Exercise 1 zero-shot baseline evaluation instead of fine-tuning")
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = {
        "model": raw["model"],
        "data": raw.get("data") or raw.get("dataset"),
        "training": raw["training"],
    }
    if "lora" in raw:
        config["lora"] = raw["lora"]


    t = config["training"]
    if args.run_name is not None:
        t["run_name"] = args.run_name
    if args.epochs is not None:
        t["epochs"] = args.epochs
    if args.seed is not None:
        t["seed"] = args.seed
    if args.weight_decay is not None:
        t["weight_decay"] = args.weight_decay

    if args.baseline or "baseline" in t.get("run_name", "").lower() or t.get("epochs", 4) == 0:
        print("=" * 65)
        print(" RUNNING EXERCISE 1: ZERO-SHOT BASELINE EVALUATION")
        print("=" * 65)
        import src.experiments.baseline_imdb_sentiment as base_exp
        device_id = 0 if torch.cuda.is_available() else -1
        base_exp.evaluate_zero_shot_imdb(
            model_name=config["model"]["name"],
            data_dir=Path(config["data"]["processed_dir"]),
            device_id=device_id,
            output_json=Path("experiments/results/baseline_imdb_sentiment.json"),
        )
        return

    resume_from = None
    force = False
    if args.force_resume or args.resume:
        resume_from = latest_run_dir(t["run_root"], t["run_name"])
        force = args.force_resume
        if resume_from is None:
            raise SystemExit(f"No prior run dir found for '{t['run_name']}' under {t['run_root']}")

    result = _train(config, args, resume_from=resume_from, force_resume=force)
    print(f"\nRun complete: {result['run_dir']}")
    print(f"Best: {result['best']}")
    print(result["vram"])


if __name__ == "__main__":
    main()
