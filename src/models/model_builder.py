"""Model Builder & LLRD Optimizer Parameter Utilities."""

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification


def build_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    classifier_dropout: float = 0.20,
    id2label: dict[str, str] | None = None,
    label2id: dict[str, int] | None = None,
    freeze_layers: int = 0,
    use_lora: bool = False,
    lora_config_dict: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Build and initialize HuggingFace Sequence Classification Model with optional LoRA PEFT wrapping."""
    kwargs: dict[str, Any] = {
        "num_labels": num_labels,
        "seq_classif_dropout": classifier_dropout,
    }
    if id2label is not None:
        kwargs["id2label"] = id2label
    if label2id is not None:
        kwargs["label2id"] = label2id

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)

    # Freeze bottom transformer layers if requested
    if freeze_layers > 0 and hasattr(model, "distilbert"):
        for layer in model.distilbert.transformer.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    # Apply LoRA Low-Rank Adaptation if requested
    if use_lora or lora_config_dict is not None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as err:
            raise ImportError("PEFT package is required for LoRA adaptation. Install via `pip install peft`.") from err

        cfg = lora_config_dict or {}
        target_mods = cfg.get("target_modules", ["q_lin", "v_lin"] if "distilbert" in model_name.lower() else ["query", "value"])

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.get("r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.10),
            target_modules=target_mods,
            bias=cfg.get("bias", "none"),
        )
        model = get_peft_model(model, peft_config)

        # Print Trainable Parameter Summary
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"[PEFT LoRA Loaded] Trainable Params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    return model


def get_llrd_optimizer_grouped_parameters(
    model: torch.nn.Module,
    base_lr: float = 2.0e-5,
    weight_decay: float = 0.01,
    decay_factor: float = 0.8,
) -> list[dict[str, Any]]:
    """Build Layer-wise Learning Rate Decay (LLRD) parameter groups for DistilBERT / Transformer backbones."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    distilbert_model = getattr(model, "distilbert", None)
    if distilbert_model is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": weight_decay}]

    num_layers = len(distilbert_model.transformer.layer)
    grouped_parameters: list[dict[str, Any]] = []

    # Head parameters (base_lr)
    head_modules = []
    if hasattr(model, "pre_classifier"):
        head_modules.append(model.pre_classifier)
    if hasattr(model, "classifier"):
        head_modules.append(model.classifier)

    head_params = [p for m in head_modules for p in m.parameters() if p.requires_grad]
    if head_params:
        grouped_parameters.append({
            "params": head_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        })

    # Transformer layers (decayed from top to bottom)
    for layer_idx in range(num_layers - 1, -1, -1):
        layer = distilbert_model.transformer.layer[layer_idx]
        layer_lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))

        decay_params = [p for n, p in layer.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)]
        no_decay_params = [p for n, p in layer.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)]

        if decay_params:
            grouped_parameters.append({"params": decay_params, "lr": layer_lr, "weight_decay": weight_decay})
        if no_decay_params:
            grouped_parameters.append({"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0})

    # Embeddings (lowest lr)
    embed_lr = base_lr * (decay_factor ** num_layers)
    if hasattr(distilbert_model, "embeddings"):
        embed_params = [p for p in distilbert_model.embeddings.parameters() if p.requires_grad]
        if embed_params:
            grouped_parameters.append({"params": embed_params, "lr": embed_lr, "weight_decay": weight_decay})

    return grouped_parameters
