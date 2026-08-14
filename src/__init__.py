"""Core package for the deep-learning pipeline.

Public names are exposed **lazily** (PEP 562 ``__getattr__``) so that
``import src`` — or importing any ``src.*`` submodule — never eagerly pulls in
the heavy/optional dependencies (``transformers``, ``datasets``, ``IPython``)
and never fails when an optional helper (e.g. the ``scratch`` error-audit
module) is absent from the checkout.
"""

from __future__ import annotations

import importlib as _importlib

_LAZY_EXPORTS: dict[str, str] = {
    "IMDBDatasetEDA": "src.data.eda_imdb",
    "prepare_imdb": "src.data.prepare_imdb",
    "ErrorAuditor": "src.eval.error_auditor",
    "evaluate": "src.eval.evaluate_model",
    "IMDBEvaluator": "src.eval.evaluator",
    "IMDBPlotter": "src.eval.plotter",
    "build_model": "src.models.model_builder",
    "get_llrd_optimizer_grouped_parameters": "src.models.model_builder",
    "SentimentPredictor": "src.models.predictor",
    "IMDBTrainer": "src.training.trainer",
    "average_checkpoints": "src.utils.checkpoint_utils",
    "latest_run_dir": "src.utils.checkpoint_utils",
    "safe_load_checkpoint": "src.utils.checkpoint_utils",
    "ResourceMonitor": "src.utils.resource_monitor",
    "cleanup_vram": "src.utils.resource_monitor",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    return getattr(_importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
