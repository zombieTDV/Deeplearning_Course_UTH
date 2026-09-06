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
    "IMDBCleanlabAuditor": "src.lab3.data.cleanlab_denoiser",
    "IMDBDatasetEDA": "src.lab3.data.eda_imdb",
    "clean_text": "src.lab3.data.prepare_imdb",
    "prepare_imdb": "src.lab3.data.prepare_imdb",
    "ErrorAuditor": "src.lab3.eval.error_auditor",
    "compute_comprehensive_benchmark": "src.lab3.eval.evaluate_model",
    "compute_model_metrics": "src.lab3.eval.evaluate_model",
    "evaluate": "src.lab3.eval.evaluate_model",
    "format_ascii_metrics_table": "src.lab3.eval.evaluate_model",
    "plot_metrics_heatmap_table": "src.lab3.eval.evaluate_model",
    "IMDBEvaluator": "src.lab3.eval.evaluator",
    "IMDBPlotter": "src.lab3.eval.plotter",
    "build_model": "src.lab3.models.model_builder",
    "get_llrd_optimizer_grouped_parameters": "src.lab3.models.model_builder",
    "SentimentPredictor": "src.lab3.models.predictor",
    "IMDBTrainer": "src.lab3.training.trainer",
    "average_checkpoints": "src.lab3.utils.checkpoint_utils",
    "latest_run_dir": "src.lab3.utils.checkpoint_utils",
    "safe_load_checkpoint": "src.lab3.utils.checkpoint_utils",
    "ResourceMonitor": "src.lab3.utils.resource_monitor",
    "cleanup_vram": "src.lab3.utils.resource_monitor",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'src.lab3' has no attribute {name!r}")
    return getattr(_importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
