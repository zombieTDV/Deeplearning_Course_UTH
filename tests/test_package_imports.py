"""Regression tests for package importability (PR #15 review fixes).

Covers B1 (the ``src`` package must import cleanly even when the optional
``scratch`` error-audit helper is absent) and B2 (the training CLI must expose
``safe_load_checkpoint``, used by the ``--resume``/``--force-resume`` path).
"""

import importlib

_LAZY_NAMES = [
    "IMDBCleanlabAuditor",
    "IMDBDatasetEDA",
    "IMDBEvaluator",
    "IMDBPlotter",
    "IMDBTrainer",
    "ErrorAuditor",
    "SentimentPredictor",
    "build_model",
    "prepare_imdb",
    "clean_text",
    "evaluate",
    "safe_load_checkpoint",
    "cleanup_vram",
    "average_checkpoints",
    "latest_run_dir",
    "ResourceMonitor",
]


def test_src_package_imports_cleanly():
    # B1: importing the src package must not eagerly import heavy/optional
    # helpers (transformers, IPython, scratch.analyze_misclassifications).
    import src

    assert hasattr(src, "__getattr__")  # lazy export mechanism active


def test_src_lazy_exports_resolve():
    # Every documented public name must resolve through the lazy __getattr__.
    import src

    for name in _LAZY_NAMES:
        assert getattr(src, name) is not None, f"lazy export {name} failed"


def test_error_auditor_imports_without_scratch():
    # B1: the ErrorAuditor module itself must import cleanly (no top-level
    # `import scratch.analyze_misclassifications`), even though that optional
    # helper module is not present.
    mod = importlib.import_module("src.eval.error_auditor")
    assert mod.ErrorAuditor is not None


def test_training_module_imports_and_exposes_resume_helper():
    # B2: the training CLI imports cleanly and exposes safe_load_checkpoint
    # (used by the --resume/--force-resume path).
    mod = importlib.import_module("src.training.imdb_sentiment_train")
    assert callable(mod.safe_load_checkpoint)
    assert callable(mod.build_model)
    assert callable(mod._get_llrd_optimizer_grouped_parameters)
