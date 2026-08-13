"""Core package for the deep-learning pipeline."""

from src.data.eda_imdb import IMDBDatasetEDA
from src.data.prepare_imdb import prepare_imdb
from src.eval.error_auditor import ErrorAuditor
from src.eval.evaluate_model import evaluate
from src.eval.evaluator import IMDBEvaluator
from src.eval.plotter import IMDBPlotter
from src.models.model_builder import (
    build_model,
    get_llrd_optimizer_grouped_parameters,
)
from src.models.predictor import SentimentPredictor
from src.training.trainer import IMDBTrainer
from src.utils.checkpoint_utils import (
    average_checkpoints,
    latest_run_dir,
    safe_load_checkpoint,
)
from src.utils.resource_monitor import ResourceMonitor, cleanup_vram

__all__ = [
    "IMDBPlotter",
    "IMDBDatasetEDA",
    "IMDBTrainer",
    "IMDBEvaluator",
    "SentimentPredictor",
    "ErrorAuditor",
    "prepare_imdb",
    "evaluate",
    "build_model",
    "get_llrd_optimizer_grouped_parameters",
    "safe_load_checkpoint",
    "latest_run_dir",
    "average_checkpoints",
    "ResourceMonitor",
    "cleanup_vram",
]
