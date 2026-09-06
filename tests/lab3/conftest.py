"""Pytest Global Configuration & Environment Guard."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent PyTorch c10 ApproximateClock non-monotonic CPU frequency assertion crash on Linux during unit testing
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONUNBUFFERED"] = "1"
