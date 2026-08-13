"""Pytest Global Configuration & Environment Guard."""

import os

# Prevent PyTorch c10 ApproximateClock non-monotonic CPU frequency assertion crash on Linux during unit testing
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONUNBUFFERED"] = "1"
