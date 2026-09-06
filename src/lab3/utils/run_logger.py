"""RunLogger — real-time logging + JSONL history for training runs.

Owned by `src/utils/` per LOGGING_CHECKPOINT_RULES.md (audit finding AQ-1).
Zero-dependency: console + rotating file + JSONL epoch history. TensorBoard is
opt-in via the training CLI's ``--tb`` flag, which uses ``SummaryWriter``
directly rather than through this module.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_LEVELS = {
    "INFO": logging.INFO,
    "TRAIN": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}

MAX_LOG_BYTES = 20 * 1024 * 1024  # 20 MB — auto-rotate per logging rules §6
MAX_LOG_ARCHIVES = 3


class RunLogger:
    """Writes live progress, timestamped logs, and append-only JSONL history.

    Creates the run directory once; every artifact of a run lives inside it.
    """

    def __init__(self, run_dir: str | Path, run_name: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.log_dir = self.run_dir / "logs"
        self.metrics_dir = self.run_dir / "metrics"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{run_name}.log"
        self.history_file = self.metrics_dir / f"{run_name}_history.jsonl"

        self._logger = self._setup_logger()
        self._last_progress = ""

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"run.{self.run_name}")
        logger.setLevel(logging.INFO)
        if logger.handlers:
            return logger

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        return logger

    # -- logging -----------------------------------------------------------

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warn(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def train(self, message: str) -> None:
        self._logger.info(message)

    def progress(self, **fields: Any) -> None:
        """Print a single-line overwrite of current step state."""
        parts = [f"{k}={v}" for k, v in fields.items() if v is not None]
        line = "  " + " | ".join(parts)
        sys.stdout.write("\r" + " " * len(self._last_progress) + "\r")
        sys.stdout.write(line)
        sys.stdout.flush()
        self._last_progress = line

    def epoch_summary(self, epoch: int, metrics: dict[str, Any]) -> None:
        summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        self._logger.info("Epoch %d — %s", epoch, summary)
        self.append_history({"epoch": epoch, **metrics})

    # -- persistence ---------------------------------------------------------

    def append_history(self, row: dict[str, Any]) -> None:
        """Append one JSON line per epoch (append-only, crash-safe, §6)."""
        row = {"timestamp": datetime.now().isoformat(), **row}
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rotate_if_needed()

    def _rotate_if_needed(self) -> None:
        """Gzip-archive the log past 20 MB; keep MAX_LOG_ARCHIVES archives."""
        try:
            if self.log_file.stat().st_size < MAX_LOG_BYTES:
                return
            for idx in range(MAX_LOG_ARCHIVES, 0, -1):
                src = self.log_dir / f"{self.run_name}.{idx - 1}.log.gz"
                dst = self.log_dir / f"{self.run_name}.{idx}.log.gz"
                if src.exists():
                    if idx == MAX_LOG_ARCHIVES:
                        src.unlink()
                    else:
                        shutil.move(str(src), str(dst))
            archive = self.log_dir / f"{self.run_name}.1.log.gz"
            with self.log_file.open("rb") as f_in, gzip.open(archive, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            self.log_file.write_text("", encoding="utf-8")
        except OSError:
            pass

    def read_history(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.history_file.exists():
            with open(self.history_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        return rows
