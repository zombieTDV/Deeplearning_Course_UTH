"""
run_logger.py — Lightweight, zero-dependency run logging for training & experiments.

Why this module
---------------
Long-running training must (a) report live progress and (b) persist everything
needed to resume after an interruption — without requiring `tqdm`, `tensorboard`,
or any third-party logger.  This module provides:

* console progress — single-line overwrite updates (loss / acc / lr / ETA);
* an append-only log file with automatic gzip rotation (stdlib only);
* JSONL metric history  — ``metrics/<run>_history.jsonl``, one JSON object per epoch;
* a config snapshot     — ``metrics/<run>_config.json`` (hyperparams, seeds, code state);
* an optional TensorBoard writer — used only if ``torch.utils.tensorboard`` imports
  cleanly, otherwise silently skipped (lightweight custom logger always works).

Run directory layout (see ``agents/rules/LOGGING_CHECKPOINT_RULES.md`` for the
authoritative rules):

    <runs_root>/
    └── <YYYYmmdd_HHMMSS>_<run_name>/
        ├── checkpoints/   # <run>_best.pt, <run>_last.pt   (full-state torch saves)
        ├── logs/          # <run>.log (+ rotated <run>.<n>.log.gz archives)
        ├── metrics/       # <run>_config.json, <run>_history.jsonl
        └── tensorboard/   # optional TensorBoard event files

Usage
-----
    from src.lab2.utils.run_logger import RunLogger

    logger = RunLogger(run_name="ResNet18-frozen", runs_root="experiments/lab2/runs")
    logger.write_config({"epochs": 20, "lr": 1e-3, "seed": 42})
    logger.log("Starting training")
    logger.progress("epoch 3/20 [120/704] loss=1.234 acc=85.1% eta=12m")
    logger.epoch_summary(3, {"train_loss": 1.1, "val_acc": 90.2})
    logger.close()
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Maximum size (bytes) of a log file before it is gzip-rotated.
LOG_ROTATE_BYTES = 20 * 1024 * 1024
# Number of rotated archives kept on disk.
LOG_ROTATE_KEEP = 3


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------
class RunLogger:
    """Console + file + JSONL logger for one training/experiment run.

    Args:
        run_name: Identifier for the run (e.g. ``"ResNet18-sota"``).  It is
            sanitized and used in every artifact file name.
        runs_root: Parent directory of all runs (default ``experiments/lab2/runs``).
        console: If True (default), also print to stdout.
    """

    def __init__(self, run_name: str, runs_root: str | Path = "experiments/lab2/runs",
                 console: bool = True):
        self.name = _sanitize(run_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(runs_root) / f"{ts}_{self.name}"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        self.metric_dir = self.run_dir / "metrics"
        self.tb_dir = self.run_dir / "tensorboard"
        for d in (self.checkpoint_dir, self.log_dir, self.metric_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.console = console
        self._log_path = self.log_dir / f"{self.name}.log"
        self._log_handle = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115
        self._tb_writer: Any | None = None
        self._progress_active = False
        self.log(f"[init] run dir: {self.run_dir}", level="INFO", console=console)

    # ------------------------------------------------------------------ files
    @property
    def history_path(self) -> Path:
        """Path of the JSONL metric history file."""
        return self.metric_dir / f"{self.name}_history.jsonl"

    @property
    def config_path(self) -> Path:
        """Path of the config snapshot file."""
        return self.metric_dir / f"{self.name}_config.json"

    def write_config(self, config: dict[str, Any], extra_path: str | Path | None = None) -> None:
        """Persist a JSON snapshot of the run configuration.

        Content is caller-defined but SHOULD include: hyperparameters, optimizer
        and scheduler settings, data transforms, seed, checkpoint paths, and a
        ``description`` explaining What/Why/How the run was produced (5W1H).

        Args:
            extra_path: Optional additional destination for the same snapshot
                (e.g. ``experiments/results/<experiment>/config.json``) so the
                serialized form is identical everywhere (single code path).
        """
        config = dict(config)
        config.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        self.config_path.write_text(payload, encoding="utf-8")
        if extra_path is not None:
            Path(extra_path).parent.mkdir(parents=True, exist_ok=True)
            Path(extra_path).write_text(payload, encoding="utf-8")
        self.log(f"[config] written to {self.config_path.relative_to(self.run_dir)}")

    def append_metrics(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Append one JSON line per epoch to the JSONL history file."""
        record = {"epoch": epoch,
                  "timestamp": datetime.now().isoformat(timespec="seconds"),
                  **metrics}
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_history(self) -> list[dict[str, Any]]:
        """Return all JSONL history records (epoch order)."""
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in
                self.history_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ----------------------------------------------------------------- console
    def log(self, msg: str, level: str = "INFO", console: bool | None = None) -> None:
        """Append a timestamped line to the log file; optionally print to stdout."""
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}"
        self._maybe_rotate()
        self._log_handle.write(line + "\n")
        self._log_handle.flush()
        if (self.console if console is None else console):
            self._clear_progress()
            print(line, flush=True)

    def progress(self, text: str) -> None:
        """Overwrite the current console line with *text* (no log-file spam)."""
        if self.console:
            self._progress_active = True
            sys.stdout.write("\r" + text.ljust(80))
            sys.stdout.flush()

    def end_progress(self) -> None:
        """Terminate the active progress line with a newline."""
        if self._progress_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_active = False

    def epoch_summary(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Print a formatted per-epoch line, append it to history and to the log."""
        self.end_progress()
        parts = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items())
        self.log(f"epoch {epoch:3d}  {parts}", level="TRAIN")
        self.append_metrics(epoch, metrics)

    # ------------------------------------------------------------- tensorboard
    @property
    def tb_writer(self):
        """TensorBoard SummaryWriter for this run, or None if unavailable.

        The lightweight file/console logging works regardless; TensorBoard is an
        optional enhancement for visual monitoring.
        """
        if self._tb_writer is not None:
            return self._tb_writer
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self._tb_writer = SummaryWriter(log_dir=str(self.tb_dir))
            self.log("[tb] TensorBoard writer enabled — "
                     f"run `tensorboard --logdir {self.tb_dir.parent}`")
        except Exception as exc:  # pragma: no cover - optional dependency
            self._tb_writer = False
            self.log(f"[tb] TensorBoard unavailable ({exc}); using console/file only")
        return self._tb_writer if self._tb_writer else None

    # ---------------------------------------------------------------- closing
    def close(self) -> None:
        """Flush and close the log file; no-op afterwards."""
        self.end_progress()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
            self.log_path_guard()

    def log_path_guard(self) -> None:  # pragma: no cover - trivial helper
        pass

    # ---------------------------------------------------------------- rotation
    def _maybe_rotate(self) -> None:
        if self._log_path.stat().st_size < LOG_ROTATE_BYTES:
            return
        self._log_handle.close()
        for i in range(LOG_ROTATE_KEEP - 1, 0, -1):
            src = self._log_path.with_suffix(f".{i}.log.gz")
            dst = self._log_path.with_suffix(f".{i + 1}.log.gz")
            if src.exists():
                dst.unlink(missing_ok=True)
                src.rename(dst)
        with open(self._log_path, "rb") as raw:
            with gzip.open(self._log_path.with_suffix(".1.log.gz"), "wb") as gz:
                gz.write(raw.read())
        self._log_handle = open(self._log_path, "w", encoding="utf-8")  # noqa: SIM115
        self.log("[rotation] log file archived (.log.gz) and truncated")

    def _clear_progress(self) -> None:
        if self._progress_active:
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
            self._progress_active = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitize(name: str) -> str:
    """Make *name* filesystem-safe while keeping it readable."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def format_eta(seconds: float) -> str:
    """Format *seconds* as ``Xd Xh Xm Xs`` (used by progress lines)."""
    seconds = max(0.0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {mins}m {secs}s"
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def elapsed_str(start: float) -> str:
    """Short human-readable elapsed time from *start* (time.perf_counter)."""
    return format_eta(time.perf_counter() - start)

