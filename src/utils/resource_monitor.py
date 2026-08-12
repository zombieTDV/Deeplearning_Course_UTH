"""resource_monitor — lightweight GPU/VRAM monitoring.

Reports peak VRAM usage against the project's ≤3.5 GB target (4 GB ceiling,
PYTORCH_FRAMEWORK_RULES.md §1.2 and roadmap risk R1).
"""

from __future__ import annotations

import threading

import torch

VRAM_TARGET_GB = 3.5
VRAM_CEILING_GB = 4.0


class ResourceMonitor:
    """Tracks peak CUDA VRAM allocation while a training/eval script runs."""

    def __init__(self, target_gb: float = VRAM_TARGET_GB, ceiling_gb: float = VRAM_CEILING_GB) -> None:
        self.target_gb = target_gb
        self.ceiling_gb = ceiling_gb
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0
        self._lock = threading.Lock()

    def sample(self) -> None:
        if not torch.cuda.is_available():
            return
        with self._lock:
            self.peak_allocated_mb = max(
                self.peak_allocated_mb, torch.cuda.memory_allocated() / (1024 * 1024)
            )
            self.peak_reserved_mb = max(
                self.peak_reserved_mb, torch.cuda.memory_reserved() / (1024 * 1024)
            )

    def start_background(self) -> None:
        """Sample VRAM in a daemon thread so peak usage is captured during training."""
        if not torch.cuda.is_available():
            return

        def loop() -> None:
            while True:
                self.sample()
                threading.Event().wait(1.0)

        threading.Thread(target=loop, daemon=True).start()

    def summary(self) -> dict:
        return {
            "peak_allocated_mb": round(self.peak_allocated_mb, 2),
            "peak_reserved_mb": round(self.peak_reserved_mb, 2),
            "vram_target_gb": self.target_gb,
            "vram_ceiling_gb": self.ceiling_gb,
            "vram_within_target": self.peak_reserved_mb <= self.target_gb * 1024,
        }

    def report(self) -> str:
        s = self.summary()
        ok = s["vram_within_target"]
        return (
            f"Peak VRAM: {s['peak_reserved_mb']:.0f} MiB reserved / "
            f"{s['peak_allocated_mb']:.0f} MiB allocated "
            f"(target ≤{s['vram_target_gb']} GB) — {'OK' if ok else 'OVER TARGET'}"
        )
