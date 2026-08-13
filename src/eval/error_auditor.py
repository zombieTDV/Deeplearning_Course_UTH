"""Error Auditor for Misclassification Analysis."""

from pathlib import Path
import scratch.analyze_misclassifications


class ErrorAuditor:
    """High-level class for misclassification error audit."""

    @staticmethod
    def audit_top_misclassifications(run_root: str | Path = "experiments/runs") -> None:
        """Run top false positive and false negative root cause audit."""
        scratch.analyze_misclassifications.analyze_errors()
