"""Error Auditor for Misclassification Analysis."""

from pathlib import Path
class ErrorAuditor:
    """High-level class for misclassification error audit."""

    @staticmethod
    def audit_top_misclassifications(run_root: str | Path = "experiments/runs") -> None:
        """Run top false positive and false negative root cause audit."""
        try:
            import scratch.analyze_misclassifications as am
            am.analyze_errors()
        except ImportError:
            print("scratch.analyze_misclassifications module not available.")

