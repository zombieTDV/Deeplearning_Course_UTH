"""Error Auditor for Misclassification Analysis."""

from pathlib import Path


class ErrorAuditor:
    """High-level class for misclassification error audit."""

    @staticmethod
    def audit_top_misclassifications(run_root: str | Path = "experiments/runs") -> None:
        """Run top false positive and false negative root cause audit.

        The underlying implementation lives in the optional
        ``scratch.analyze_misclassifications`` helper module, which is not part
        of the maintained pipeline. It is imported lazily so that importing this
        module (or the ``src`` package) never fails when the helper is absent;
        calling this method without the helper raises a clear error instead.
        """
        try:
            import scratch.analyze_misclassifications as _analyzer
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "ErrorAuditor requires the optional helper module "
                "'scratch.analyze_misclassifications', which is not present in "
                "this checkout. Add it (or remove the ErrorAuditor call) to run "
                "the misclassification audit."
            ) from err
        _analyzer.analyze_errors()
