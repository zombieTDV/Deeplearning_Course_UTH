"""Unit tests for IMDBCleanlabAuditor."""

import numpy as np
from src.data.cleanlab_denoiser import IMDBCleanlabAuditor


def test_cleanlab_audit_mock():
    auditor = IMDBCleanlabAuditor()
    # Mock probabilities: sample 0 is confident pos (0.95), sample 1 is confident neg (0.05)
    # Ground truth: sample 0 is labeled 0 (noisy!), sample 1 is labeled 0 (correct)
    labels = np.array([0, 0, 1, 1])
    pred_probs = np.array([
        [0.05, 0.95],  # Mislabeled (given 0, true 1)
        [0.90, 0.10],  # Correct (given 0, pred 0)
        [0.88, 0.12],  # Mislabeled (given 1, true 0)
        [0.10, 0.90],  # Correct (given 1, pred 1)
    ])
    texts = [
        "This is an amazing masterpiece! Loved everything.",
        "Terrible waste of time. Horrible acting.",
        "Worst movie ever made. Total trash.",
        "Brilliant acting and stunning cinematography.",
    ]

    results = auditor.audit_label_errors(pred_probs, labels, texts, output_json="experiments/results/test_cleanlab.json")
    assert results["total_samples_audited"] == 4
    assert results["total_label_issues_found"] >= 1
    assert "top_issues" in results
