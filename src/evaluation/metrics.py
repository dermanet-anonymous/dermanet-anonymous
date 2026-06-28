"""Standard multiclass metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_multiclass_metrics(targets: Sequence[int], probabilities: np.ndarray) -> dict[str, float]:
    """Compute aggregate metrics from global class probabilities."""
    targets_array = np.asarray(targets, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if probabilities_array.ndim != 2 or probabilities_array.shape[0] != targets_array.shape[0]:
        raise ValueError("probabilities must have shape [num_samples, num_classes].")

    predictions = probabilities_array.argmax(axis=1)
    labels = np.arange(probabilities_array.shape[1])
    metrics = {
        "accuracy": float(accuracy_score(targets_array, predictions)),
        "balanced_accuracy": float(recall_score(targets_array, predictions, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets_array, predictions, labels=labels, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(targets_array, predictions, labels=labels, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(targets_array, predictions, labels=labels, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(targets_array, predictions, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets_array, predictions, labels=labels, average="macro", zero_division=0)),
    }
    try:
        metrics["auc_macro_ovr"] = float(
            roc_auc_score(targets_array, probabilities_array, labels=labels, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["auc_macro_ovr"] = float("nan")
    return metrics
