"""Evaluation metrics for multiclass DermaNet experiments."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_inputs(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate multiclass targets and probability matrix."""
    targets_array = np.asarray(targets, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)

    if targets_array.ndim != 1:
        raise ValueError("targets must have shape [num_samples].")

    if probabilities_array.ndim != 2:
        raise ValueError(
            "probabilities must have shape [num_samples, num_classes]."
        )

    if len(targets_array) != len(probabilities_array):
        raise ValueError(
            "targets and probabilities must contain the same number of samples."
        )

    if len(targets_array) == 0:
        raise ValueError("No samples were provided.")

    num_classes = probabilities_array.shape[1]

    if num_classes < 2:
        raise ValueError("At least two classes are required.")

    if np.any(targets_array < 0) or np.any(targets_array >= num_classes):
        raise ValueError("targets contain invalid class indices.")

    if not np.all(np.isfinite(probabilities_array)):
        raise ValueError("probabilities contain NaN or infinite values.")

    return targets_array, probabilities_array


def top_k_predictions(
    probabilities: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Return the top-k predicted class indices for every sample.

    Args:
        probabilities: Array with shape [num_samples, num_classes].
        k: Number of allowed predictions per sample.

    Returns:
        Integer array with shape [num_samples, k].
    """
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape [num_samples, num_classes]."
        )

    num_classes = probabilities.shape[1]

    if not 1 <= k <= num_classes:
        raise ValueError(
            f"k must be between 1 and {num_classes}, received {k}."
        )

    unsorted_indices = np.argpartition(
        probabilities,
        kth=num_classes - k,
        axis=1,
    )[:, -k:]

    row_indices = np.arange(probabilities.shape[0])[:, None]
    sorted_order = np.argsort(
        probabilities[row_indices, unsorted_indices],
        axis=1,
    )[:, ::-1]

    return unsorted_indices[row_indices, sorted_order]


def top_k_accuracy(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> float:
    """Compute the fraction of samples whose true class is among top-k."""
    targets, probabilities = _validate_inputs(targets, probabilities)
    predictions = top_k_predictions(probabilities, k=k)

    hits = (predictions == targets[:, None]).any(axis=1)
    return float(hits.mean())


def top_k_balanced_accuracy(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> float:
    """
    Compute macro recall when a prediction is counted correct if the true class
    appears in the model's top-k predictions.

    This is the class-balanced top-k measure reported as balanced accuracy@k.
    """
    targets, probabilities = _validate_inputs(targets, probabilities)
    predictions = top_k_predictions(probabilities, k=k)

    hits = (predictions == targets[:, None]).any(axis=1)
    num_classes = probabilities.shape[1]

    per_class_recalls = []

    for class_index in range(num_classes):
        class_mask = targets == class_index

        if class_mask.any():
            per_class_recalls.append(float(hits[class_mask].mean()))

    if not per_class_recalls:
        raise RuntimeError("Could not compute top-k balanced accuracy.")

    return float(np.mean(per_class_recalls))


def compute_multiclass_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """
    Compute standard DermaNet multiclass metrics.

    Args:
        targets:
            Ground-truth global class indices with shape [num_samples].

        probabilities:
            Predicted class probabilities with shape
            [num_samples, num_classes].

        class_names:
            Optional class labels. When supplied, per-class metrics are returned
            using these names.

    Returns:
        Dictionary containing aggregate metrics and per-class metrics.
    """
    targets, probabilities = _validate_inputs(targets, probabilities)

    num_classes = probabilities.shape[1]
    class_indices = np.arange(num_classes)
    predictions = probabilities.argmax(axis=1)

    if class_names is None:
        class_names = [str(index) for index in class_indices]
    elif len(class_names) != num_classes:
        raise ValueError(
            "class_names must have one entry for every model class."
        )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            recall_score(
                targets,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                targets,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_micro": float(
            f1_score(
                targets,
                predictions,
                labels=class_indices,
                average="micro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                targets,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "precision_macro": float(
            precision_score(
                targets,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                targets,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                targets,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                targets,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "top_1_accuracy": top_k_accuracy(targets, probabilities, k=1),
        "top_1_balanced_accuracy": top_k_balanced_accuracy(
            targets,
            probabilities,
            k=1,
        ),
        "top_2_accuracy": top_k_accuracy(targets, probabilities, k=2),
        "top_2_balanced_accuracy": top_k_balanced_accuracy(
            targets,
            probabilities,
            k=2,
        ),
        "top_3_accuracy": top_k_accuracy(targets, probabilities, k=3),
        "top_3_balanced_accuracy": top_k_balanced_accuracy(
            targets,
            probabilities,
            k=3,
        ),
    }

    try:
        metrics["auc_macro_ovr"] = float(
            roc_auc_score(
                targets,
                probabilities,
                labels=class_indices,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        metrics["auc_macro_ovr"] = float("nan")

    per_class: dict[str, dict[str, float]] = {}

    per_class_precision = precision_score(
        targets,
        predictions,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        targets,
        predictions,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        targets,
        predictions,
        labels=class_indices,
        average=None,
        zero_division=0,
    )

    for class_index, class_name in enumerate(class_names):
        class_targets = (targets == class_index).astype(np.int64)

        try:
            class_auc = float(
                roc_auc_score(
                    class_targets,
                    probabilities[:, class_index],
                )
            )
        except ValueError:
            class_auc = float("nan")

        per_class[str(class_name)] = {
            "support": int((targets == class_index).sum()),
            "precision": float(per_class_precision[class_index]),
            "recall": float(per_class_recall[class_index]),
            "f1": float(per_class_f1[class_index]),
            "auc_ovr": class_auc,
        }

    metrics["per_class"] = per_class
    return metrics
