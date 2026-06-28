"""Hierarchy-specific inference and diagnostic utilities for DermaNet."""

from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.hierarchy import MelanocyticHierarchy


def get_hierarchy_probabilities(
    hierarchy_outputs: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """
    Convert hierarchy logits into probabilities.

    Returns:
        group_probabilities: [batch_size, 2]
        mc_probabilities: [batch_size, num_mc_classes]
        nonmc_probabilities: [batch_size, num_nonmc_classes]
    """
    required_keys = {"group_logits", "mc_logits", "nonmc_logits"}

    if set(hierarchy_outputs.keys()) != required_keys:
        raise ValueError(
            f"Expected hierarchy outputs with keys {required_keys}, "
            f"received {set(hierarchy_outputs.keys())}."
        )

    return {
        "group_probabilities": F.softmax(
            hierarchy_outputs["group_logits"],
            dim=1,
        ),
        "mc_probabilities": F.softmax(
            hierarchy_outputs["mc_logits"],
            dim=1,
        ),
        "nonmc_probabilities": F.softmax(
            hierarchy_outputs["nonmc_logits"],
            dim=1,
        ),
    }


@torch.no_grad()
def predict_hard_routing(
    hierarchy: MelanocyticHierarchy,
    hierarchy_outputs: Dict[str, Tensor],
    threshold: float = 0.5,
) -> Tensor:
    """
    Predict global class indices using threshold-based hard routing.

    Samples with p(MC | x) >= threshold are classified by the MC head.
    All remaining samples are classified by the NonMC head.
    """
    return hierarchy.hard_predict(
        outputs=hierarchy_outputs,
        threshold=threshold,
    )


@torch.no_grad()
def predict_soft_stitching(
    hierarchy: MelanocyticHierarchy,
    hierarchy_outputs: Dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Predict using soft probability stitching.

    Returns:
        predictions:
            Global predicted class indices.

        stitched_probabilities:
            Full global class probability matrix with shape
            [batch_size, num_classes].
    """
    stitched_probabilities = hierarchy.soft_probabilities(
        hierarchy_outputs,
    )
    predictions = stitched_probabilities.argmax(dim=1)

    return predictions, stitched_probabilities


@torch.no_grad()
def get_hierarchy_predictions(
    hierarchy: MelanocyticHierarchy,
    hierarchy_outputs: Dict[str, Tensor],
    mode: Literal["hard", "soft"] = "hard",
    threshold: float = 0.5,
) -> Dict[str, Tensor]:
    """
    Generate predictions and diagnostic probabilities for either inference mode.

    Soft stitched probabilities are always returned because they are needed for
    ranking-based metrics such as one-vs-rest AUC.
    """
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be either 'hard' or 'soft'.")

    probabilities = get_hierarchy_probabilities(hierarchy_outputs)

    soft_predictions, stitched_probabilities = predict_soft_stitching(
        hierarchy=hierarchy,
        hierarchy_outputs=hierarchy_outputs,
    )

    hard_predictions = predict_hard_routing(
        hierarchy=hierarchy,
        hierarchy_outputs=hierarchy_outputs,
        threshold=threshold,
    )

    final_predictions = (
        hard_predictions if mode == "hard" else soft_predictions
    )

    return {
        "predictions": final_predictions,
        "hard_predictions": hard_predictions,
        "soft_predictions": soft_predictions,
        "stitched_probabilities": stitched_probabilities,
        **probabilities,
    }


@torch.no_grad()
def group_targets_from_labels(
    hierarchy: MelanocyticHierarchy,
    targets: Tensor,
) -> Tensor:
    """
    Map global class labels to group labels.

    Group encoding:
        0 = melanocytic
        1 = non-melanocytic
    """
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch_size].")

    targets = targets.long()

    if torch.any(targets < 0) or torch.any(targets >= hierarchy.num_classes):
        raise ValueError("targets contain invalid global class indices.")

    return torch.where(
        hierarchy.is_mc_class[targets],
        torch.zeros_like(targets),
        torch.ones_like(targets),
    )


@torch.no_grad()
def hierarchy_diagnostics(
    hierarchy: MelanocyticHierarchy,
    hierarchy_outputs: Dict[str, Tensor],
    targets: Tensor,
) -> Dict[str, float]:
    """
    Compute hierarchy-specific diagnostic values.

    Reports:
        group_accuracy:
            Accuracy of the MC vs. NonMC group head.

        mc_conditional_accuracy:
            Accuracy of the MC sub-head on ground-truth MC samples.

        nonmc_conditional_accuracy:
            Accuracy of the NonMC sub-head on ground-truth NonMC samples.

        routing_accuracy:
            Fraction of samples routed to their true group using tau=0.5.
    """
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch_size].")

    targets = targets.long()
    group_targets = group_targets_from_labels(hierarchy, targets)

    group_logits = hierarchy_outputs["group_logits"]
    mc_logits = hierarchy_outputs["mc_logits"]
    nonmc_logits = hierarchy_outputs["nonmc_logits"]

    if group_logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "The number of hierarchy outputs must match the number of targets."
        )

    group_predictions = group_logits.argmax(dim=1)
    group_accuracy = (group_predictions == group_targets).float().mean()

    mc_mask = group_targets == 0
    nonmc_mask = group_targets == 1

    diagnostics: Dict[str, float] = {
        "group_accuracy": float(group_accuracy.item()),
        "mc_conditional_accuracy": float("nan"),
        "nonmc_conditional_accuracy": float("nan"),
        "routing_accuracy": float("nan"),
    }

    if mc_mask.any():
        mc_targets = hierarchy.global_to_mc_local[targets[mc_mask]]
        mc_predictions = mc_logits[mc_mask].argmax(dim=1)

        diagnostics["mc_conditional_accuracy"] = float(
            (mc_predictions == mc_targets).float().mean().item()
        )

    if nonmc_mask.any():
        nonmc_targets = hierarchy.global_to_nonmc_local[targets[nonmc_mask]]
        nonmc_predictions = nonmc_logits[nonmc_mask].argmax(dim=1)

        diagnostics["nonmc_conditional_accuracy"] = float(
            (nonmc_predictions == nonmc_targets).float().mean().item()
        )

    hard_predictions = hierarchy.hard_predict(
        outputs=hierarchy_outputs,
        threshold=0.5,
    )

    routed_groups = torch.where(
        hierarchy.is_mc_class[hard_predictions],
        torch.zeros_like(hard_predictions),
        torch.ones_like(hard_predictions),
    )

    diagnostics["routing_accuracy"] = float(
        (routed_groups == group_targets).float().mean().item()
    )

    return diagnostics
