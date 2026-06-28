"""Melanocytic/non-melanocytic hierarchy and soft probability stitching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class HierarchySpec:
    """Global class membership for the two-group MILK10k hierarchy."""

    groups: Mapping[int, tuple[int, ...]]

    @classmethod
    def milk10k(cls) -> "HierarchySpec":
        # MEL=7 and NV=8 in the reference class order.
        return cls(groups={0: (7, 8), 1: (0, 1, 2, 3, 4, 5, 6, 9, 10)})

    @property
    def num_classes(self) -> int:
        return sum(len(indices) for indices in self.groups.values())

    @property
    def sub_head_sizes(self) -> dict[int, int]:
        return {group_id: len(indices) for group_id, indices in self.groups.items()}

    def group_target(self, targets: Tensor) -> Tensor:
        result = torch.empty_like(targets)
        for group_id, class_indices in self.groups.items():
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for class_index in class_indices:
                mask |= targets == class_index
            result[mask] = group_id
        return result

    def sub_target(self, targets: Tensor) -> Tensor:
        result = torch.empty_like(targets)
        for group_id, class_indices in self.groups.items():
            for local_index, class_index in enumerate(class_indices):
                result[targets == class_index] = local_index
        return result


def soft_stitch_probabilities(
    group_logits: Tensor,
    melanocytic_logits: Tensor,
    other_logits: Tensor,
    hierarchy: HierarchySpec,
    temperature: float = 1.0,
    log_priors: Mapping[int, Tensor] | None = None,
    logit_adjustment_tau: float = 0.0,
) -> Tensor:
    """
    Produce global class probabilities with hierarchical multiplication:

        p(y | x) = p(group(y) | x) p(y | x, group(y)).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if log_priors is not None and logit_adjustment_tau != 0.0:
        melanocytic_logits = melanocytic_logits - logit_adjustment_tau * log_priors[0]
        other_logits = other_logits - logit_adjustment_tau * log_priors[1]

    group_probabilities = F.softmax(group_logits / temperature, dim=1)
    melanocytic_probabilities = F.softmax(melanocytic_logits / temperature, dim=1)
    other_probabilities = F.softmax(other_logits / temperature, dim=1)

    batch_size = group_logits.shape[0]
    final = torch.zeros(batch_size, hierarchy.num_classes, dtype=group_logits.dtype, device=group_logits.device)

    for local_index, class_index in enumerate(hierarchy.groups[0]):
        final[:, class_index] = group_probabilities[:, 0] * melanocytic_probabilities[:, local_index]
    for local_index, class_index in enumerate(hierarchy.groups[1]):
        final[:, class_index] = group_probabilities[:, 1] * other_probabilities[:, local_index]

    return final
