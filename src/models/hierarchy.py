"""Clinically guided melanocytic/non-melanocytic hierarchy for DermaNet."""

from __future__ import annotations

from typing import Dict, Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MelanocyticHierarchy(nn.Module):
    """
    Hierarchical classifier with:

    - A binary group head: melanocytic (MC) vs. non-melanocytic (NonMC)
    - One fine-grained classifier for MC classes
    - One fine-grained classifier for NonMC classes

    Group order:
        0 -> MC
        1 -> NonMC

    The class membership is configurable so the same module can be used for
    MILK10k and ISIC2019.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        mc_class_indices: Iterable[int],
        group_loss_weight: float = 1.0,
        subgroup_loss_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        mc_indices = sorted(set(int(index) for index in mc_class_indices))

        if not mc_indices:
            raise ValueError("mc_class_indices must contain at least one class.")

        if min(mc_indices) < 0 or max(mc_indices) >= num_classes:
            raise ValueError(
                "Every melanocytic class index must be in [0, num_classes)."
            )

        nonmc_indices = [
            index for index in range(num_classes) if index not in mc_indices
        ]

        if not nonmc_indices:
            raise ValueError(
                "At least one non-melanocytic class is required."
            )

        if group_loss_weight < 0 or subgroup_loss_weight < 0:
            raise ValueError("Loss weights must be non-negative.")

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.group_loss_weight = group_loss_weight
        self.subgroup_loss_weight = subgroup_loss_weight
        self.label_smoothing = label_smoothing

        self.group_head = nn.Linear(feature_dim, 2)
        self.mc_head = nn.Linear(feature_dim, len(mc_indices))
        self.nonmc_head = nn.Linear(feature_dim, len(nonmc_indices))

        self.register_buffer(
            "mc_class_indices",
            torch.tensor(mc_indices, dtype=torch.long),
        )
        self.register_buffer(
            "nonmc_class_indices",
            torch.tensor(nonmc_indices, dtype=torch.long),
        )

        is_mc = torch.zeros(num_classes, dtype=torch.bool)
        is_mc[self.mc_class_indices] = True
        self.register_buffer("is_mc_class", is_mc)

        global_to_mc_local = torch.full(
            (num_classes,),
            fill_value=-1,
            dtype=torch.long,
        )
        global_to_mc_local[self.mc_class_indices] = torch.arange(
            len(mc_indices),
            dtype=torch.long,
        )
        self.register_buffer("global_to_mc_local", global_to_mc_local)

        global_to_nonmc_local = torch.full(
            (num_classes,),
            fill_value=-1,
            dtype=torch.long,
        )
        global_to_nonmc_local[self.nonmc_class_indices] = torch.arange(
            len(nonmc_indices),
            dtype=torch.long,
        )
        self.register_buffer(
            "global_to_nonmc_local",
            global_to_nonmc_local,
        )

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            features: Shared fused representation with shape
                [batch_size, feature_dim].

        Returns:
            Dictionary containing logits for the binary group head and both
            fine-grained subgroup heads.
        """
        if features.ndim != 2:
            raise ValueError(
                "features must have shape [batch_size, feature_dim]."
            )

        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"received {features.shape[1]}."
            )

        return {
            "group_logits": self.group_head(features),
            "mc_logits": self.mc_head(features),
            "nonmc_logits": self.nonmc_head(features),
        }

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute the joint group and subgroup classification objective.

        Each sample updates:
        - the binary group head; and
        - only its ground-truth subgroup classifier.
        """
        self._validate_outputs(outputs)

        if targets.ndim != 1:
            raise ValueError("targets must have shape [batch_size].")

        targets = targets.long()

        if torch.any(targets < 0) or torch.any(targets >= self.num_classes):
            raise ValueError("targets contain invalid global class indices.")

        group_targets = torch.where(
            self.is_mc_class[targets],
            torch.zeros_like(targets),
            torch.ones_like(targets),
        )

        mc_mask = self.is_mc_class[targets]
        nonmc_mask = ~mc_mask

        group_loss = F.cross_entropy(
            outputs["group_logits"],
            group_targets,
            label_smoothing=self.label_smoothing,
        )

        zero = outputs["group_logits"].sum() * 0.0

        if mc_mask.any():
            mc_targets = self.global_to_mc_local[targets[mc_mask]]
            mc_loss = F.cross_entropy(
                outputs["mc_logits"][mc_mask],
                mc_targets,
                label_smoothing=self.label_smoothing,
            )
        else:
            mc_loss = zero

        if nonmc_mask.any():
            nonmc_targets = self.global_to_nonmc_local[targets[nonmc_mask]]
            nonmc_loss = F.cross_entropy(
                outputs["nonmc_logits"][nonmc_mask],
                nonmc_targets,
                label_smoothing=self.label_smoothing,
            )
        else:
            nonmc_loss = zero

        total_loss = (
            self.group_loss_weight * group_loss
            + self.subgroup_loss_weight * (mc_loss + nonmc_loss)
        )

        return {
            "loss": total_loss,
            "group_loss": group_loss,
            "mc_loss": mc_loss,
            "nonmc_loss": nonmc_loss,
        }

    @torch.no_grad()
    def hard_predict(
        self,
        outputs: Dict[str, Tensor],
        threshold: float = 0.5,
    ) -> Tensor:
        """
        Predict with hard group routing.

        A sample is routed to the MC head when:
            p(MC | x) >= threshold

        Otherwise it is routed to the NonMC head.
        """
        self._validate_outputs(outputs)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")

        group_probabilities = torch.softmax(outputs["group_logits"], dim=1)
        mc_probabilities = group_probabilities[:, 0]

        mc_selected = mc_probabilities >= threshold

        mc_predictions = self.mc_class_indices[
            outputs["mc_logits"].argmax(dim=1)
        ]
        nonmc_predictions = self.nonmc_class_indices[
            outputs["nonmc_logits"].argmax(dim=1)
        ]

        return torch.where(
            mc_selected,
            mc_predictions,
            nonmc_predictions,
        )

    @torch.no_grad()
    def soft_probabilities(
        self,
        outputs: Dict[str, Tensor],
    ) -> Tensor:
        """
        Return full class probabilities using soft probability stitching:

            p(y | x) = p(g(y) | x) * p(y | x, g(y))
        """
        self._validate_outputs(outputs)

        group_probabilities = torch.softmax(outputs["group_logits"], dim=1)
        mc_probabilities = torch.softmax(outputs["mc_logits"], dim=1)
        nonmc_probabilities = torch.softmax(outputs["nonmc_logits"], dim=1)

        batch_size = group_probabilities.shape[0]

        full_probabilities = torch.zeros(
            batch_size,
            self.num_classes,
            device=group_probabilities.device,
            dtype=group_probabilities.dtype,
        )

        full_probabilities[:, self.mc_class_indices] = (
            group_probabilities[:, 0:1] * mc_probabilities
        )
        full_probabilities[:, self.nonmc_class_indices] = (
            group_probabilities[:, 1:2] * nonmc_probabilities
        )

        return full_probabilities

    @torch.no_grad()
    def soft_predict(self, outputs: Dict[str, Tensor]) -> Tensor:
        """Predict the global class with the largest stitched probability."""
        return self.soft_probabilities(outputs).argmax(dim=1)

    @torch.no_grad()
    def predict(
        self,
        outputs: Dict[str, Tensor],
        mode: Literal["hard", "soft"] = "hard",
        threshold: float = 0.5,
    ) -> Tensor:
        """Run either hard routing or soft probability stitching."""
        if mode == "hard":
            return self.hard_predict(outputs, threshold=threshold)

        if mode == "soft":
            return self.soft_predict(outputs)

        raise ValueError("mode must be either 'hard' or 'soft'.")

    def _validate_outputs(self, outputs: Dict[str, Tensor]) -> None:
        required_keys = {"group_logits", "mc_logits", "nonmc_logits"}

        if set(outputs.keys()) != required_keys:
            raise ValueError(
                f"outputs must contain exactly {required_keys}, "
                f"received {set(outputs.keys())}."
            )

        batch_size = outputs["group_logits"].shape[0]

        expected_shapes = {
            "group_logits": 2,
            "mc_logits": len(self.mc_class_indices),
            "nonmc_logits": len(self.nonmc_class_indices),
        }

        for name, expected_classes in expected_shapes.items():
            tensor = outputs[name]

            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must have shape [batch_size, classes]."
                )

            if tensor.shape[0] != batch_size:
                raise ValueError(
                    "All hierarchy outputs must share the same batch size."
                )

            if tensor.shape[1] != expected_classes:
                raise ValueError(
                    f"{name} has {tensor.shape[1]} output classes, "
                    f"expected {expected_classes}."
                )
