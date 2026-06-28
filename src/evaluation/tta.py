"""Test-time augmentation utilities for DermaNet."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def d4_views(images: Tensor) -> Iterator[Tensor]:
    """
    Yield eight geometric views of a batch of images.

    Views:
        rotations of 0, 90, 180, and 270 degrees;
        the same four rotations after a horizontal flip.

    Args:
        images: Tensor with shape [batch_size, channels, height, width].
    """
    if images.ndim != 4:
        raise ValueError(
            "images must have shape [batch_size, channels, height, width]."
        )

    for rotation_k in range(4):
        rotated = torch.rot90(images, k=rotation_k, dims=(2, 3))
        yield rotated

    flipped = torch.flip(images, dims=(3,))

    for rotation_k in range(4):
        rotated = torch.rot90(flipped, k=rotation_k, dims=(2, 3))
        yield rotated


@torch.no_grad()
def aggregate_tta_hierarchy_outputs(
    model: nn.Module,
    dermoscopic_images: Optional[Tensor] = None,
    clinical_images: Optional[Tensor] = None,
    availability: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """
    Run eight-view TTA and average probability distributions from each head.

    Both views of a paired sample receive the same geometric transformation,
    preserving their spatial correspondence.

    Returns:
        A hierarchy-output dictionary whose tensors are log-probabilities.
        Applying softmax to these outputs recovers averaged probabilities.
    """
    if dermoscopic_images is None and clinical_images is None:
        raise ValueError(
            "At least one of dermoscopic_images or clinical_images is required."
        )

    if dermoscopic_images is not None:
        batch_size = dermoscopic_images.shape[0]
        device = dermoscopic_images.device
    else:
        batch_size = clinical_images.shape[0]
        device = clinical_images.device

    if (
        dermoscopic_images is not None
        and clinical_images is not None
        and dermoscopic_images.shape[0] != clinical_images.shape[0]
    ):
        raise ValueError(
            "Dermoscopic and clinical image batches must have equal sizes."
        )

    dermoscopic_views = (
        list(d4_views(dermoscopic_images))
        if dermoscopic_images is not None
        else [None] * 8
    )
    clinical_views = (
        list(d4_views(clinical_images))
        if clinical_images is not None
        else [None] * 8
    )

    if len(dermoscopic_views) != len(clinical_views):
        raise RuntimeError("TTA view counts do not match.")

    was_training = model.training
    model.eval()

    group_probabilities: Optional[Tensor] = None
    mc_probabilities: Optional[Tensor] = None
    nonmc_probabilities: Optional[Tensor] = None

    try:
        for dermoscopic_view, clinical_view in zip(
            dermoscopic_views,
            clinical_views,
        ):
            outputs: dict[str, Any] = model(
                dermoscopic_images=dermoscopic_view,
                clinical_images=clinical_view,
                availability=availability,
            )

            hierarchy_outputs = outputs["hierarchy"]

            group_view = F.softmax(
                hierarchy_outputs["group_logits"],
                dim=1,
            )
            mc_view = F.softmax(
                hierarchy_outputs["mc_logits"],
                dim=1,
            )
            nonmc_view = F.softmax(
                hierarchy_outputs["nonmc_logits"],
                dim=1,
            )

            if group_probabilities is None:
                group_probabilities = torch.zeros_like(group_view)
                mc_probabilities = torch.zeros_like(mc_view)
                nonmc_probabilities = torch.zeros_like(nonmc_view)

            group_probabilities += group_view
            mc_probabilities += mc_view
            nonmc_probabilities += nonmc_view
    finally:
        model.train(was_training)

    num_views = len(dermoscopic_views)

    if (
        group_probabilities is None
        or mc_probabilities is None
        or nonmc_probabilities is None
    ):
        raise RuntimeError("No TTA predictions were produced.")

    epsilon = torch.finfo(group_probabilities.dtype).eps

    group_probabilities /= num_views
    mc_probabilities /= num_views
    nonmc_probabilities /= num_views

    return {
        "group_logits": torch.log(group_probabilities.clamp_min(epsilon)),
        "mc_logits": torch.log(mc_probabilities.clamp_min(epsilon)),
        "nonmc_logits": torch.log(nonmc_probabilities.clamp_min(epsilon)),
    }


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    dermoscopic_images: Optional[Tensor] = None,
    clinical_images: Optional[Tensor] = None,
    availability: Optional[Tensor] = None,
    mode: Literal["hard", "soft"] = "hard",
    threshold: float = 0.5,
) -> dict[str, Tensor]:
    """
    Generate eight-view TTA predictions.

    Args:
        model: DermaNet model.
        dermoscopic_images: Optional dermoscopic image batch.
        clinical_images: Optional clinical image batch.
        availability: Optional [batch_size, 2] modality-availability mask.
        mode: ``"hard"`` for threshold-based routing or ``"soft"`` for
            probability stitching.
        threshold: Melanocytic routing threshold used for hard inference.

    Returns:
        Dictionary with:
            predictions:
                Final class predictions.

            stitched_probabilities:
                Full class probabilities from soft probability stitching.
                These are useful for AUC computation.

            hierarchy_outputs:
                Averaged hierarchy probabilities represented as log-probabilities.
    """
    if not hasattr(model, "hierarchy"):
        raise AttributeError(
            "The supplied model must expose a `hierarchy` attribute."
        )

    hierarchy_outputs = aggregate_tta_hierarchy_outputs(
        model=model,
        dermoscopic_images=dermoscopic_images,
        clinical_images=clinical_images,
        availability=availability,
    )

    predictions = model.hierarchy.predict(
        outputs=hierarchy_outputs,
        mode=mode,
        threshold=threshold,
    )

    stitched_probabilities = model.hierarchy.soft_probabilities(
        hierarchy_outputs,
    )

    return {
        "predictions": predictions,
        "stitched_probabilities": stitched_probabilities,
        "hierarchy_outputs": hierarchy_outputs,
    }
