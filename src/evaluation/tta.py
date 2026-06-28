"""Eight-view paired-image test-time augmentation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.models.hierarchy import HierarchySpec, soft_stitch_probabilities


@torch.no_grad()
def predict_eight_view_tta(
    model: nn.Module,
    clinical_images: Tensor,
    dermoscopic_images: Tensor,
    hierarchy: HierarchySpec,
    log_priors: Mapping[int, Tensor] | None = None,
    logit_adjustment_tau: float = 0.12,
) -> Tensor:
    """Average stitched probabilities over 4 rotations x 2 horizontal-flip states."""
    was_training = model.training
    model.eval()
    probabilities: list[Tensor] = []
    try:
        for rotation in range(4):
            clinical_rotated = torch.rot90(clinical_images, k=rotation, dims=(2, 3))
            dermoscopic_rotated = torch.rot90(dermoscopic_images, k=rotation, dims=(2, 3))
            for clinical_view, dermoscopic_view in (
                (clinical_rotated, dermoscopic_rotated),
                (torch.flip(clinical_rotated, dims=[3]), torch.flip(dermoscopic_rotated, dims=[3])),
            ):
                group_logits, mel_logits, other_logits = model(clinical_view, dermoscopic_view)
                probabilities.append(
                    soft_stitch_probabilities(
                        group_logits,
                        mel_logits,
                        other_logits,
                        hierarchy,
                        log_priors=log_priors,
                        logit_adjustment_tau=logit_adjustment_tau,
                    )
                )
    finally:
        model.train(was_training)

    return torch.stack(probabilities, dim=0).mean(dim=0)
