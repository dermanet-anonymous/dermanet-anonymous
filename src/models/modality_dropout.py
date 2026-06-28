"""Image-level modality dropout from the reference MILK10k workflow."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def apply_modality_dropout(
    clinical_images: Tensor,
    dermoscopic_images: Tensor,
    probability: float = 0.15,
    training: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Randomly zero one entire image modality for the current batch.

    The two modalities are sampled independently, then a tie-break prevents
    both streams from being dropped. This reproduces the image-level strategy
    in the reference notebook.
    """
    if not training or probability <= 0:
        return clinical_images, dermoscopic_images
    if probability >= 1:
        raise ValueError("probability must be smaller than 1.")

    drop_clinical = random.random() < probability
    drop_dermoscopic = random.random() < probability

    if drop_clinical and drop_dermoscopic:
        if random.random() < 0.5:
            drop_clinical = False
        else:
            drop_dermoscopic = False

    if drop_clinical:
        clinical_images = torch.zeros_like(clinical_images)
    if drop_dermoscopic:
        dermoscopic_images = torch.zeros_like(dermoscopic_images)

    return clinical_images, dermoscopic_images
