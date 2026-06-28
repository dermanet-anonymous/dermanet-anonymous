"""Training-time modality dropout for paired dermoscopic--clinical inputs."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class ModalityDropout(nn.Module):
    """
    Randomly mask one available modality during training.

    For paired samples, the mask is sampled in dermoscopic--clinical order:

        (0, 1) with probability rho
        (1, 0) with probability rho
        (1, 1) with probability 1 - 2 * rho

    Existing single-view availability is always preserved. For example, a
    dermoscopy-only sample with availability (1, 0) remains dermoscopy-only.

    The module operates on modality embeddings after each modality-specific
    backbone and depth aggregation stage.
    """

    def __init__(self, rho: float = 0.1) -> None:
        super().__init__()

        if not 0.0 <= rho < 0.5:
            raise ValueError("rho must be in the range [0.0, 0.5).")

        self.rho = rho

    def forward(
        self,
        dermoscopic_embedding: Tensor,
        clinical_embedding: Tensor,
        availability: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            dermoscopic_embedding:
                Tensor with shape [batch_size, embedding_dim].

            clinical_embedding:
                Tensor with shape [batch_size, embedding_dim].

            availability:
                Optional binary tensor with shape [batch_size, 2], in
                dermoscopic--clinical order. A value of 1 means that the
                modality is available. If omitted, all samples are treated
                as paired.

        Returns:
            masked_dermoscopic_embedding:
                Dermoscopic embedding after availability and dropout masking.

            masked_clinical_embedding:
                Clinical embedding after availability and dropout masking.

            effective_mask:
                Binary tensor with shape [batch_size, 2] representing the
                modality availability actually used for fusion.
        """
        self._validate_embeddings(
            dermoscopic_embedding=dermoscopic_embedding,
            clinical_embedding=clinical_embedding,
        )

        batch_size = dermoscopic_embedding.shape[0]
        device = dermoscopic_embedding.device
        dtype = dermoscopic_embedding.dtype

        availability = self._prepare_availability(
            availability=availability,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        effective_mask = availability.clone()

        if self.training and self.rho > 0.0:
            paired_samples = availability.sum(dim=1) == 2

            if paired_samples.any():
                random_values = torch.rand(
                    batch_size,
                    device=device,
                )

                drop_dermoscopic = paired_samples & (
                    random_values < self.rho
                )
                drop_clinical = paired_samples & (
                    (random_values >= self.rho)
                    & (random_values < 2.0 * self.rho)
                )

                effective_mask[drop_dermoscopic, 0] = 0.0
                effective_mask[drop_clinical, 1] = 0.0

        masked_dermoscopic_embedding = (
            dermoscopic_embedding * effective_mask[:, 0:1]
        )
        masked_clinical_embedding = (
            clinical_embedding * effective_mask[:, 1:2]
        )

        return (
            masked_dermoscopic_embedding,
            masked_clinical_embedding,
            effective_mask,
        )

    @staticmethod
    def _validate_embeddings(
        dermoscopic_embedding: Tensor,
        clinical_embedding: Tensor,
    ) -> None:
        if dermoscopic_embedding.ndim != 2:
            raise ValueError(
                "dermoscopic_embedding must have shape "
                "[batch_size, embedding_dim]."
            )

        if clinical_embedding.ndim != 2:
            raise ValueError(
                "clinical_embedding must have shape "
                "[batch_size, embedding_dim]."
            )

        if dermoscopic_embedding.shape != clinical_embedding.shape:
            raise ValueError(
                "Dermoscopic and clinical embeddings must have identical "
                "shapes."
            )

        if dermoscopic_embedding.device != clinical_embedding.device:
            raise ValueError(
                "Dermoscopic and clinical embeddings must be on the same "
                "device."
            )

    @staticmethod
    def _prepare_availability(
        availability: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if availability is None:
            return torch.ones(
                batch_size,
                2,
                device=device,
                dtype=dtype,
            )

        if availability.ndim != 2 or availability.shape != (batch_size, 2):
            raise ValueError(
                "availability must have shape [batch_size, 2] in "
                "dermoscopic--clinical order."
            )

        availability = availability.to(device=device, dtype=dtype)

        if not torch.all(
            (availability == 0.0) | (availability == 1.0)
        ):
            raise ValueError(
                "availability must contain only binary values: 0 or 1."
            )

        if torch.any(availability.sum(dim=1) == 0):
            raise ValueError(
                "Each sample must contain at least one available modality."
            )

        return availability
