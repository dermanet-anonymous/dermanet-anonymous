"""Multi-level depth aggregation and paired-view feature fusion for DermaNet."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class DepthGatedAggregator(nn.Module):
    """
    Input-dependent aggregation of feature embeddings from multiple backbone depths.

    Each depth embedding must already be projected into the same dimensionality.
    For DermaNet, three embeddings from levels 2, 3, and 4 are combined using
    softmax-normalized, sample-specific weights.
    """

    def __init__(self, embedding_dim: int = 512, num_levels: int = 3) -> None:
        super().__init__()

        if num_levels < 2:
            raise ValueError("num_levels must be at least 2.")

        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.gate = nn.Linear(embedding_dim * num_levels, num_levels)

    def forward(self, level_embeddings: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Args:
            level_embeddings: Sequence of tensors with shape [batch_size, embedding_dim].
                The expected DermaNet order is [level_2, level_3, level_4].

        Returns:
            aggregated_embedding: Tensor of shape [batch_size, embedding_dim].
            depth_weights: Tensor of shape [batch_size, num_levels].
        """
        if len(level_embeddings) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} level embeddings, "
                f"but received {len(level_embeddings)}."
            )

        batch_size = level_embeddings[0].shape[0]

        for index, embedding in enumerate(level_embeddings):
            if embedding.ndim != 2:
                raise ValueError(
                    f"Level embedding {index} must have shape [batch, features], "
                    f"but received shape {tuple(embedding.shape)}."
                )

            if embedding.shape[0] != batch_size:
                raise ValueError("All level embeddings must share the same batch size.")

            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Level embedding {index} has feature dimension "
                    f"{embedding.shape[1]}, expected {self.embedding_dim}."
                )

        concatenated = torch.cat(level_embeddings, dim=1)
        depth_weights = torch.softmax(self.gate(concatenated), dim=1)

        stacked_embeddings = torch.stack(level_embeddings, dim=1)
        aggregated_embedding = torch.sum(
            depth_weights.unsqueeze(-1) * stacked_embeddings,
            dim=1,
        )

        return aggregated_embedding, depth_weights


class FusionMLP(nn.Module):
    """
    Feature-level fusion network for paired dermoscopic and clinical embeddings.

    Default architecture:
        1024 -> 512 -> 256
    with BatchNorm, SiLU, and dropout after the hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, fused_input: Tensor) -> Tensor:
        """Map a concatenated paired-view embedding to a shared representation."""
        return self.network(fused_input)


class PairedFeatureFusion(nn.Module):
    """
    Concatenate dermoscopic and clinical embeddings, then produce a shared feature.

    Both inputs must have the same embedding dimension. Missing-view masking is
    intentionally handled outside this module by the modality-dropout component.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        fused_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.fusion_mlp = FusionMLP(
            input_dim=2 * embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=fused_dim,
            dropout=dropout,
        )

    def forward(
        self,
        dermoscopic_embedding: Tensor,
        clinical_embedding: Tensor,
    ) -> Tensor:
        """Fuse paired dermoscopic and clinical embeddings."""
        if dermoscopic_embedding.shape != clinical_embedding.shape:
            raise ValueError(
                "Dermoscopic and clinical embeddings must have identical shapes. "
                f"Received {tuple(dermoscopic_embedding.shape)} and "
                f"{tuple(clinical_embedding.shape)}."
            )

        if dermoscopic_embedding.ndim != 2:
            raise ValueError(
                "Embeddings must have shape [batch_size, embedding_dim]."
            )

        if dermoscopic_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"received {dermoscopic_embedding.shape[1]}."
            )

        concatenated = torch.cat(
            [dermoscopic_embedding, clinical_embedding],
            dim=1,
        )
        return self.fusion_mlp(concatenated)
