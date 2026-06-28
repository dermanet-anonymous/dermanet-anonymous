"""Multi-level depth aggregation and paired-view feature fusion for DermaNet."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class DepthGatedAggregator(nn.Module):
    """
    Global learned aggregation of feature embeddings from multiple backbone depths.

    Each depth embedding is projected to the same dimensionality. A single
    learnable softmax-normalized weight vector determines the contribution of
    levels 2, 3, and 4 across all samples.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_levels: int = 3,
        init_logits: Tensor | None = None,
    ) -> None:
        super().__init__()

        if num_levels < 2:
            raise ValueError("num_levels must be at least 2.")

        self.embedding_dim = embedding_dim
        self.num_levels = num_levels

        if init_logits is None:
            init_logits = torch.linspace(-1.0, 1.0, steps=num_levels)

        init_logits = torch.as_tensor(init_logits, dtype=torch.float32)

        if init_logits.numel() != num_levels:
            raise ValueError(
                f"init_logits must contain {num_levels} values."
            )

        self.weight_logits = nn.Parameter(init_logits.clone())

    def get_weights(self) -> Tensor:
        """Return the global softmax-normalized depth weights."""
        return torch.softmax(self.weight_logits, dim=0)

    def forward(self, level_embeddings: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Args:
            level_embeddings: Sequence of tensors with shape
                [batch_size, embedding_dim], ordered as [level_2, level_3, level_4].

        Returns:
            aggregated_embedding: Tensor with shape [batch_size, embedding_dim].
            depth_weights: Global weights expanded to [batch_size, num_levels].
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
                    f"Level embedding {index} must have shape [batch, features]."
                )

            if embedding.shape[0] != batch_size:
                raise ValueError("All level embeddings must share batch size.")

            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Level embedding {index} has feature dimension "
                    f"{embedding.shape[1]}, expected {self.embedding_dim}."
                )

        weights = self.get_weights()
        stacked_embeddings = torch.stack(level_embeddings, dim=1)

        aggregated_embedding = (
            stacked_embeddings * weights.view(1, self.num_levels, 1)
        ).sum(dim=1)

        expanded_weights = weights.unsqueeze(0).expand(batch_size, -1)

        return aggregated_embedding, expanded_weights


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
