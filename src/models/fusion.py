"""Global depth aggregation and paired-view fusion blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AttentionGatedFusion(nn.Module):
    """
    Project feature maps from multiple backbone depths and combine them using
    one global learned softmax-normalized weight vector.

    Despite the historical class name, the weights are shared across samples.
    """

    def __init__(self, channels_list: Sequence[int], embedding_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        if len(channels_list) < 2:
            raise ValueError("At least two feature levels are required.")

        self.embedding_dim = embedding_dim
        self.num_levels = len(channels_list)
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(channels, embedding_dim),
                    nn.BatchNorm1d(embedding_dim),
                    nn.SiLU(),
                )
                for channels in channels_list
            ]
        )
        self.weight_logits = nn.Parameter(torch.linspace(-1.0, 1.0, steps=self.num_levels))

    def get_weights(self) -> Tensor:
        """Return global depth weights in shallow-to-deep order."""
        return F.softmax(self.weight_logits, dim=0)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature maps, received {len(features)}.")

        embeddings = [projection(feature) for projection, feature in zip(self.projs, features)]
        weights = self.get_weights().view(1, self.num_levels, 1)
        return (torch.stack(embeddings, dim=1) * weights).sum(dim=1)


class FunnelFusion(nn.Module):
    """The reference `1024 -> 512 -> 256` clinical--dermoscopic fusion MLP."""

    def __init__(self, embedding_dim: int = 512, output_dim: int = 256, dropout: float = 0.408) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, clinical_embedding: Tensor, dermoscopic_embedding: Tensor) -> Tensor:
        if clinical_embedding.shape != dermoscopic_embedding.shape:
            raise ValueError("Clinical and dermoscopic embeddings must have the same shape.")
        return self.network(torch.cat([clinical_embedding, dermoscopic_embedding], dim=1))
