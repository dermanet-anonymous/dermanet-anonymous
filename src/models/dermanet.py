"""Dual-backbone DermaNet reference architecture."""

from __future__ import annotations

import timm
from torch import Tensor, nn

from .fusion import AttentionGatedFusion, FunnelFusion
from .hierarchy import HierarchySpec


class DualHierarchicalModel(nn.Module):
    """Paired EfficientNetV2-XL model with global depth gating and three heads."""

    def __init__(
        self,
        architecture: str = "tf_efficientnetv2_xl.in21k_ft_in1k",
        embedding_dim: int = 512,
        fusion_dim: int = 256,
        dropout: float = 0.408,
        drop_path_rate: float = 0.1,
        pretrained: bool = True,
        feature_levels: tuple[int, int, int] = (2, 3, 4),
        hierarchy: HierarchySpec | None = None,
    ) -> None:
        super().__init__()
        self.hierarchy = hierarchy or HierarchySpec.milk10k()

        self.clin = timm.create_model(
            architecture,
            pretrained=pretrained,
            features_only=True,
            out_indices=feature_levels,
            drop_path_rate=drop_path_rate,
        )
        self.derm = timm.create_model(
            architecture,
            pretrained=pretrained,
            features_only=True,
            out_indices=feature_levels,
            drop_path_rate=drop_path_rate,
        )

        channels = self.clin.feature_info.channels()
        self.clin_gate = AttentionGatedFusion(channels, embedding_dim, dropout)
        self.derm_gate = AttentionGatedFusion(channels, embedding_dim, dropout)
        self.fusion = FunnelFusion(embedding_dim, fusion_dim, dropout)

        sub_head_sizes = self.hierarchy.sub_head_sizes
        self.head_group = nn.Linear(fusion_dim, 2)
        self.head_mel = nn.Linear(fusion_dim, sub_head_sizes[0])
        self.head_other = nn.Linear(fusion_dim, sub_head_sizes[1])

    def forward(self, clinical_images: Tensor, dermoscopic_images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        clinical_embedding = self.clin_gate(self.clin(clinical_images))
        dermoscopic_embedding = self.derm_gate(self.derm(dermoscopic_images))
        fused = self.fusion(clinical_embedding, dermoscopic_embedding)
        return self.head_group(fused), self.head_mel(fused), self.head_other(fused)

    def depth_weights(self) -> dict[str, Tensor]:
        """Return current global depth weights for both modality streams."""
        return {
            "clinical": self.clin_gate.get_weights(),
            "dermoscopic": self.derm_gate.get_weights(),
        }
