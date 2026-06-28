"""Main DermaNet architecture for paired dermoscopic--clinical classification."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import timm
import torch
from torch import Tensor, nn

from .fusion import DepthGatedAggregator, PairedFeatureFusion
from .hierarchy import MelanocyticHierarchy
from .modality_dropout import ModalityDropout


class MultiLevelEfficientNetEncoder(nn.Module):
    """
    EfficientNetV2 encoder that extracts three intermediate feature levels,
    projects each into a shared embedding space, and aggregates them using
    learned depth gating.
    """

    def __init__(
        self,
        backbone_name: str,
        feature_levels: Sequence[int] = (2, 3, 4),
        embedding_dim: int = 512,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(feature_levels) != 3:
            raise ValueError(
                "DermaNet expects exactly three feature levels, "
                "typically (2, 3, 4)."
            )

        self.embedding_dim = embedding_dim
        self.num_levels = len(feature_levels)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(feature_levels),
            drop_path_rate=drop_path_rate,
        )

        feature_channels = self.backbone.feature_info.channels()

        if len(feature_channels) != self.num_levels:
            raise RuntimeError(
                "The backbone did not return the requested number of "
                "intermediate feature maps."
            )

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(channels, embedding_dim),
                    nn.BatchNorm1d(embedding_dim),
                    nn.SiLU(inplace=True),
                    nn.Dropout(projection_dropout),
                )
                for channels in feature_channels
            ]
        )

        self.depth_aggregator = DepthGatedAggregator(
            embedding_dim=embedding_dim,
            num_levels=self.num_levels,
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            images: Input image tensor with shape [batch_size, 3, height, width].

        Returns:
            embedding:
                Aggregated embedding with shape [batch_size, embedding_dim].

            depth_weights:
                Per-sample depth weights with shape [batch_size, 3].
        """
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [batch_size, channels, height, width]."
            )

        feature_maps = self.backbone(images)

        if len(feature_maps) != self.num_levels:
            raise RuntimeError(
                f"Expected {self.num_levels} feature maps, "
                f"received {len(feature_maps)}."
            )

        level_embeddings = [
            projection(feature_map)
            for projection, feature_map in zip(
                self.projections,
                feature_maps,
            )
        ]

        return self.depth_aggregator(level_embeddings)


class DermaNet(nn.Module):
    """
    DermaNet for paired dermoscopic--clinical image classification.

    The model supports:
    - paired dermoscopic and clinical inputs;
    - dermoscopy-only inference;
    - clinical-only inference;
    - training-time modality dropout for paired samples;
    - hard hierarchical routing;
    - soft probability stitching.
    """

    def __init__(
        self,
        num_classes: int,
        mc_class_indices: Iterable[int],
        backbone_name: str = "tf_efficientnetv2_xl.in21k_ft_in1k",
        feature_levels: Sequence[int] = (2, 3, 4),
        embedding_dim: int = 512,
        fused_dim: int = 256,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        projection_dropout: float = 0.0,
        fusion_dropout: float = 0.4,
        modality_dropout_rate: float = 0.1,
        group_loss_weight: float = 1.0,
        subgroup_loss_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_levels = len(feature_levels)

        self.dermoscopic_encoder = MultiLevelEfficientNetEncoder(
            backbone_name=backbone_name,
            feature_levels=feature_levels,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            projection_dropout=projection_dropout,
        )

        self.clinical_encoder = MultiLevelEfficientNetEncoder(
            backbone_name=backbone_name,
            feature_levels=feature_levels,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            projection_dropout=projection_dropout,
        )

        self.modality_dropout = ModalityDropout(
            rho=modality_dropout_rate,
        )

        self.feature_fusion = PairedFeatureFusion(
            embedding_dim=embedding_dim,
            fused_dim=fused_dim,
            dropout=fusion_dropout,
        )

        self.hierarchy = MelanocyticHierarchy(
            feature_dim=fused_dim,
            num_classes=num_classes,
            mc_class_indices=mc_class_indices,
            group_loss_weight=group_loss_weight,
            subgroup_loss_weight=subgroup_loss_weight,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        dermoscopic_images: Optional[Tensor] = None,
        clinical_images: Optional[Tensor] = None,
        availability: Optional[Tensor] = None,
    ) -> dict[str, Any]:
        """
        Args:
            dermoscopic_images:
                Optional dermoscopic images with shape [batch_size, 3, H, W].

            clinical_images:
                Optional clinical images with shape [batch_size, 3, H, W].

            availability:
                Optional binary tensor with shape [batch_size, 2], in
                dermoscopic--clinical order. Use [1, 1] for paired images,
                [1, 0] for dermoscopy-only, and [0, 1] for clinical-only.

        Returns:
            Dictionary containing hierarchy logits, fused features, depth weights,
            and the effective modality mask used for the current forward pass.
        """
        batch_size, device = self._get_batch_size_and_device(
            dermoscopic_images=dermoscopic_images,
            clinical_images=clinical_images,
        )

        availability = self._prepare_availability(
            availability=availability,
            batch_size=batch_size,
            device=device,
            dermoscopic_present=dermoscopic_images is not None,
            clinical_present=clinical_images is not None,
        )

        dermoscopic_embedding: Optional[Tensor] = None
        clinical_embedding: Optional[Tensor] = None
        dermoscopic_depth_weights: Optional[Tensor] = None
        clinical_depth_weights: Optional[Tensor] = None

        if dermoscopic_images is not None:
            dermoscopic_embedding, dermoscopic_depth_weights = (
                self.dermoscopic_encoder(dermoscopic_images)
            )

        if clinical_images is not None:
            clinical_embedding, clinical_depth_weights = (
                self.clinical_encoder(clinical_images)
            )

        reference_embedding = (
            dermoscopic_embedding
            if dermoscopic_embedding is not None
            else clinical_embedding
        )

        if reference_embedding is None:
            raise RuntimeError("At least one modality must be provided.")

        if dermoscopic_embedding is None:
            dermoscopic_embedding = torch.zeros_like(reference_embedding)
            dermoscopic_depth_weights = torch.zeros(
                batch_size,
                self.num_levels,
                device=device,
                dtype=reference_embedding.dtype,
            )

        if clinical_embedding is None:
            clinical_embedding = torch.zeros_like(reference_embedding)
            clinical_depth_weights = torch.zeros(
                batch_size,
                self.num_levels,
                device=device,
                dtype=reference_embedding.dtype,
            )

        (
            masked_dermoscopic_embedding,
            masked_clinical_embedding,
            effective_mask,
        ) = self.modality_dropout(
            dermoscopic_embedding=dermoscopic_embedding,
            clinical_embedding=clinical_embedding,
            availability=availability,
        )

        fused_features = self.feature_fusion(
            dermoscopic_embedding=masked_dermoscopic_embedding,
            clinical_embedding=masked_clinical_embedding,
        )

        hierarchy_outputs = self.hierarchy(fused_features)

        return {
            "hierarchy": hierarchy_outputs,
            "fused_features": fused_features,
            "dermoscopic_embedding": masked_dermoscopic_embedding,
            "clinical_embedding": masked_clinical_embedding,
            "dermoscopic_depth_weights": dermoscopic_depth_weights,
            "clinical_depth_weights": clinical_depth_weights,
            "effective_modality_mask": effective_mask,
        }

    def compute_loss(
        self,
        model_outputs: dict[str, Any],
        targets: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the joint group and subgroup classification loss."""
        return self.hierarchy.compute_loss(
            outputs=model_outputs["hierarchy"],
            targets=targets,
        )

    @torch.no_grad()
    def predict(
        self,
        model_outputs: dict[str, Any],
        mode: str = "hard",
        threshold: float = 0.5,
    ) -> Tensor:
        """
        Predict global class indices.

        Args:
            model_outputs: Output dictionary returned by forward().
            mode: "hard" for threshold-based routing or "soft" for stitching.
            threshold: Melanocytic routing threshold for hard inference.
        """
        return self.hierarchy.predict(
            outputs=model_outputs["hierarchy"],
            mode=mode,
            threshold=threshold,
        )

    @staticmethod
    def _get_batch_size_and_device(
        dermoscopic_images: Optional[Tensor],
        clinical_images: Optional[Tensor],
    ) -> tuple[int, torch.device]:
        available_images = [
            image
            for image in [dermoscopic_images, clinical_images]
            if image is not None
        ]

        if not available_images:
            raise ValueError(
                "At least one of dermoscopic_images or clinical_images "
                "must be provided."
            )

        reference = available_images[0]

        if reference.ndim != 4:
            raise ValueError(
                "Each image tensor must have shape "
                "[batch_size, channels, height, width]."
            )

        for image in available_images[1:]:
            if image.ndim != 4:
                raise ValueError(
                    "Each image tensor must have shape "
                    "[batch_size, channels, height, width]."
                )

            if image.shape[0] != reference.shape[0]:
                raise ValueError(
                    "Dermoscopic and clinical image tensors must share "
                    "the same batch size."
                )

            if image.device != reference.device:
                raise ValueError(
                    "Dermoscopic and clinical image tensors must be on "
                    "the same device."
                )

        return reference.shape[0], reference.device

    @staticmethod
    def _prepare_availability(
        availability: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dermoscopic_present: bool,
        clinical_present: bool,
    ) -> Tensor:
        if availability is None:
            availability = torch.tensor(
                [
                    [
                        float(dermoscopic_present),
                        float(clinical_present),
                    ]
                ],
                device=device,
            ).expand(batch_size, -1)
        else:
            if availability.ndim != 2 or availability.shape != (batch_size, 2):
                raise ValueError(
                    "availability must have shape [batch_size, 2] in "
                    "dermoscopic--clinical order."
                )

            availability = availability.to(
                device=device,
                dtype=torch.float32,
            )

        if not torch.all((availability == 0.0) | (availability == 1.0)):
            raise ValueError(
                "availability must contain only binary values: 0 or 1."
            )

        if torch.any(availability.sum(dim=1) == 0):
            raise ValueError(
                "Each sample must have at least one available modality."
            )

        if not dermoscopic_present and torch.any(availability[:, 0] == 1):
            raise ValueError(
                "availability marks dermoscopy as present, but "
                "dermoscopic_images was not supplied."
            )

        if not clinical_present and torch.any(availability[:, 1] == 1):
            raise ValueError(
                "availability marks clinical input as present, but "
                "clinical_images was not supplied."
            )

        return availability
