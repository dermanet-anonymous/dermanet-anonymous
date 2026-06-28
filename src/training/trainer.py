"""Training and validation loops derived from the reference notebook."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.models.hierarchy import HierarchySpec, soft_stitch_probabilities
from src.models.modality_dropout import apply_modality_dropout


def build_log_priors(class_counts: Mapping[int, int], hierarchy: HierarchySpec, device: torch.device) -> dict[int, Tensor]:
    """Build within-group log class priors used for validation-time adjustment."""
    priors: dict[int, Tensor] = {}
    for group_id, class_indices in hierarchy.groups.items():
        counts = np.asarray([class_counts.get(class_index, 1) for class_index in class_indices], dtype=np.float64)
        probabilities = counts / counts.sum()
        priors[group_id] = torch.tensor(np.log(probabilities + 1e-9), dtype=torch.float32, device=device)
    return priors


def _autocast_context(device: torch.device):
    return autocast(device_type=device.type, enabled=device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    criterion: nn.Module,
    hierarchy: HierarchySpec,
    device: torch.device,
    accumulation_steps: int,
    modality_dropout_probability: float,
    group_loss_weight: float,
    melanocytic_loss_weight: float,
    non_melanocytic_loss_weight: float,
) -> float:
    """Train one epoch using the reference hierarchy loss weighting."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        (clinical_images, dermoscopic_images), targets = batch
        clinical_images = clinical_images.to(device, non_blocking=True)
        dermoscopic_images = dermoscopic_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        clinical_images, dermoscopic_images = apply_modality_dropout(
            clinical_images,
            dermoscopic_images,
            probability=modality_dropout_probability,
            training=True,
        )

        group_targets = hierarchy.group_target(targets)
        sub_targets = hierarchy.sub_target(targets)

        with _autocast_context(device):
            group_logits, melanocytic_logits, other_logits = model(clinical_images, dermoscopic_images)
            group_loss = criterion(group_logits, group_targets)

            melanocytic_mask = group_targets == 0
            other_mask = group_targets == 1

            melanocytic_loss = (
                criterion(melanocytic_logits[melanocytic_mask], sub_targets[melanocytic_mask])
                if melanocytic_mask.any()
                else group_logits.sum() * 0.0
            )
            other_loss = (
                criterion(other_logits[other_mask], sub_targets[other_mask])
                if other_mask.any()
                else group_logits.sum() * 0.0
            )

            raw_loss = (
                group_loss_weight * group_loss
                + melanocytic_loss_weight * melanocytic_loss
                + non_melanocytic_loss_weight * other_loss
            )
            loss = raw_loss / accumulation_steps

        scaler.scale(loss).backward()
        should_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += raw_loss.detach().item() * targets.shape[0]
        total_samples += targets.shape[0]

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_hierarchical(
    model: nn.Module,
    loader,
    hierarchy: HierarchySpec,
    device: torch.device,
    log_priors: Mapping[int, Tensor] | None = None,
    logit_adjustment_tau: float = 0.12,
    use_horizontal_flip_tta: bool = True,
) -> dict[str, float]:
    """Evaluate macro-F1 and hierarchy-head macro-F1 with soft stitching."""
    model.eval()
    probabilities_all: list[Tensor] = []
    targets_all: list[Tensor] = []
    group_predictions_all: list[Tensor] = []
    group_targets_all: list[Tensor] = []
    mel_predictions_all: list[Tensor] = []
    mel_targets_all: list[Tensor] = []
    other_predictions_all: list[Tensor] = []
    other_targets_all: list[Tensor] = []

    for (clinical_images, dermoscopic_images), targets in tqdm(loader, desc="Validation", leave=False):
        clinical_images = clinical_images.to(device, non_blocking=True)
        dermoscopic_images = dermoscopic_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        group_targets = hierarchy.group_target(targets)
        sub_targets = hierarchy.sub_target(targets)

        with _autocast_context(device):
            group_logits, mel_logits, other_logits = model(clinical_images, dermoscopic_images)
            if use_horizontal_flip_tta:
                group_flip, mel_flip, other_flip = model(
                    torch.flip(clinical_images, dims=[3]),
                    torch.flip(dermoscopic_images, dims=[3]),
                )
                group_logits = (group_logits + group_flip) / 2
                mel_logits = (mel_logits + mel_flip) / 2
                other_logits = (other_logits + other_flip) / 2

            probabilities = soft_stitch_probabilities(
                group_logits,
                mel_logits,
                other_logits,
                hierarchy,
                log_priors=log_priors,
                logit_adjustment_tau=logit_adjustment_tau,
            )

        probabilities_all.append(probabilities.cpu())
        targets_all.append(targets.cpu())
        group_predictions_all.append(F.softmax(group_logits, dim=1).argmax(dim=1).cpu())
        group_targets_all.append(group_targets.cpu())

        melanocytic_mask = (group_targets == 0).cpu()
        if melanocytic_mask.any():
            mel_predictions_all.append(mel_logits.cpu()[melanocytic_mask].argmax(dim=1))
            mel_targets_all.append(sub_targets.cpu()[melanocytic_mask])

        other_mask = (group_targets == 1).cpu()
        if other_mask.any():
            other_predictions_all.append(other_logits.cpu()[other_mask].argmax(dim=1))
            other_targets_all.append(sub_targets.cpu()[other_mask])

    probabilities = torch.cat(probabilities_all).numpy()
    targets = torch.cat(targets_all).numpy()
    predictions = probabilities.argmax(axis=1)

    metrics = {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "group_f1": float(
            f1_score(
                torch.cat(group_targets_all).numpy(),
                torch.cat(group_predictions_all).numpy(),
                average="macro",
                zero_division=0,
            )
        ),
        "mel_head_f1": 0.0,
        "other_head_f1": 0.0,
    }
    if mel_targets_all:
        metrics["mel_head_f1"] = float(
            f1_score(torch.cat(mel_targets_all).numpy(), torch.cat(mel_predictions_all).numpy(), average="macro", zero_division=0)
        )
    if other_targets_all:
        metrics["other_head_f1"] = float(
            f1_score(torch.cat(other_targets_all).numpy(), torch.cat(other_predictions_all).numpy(), average="macro", zero_division=0)
        )
    return metrics
