"""Train the reference DermaNet MILK10k model."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.milk10k import (
    Milk10kDualDataset,
    build_train_manifest,
    collate_dual,
    compute_rfs_weights,
    get_transforms,
)
from src.models.dermanet import DualHierarchicalModel
from src.models.hierarchy import HierarchySpec
from src.training.trainer import build_log_priors, evaluate_hierarchical, train_one_epoch
from src.utils.config import load_yaml
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/milk10k_reference.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    dataframe = build_train_manifest(data_config["train_image_dir"], data_config["ground_truth_csv"])
    labeled = dataframe[dataframe["ClassId"].notna()].copy()
    labeled["ClassId"] = labeled["ClassId"].astype(int)

    counts = labeled["ClassId"].value_counts()
    safe = labeled[labeled["ClassId"].isin(counts[counts > 1].index)].copy()
    if len(safe) < 2:
        raise RuntimeError("At least two labeled samples are required.")

    train_indices, validation_indices = train_test_split(
        safe.index,
        test_size=float(data_config["validation_fraction"]),
        stratify=safe["ClassId"],
        random_state=int(config["seed"]),
    )
    train_dataframe = safe.loc[train_indices].reset_index(drop=True)
    validation_dataframe = safe.loc[validation_indices].reset_index(drop=True)

    derm_train, derm_eval, clinical_train, clinical_eval = get_transforms(int(training_config["image_size"]))
    train_dataset = Milk10kDualDataset(train_dataframe, clinical_train, derm_train)
    validation_dataset = Milk10kDualDataset(validation_dataframe, clinical_eval, derm_eval)

    sample_weights = compute_rfs_weights(
        train_dataframe,
        tau=float(training_config["weighted_sampler_tau"]),
        cap=float(training_config["weighted_sampler_cap"]),
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_kwargs = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_dual,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    hierarchy = HierarchySpec.milk10k()
    model = DualHierarchicalModel(
        architecture=model_config["architecture"],
        embedding_dim=int(model_config["embedding_dim"]),
        fusion_dim=int(model_config["fusion_dim"]),
        dropout=float(model_config["dropout"]),
        drop_path_rate=float(model_config["drop_path_rate"]),
        pretrained=bool(model_config["pretrained"]),
        feature_levels=tuple(model_config["feature_levels"]),
        hierarchy=hierarchy,
    ).to(device)

    gate_parameters = [parameter for name, parameter in model.named_parameters() if "weight_logits" in name]
    main_parameters = [parameter for name, parameter in model.named_parameters() if "weight_logits" not in name]
    learning_rate = float(training_config["learning_rate"])
    optimizer = AdamW(
        [
            {"params": main_parameters, "lr": learning_rate, "weight_decay": float(training_config["weight_decay"])},
            {"params": gate_parameters, "lr": learning_rate * float(training_config["gate_learning_rate_multiplier"]), "weight_decay": 0.0},
        ]
    )

    updates_per_epoch = math.ceil(len(train_loader) / int(training_config["accumulation_steps"]))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["num_epochs"]) * updates_per_epoch,
        eta_min=learning_rate * float(training_config["min_learning_rate_ratio"]),
    )
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")
    criterion = CrossEntropyLoss(label_smoothing=float(training_config["label_smoothing"]))
    class_counts = Counter(train_dataframe["ClassId"].tolist())
    log_priors = build_log_priors(class_counts, hierarchy, device)

    checkpoint_path = Path(output_config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = float("-inf")

    for epoch in range(1, int(training_config["num_epochs"]) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            criterion=criterion,
            hierarchy=hierarchy,
            device=device,
            accumulation_steps=int(training_config["accumulation_steps"]),
            modality_dropout_probability=float(training_config["modality_dropout_probability"]),
            group_loss_weight=float(training_config["group_loss_weight"]),
            melanocytic_loss_weight=float(training_config["melanocytic_loss_weight"]),
            non_melanocytic_loss_weight=float(training_config["non_melanocytic_loss_weight"]),
        )
        metrics = evaluate_hierarchical(
            model=model,
            loader=validation_loader,
            hierarchy=hierarchy,
            device=device,
            log_priors=log_priors,
            logit_adjustment_tau=float(config["inference"]["logit_adjustment_tau"]),
            use_horizontal_flip_tta=True,
        )

        weights = model.depth_weights()
        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | macro_f1={metrics['macro_f1']:.4f} "
            f"| group_f1={metrics['group_f1']:.4f} | mel_f1={metrics['mel_head_f1']:.4f} "
            f"| other_f1={metrics['other_head_f1']:.4f}"
        )
        print(
            "Depth weights | clinical="
            f"{weights['clinical'].detach().cpu().numpy().round(4).tolist()} | dermoscopic="
            f"{weights['dermoscopic'].detach().cpu().numpy().round(4).tolist()}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print(f"Best validation macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
