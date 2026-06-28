"""Run paired MILK10k inference and export raw stitched probabilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.milk10k import CLASS_NAMES, Milk10kDualDataset, build_test_manifest, get_transforms
from src.evaluation.tta import predict_eight_view_tta
from src.models.dermanet import DualHierarchicalModel
from src.models.hierarchy import HierarchySpec
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/milk10k_reference.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    inference_config = config["inference"]
    training_config = config["training"]

    _, evaluation_transform, _, _ = get_transforms(int(training_config["image_size"]))
    dataframe = build_test_manifest(args.input_root)
    dataset = Milk10kDualDataset(
        dataframe,
        clinical_transform=evaluation_transform,
        dermoscopic_transform=evaluation_transform,
        include_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    hierarchy = HierarchySpec.milk10k()
    model = DualHierarchicalModel(
        architecture=model_config["architecture"],
        embedding_dim=int(model_config["embedding_dim"]),
        fusion_dim=int(model_config["fusion_dim"]),
        dropout=0.0,
        drop_path_rate=float(model_config["drop_path_rate"]),
        pretrained=False,
        feature_levels=tuple(model_config["feature_levels"]),
        hierarchy=hierarchy,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    rows: list[dict[str, float | str]] = []
    for (clinical_images, dermoscopic_images), lesion_ids in tqdm(loader, desc="Inference"):
        clinical_images = clinical_images.to(device, non_blocking=True)
        dermoscopic_images = dermoscopic_images.to(device, non_blocking=True)
        probabilities = predict_eight_view_tta(
            model,
            clinical_images,
            dermoscopic_images,
            hierarchy,
            log_priors=None,
            logit_adjustment_tau=0.0,
        )
        for lesion_id, sample_probabilities in zip(lesion_ids, probabilities.cpu().numpy()):
            row = {"lesion": str(lesion_id)}
            row.update({class_name: float(value) for class_name, value in zip(CLASS_NAMES, sample_probabilities)})
            rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved raw stitched probabilities to {output_path}")


if __name__ == "__main__":
    main()
