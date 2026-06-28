"""MILK10k paired clinical--dermoscopic data utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TVF

CLASS_NAMES = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

MEAN_FILL = (124, 116, 104)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_isic_number(path: Path) -> Optional[int]:
    """Extract an integer image identifier from an ISIC-style filename."""
    stem = path.stem
    if "ISIC_" in stem:
        tail = stem.split("ISIC_", 1)[1]
        digits = "".join(character for character in tail if character.isdigit())
        if digits:
            return int(digits)

    digits = "".join(character for character in stem if character.isdigit())
    return int(digits) if digits else None


def _find_paired_images(lesion_dir: Path) -> tuple[Path, Path] | None:
    images = sorted(
        path
        for path in lesion_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if len(images) < 2:
        return None

    parsed = [(path, parse_isic_number(path)) for path in images]
    valid = [(path, number) for path, number in parsed if number is not None]
    if len(valid) < 2:
        return None

    valid.sort(key=lambda item: item[1])
    clinical_path = valid[0][0]
    dermoscopic_path = valid[-1][0]
    return clinical_path, dermoscopic_path


def _scan_lesion_directories(image_dir: Path) -> pd.DataFrame:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    rows: list[dict[str, str]] = []
    for lesion_dir in sorted(path for path in image_dir.iterdir() if path.is_dir()):
        pair = _find_paired_images(lesion_dir)
        if pair is None:
            continue

        clinical_path, dermoscopic_path = pair
        rows.append(
            {
                "LesionID": lesion_dir.name,
                "Clinical": str(clinical_path.resolve()),
                "Dermoscopic": str(dermoscopic_path.resolve()),
            }
        )

    return pd.DataFrame(rows)


def build_train_manifest(image_dir: str | Path, ground_truth_csv: str | Path) -> pd.DataFrame:
    """Build a paired-image manifest and attach integer class labels."""
    image_dir = Path(image_dir)
    ground_truth_csv = Path(ground_truth_csv)

    manifest = _scan_lesion_directories(image_dir)
    if manifest.empty:
        raise RuntimeError(f"No valid paired lesions were found in {image_dir}.")

    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground-truth CSV does not exist: {ground_truth_csv}")

    labels = pd.read_csv(ground_truth_csv)
    if labels.empty:
        raise RuntimeError("Ground-truth CSV is empty.")

    lesion_column = labels.columns[0]
    labels = labels.rename(columns={lesion_column: "LesionID"})
    labels["LesionID"] = labels["LesionID"].astype(str)

    missing_columns = [name for name in CLASS_NAMES if name not in labels.columns]
    if missing_columns:
        raise ValueError(
            "Ground-truth CSV is missing expected class columns: "
            + ", ".join(missing_columns)
        )

    values = labels[CLASS_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    valid_rows = values.sum(axis=1) > 0
    class_ids = np.full(len(labels), np.nan)
    class_ids[valid_rows] = values[valid_rows].argmax(axis=1)
    labels["ClassId"] = class_ids

    manifest = manifest.merge(labels[["LesionID", "ClassId"]], on="LesionID", how="left")
    manifest["ClassId"] = manifest["ClassId"].astype("Int64")
    return manifest


def build_test_manifest(image_dir: str | Path) -> pd.DataFrame:
    """Build an unlabeled paired-image test manifest."""
    manifest = _scan_lesion_directories(Path(image_dir))
    if manifest.empty:
        raise RuntimeError(f"No valid paired lesions were found in {image_dir}.")
    return manifest


class ResizePad:
    """Resize the longer image side then pad to a square target size."""

    def __init__(self, target_size: int, fill: tuple[int, int, int] = MEAN_FILL) -> None:
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = TVF.resize(
            image,
            (new_height, new_width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        pad_left = (self.target_size - new_width) // 2
        pad_top = (self.target_size - new_height) // 2
        pad_right = self.target_size - new_width - pad_left
        pad_bottom = self.target_size - new_height - pad_top
        return TVF.pad(
            image,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
            padding_mode="constant",
        )


class SafeRotate:
    """Rotate after reflection padding, then return to the initial dimensions."""

    def __init__(self, degrees: float) -> None:
        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        padding = max(width, height) // 2
        padded = TVF.pad(image, padding=padding, padding_mode="reflect")
        angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])
        rotated = TVF.rotate(
            padded,
            angle,
            interpolation=InterpolationMode.BICUBIC,
        )
        return TVF.center_crop(rotated, (height, width))


def get_transforms(image_size: int):
    """Return dermoscopic train/eval and clinical train/eval transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    base = [transforms.Lambda(lambda image: image.convert("RGB"))]

    dermoscopic_train = transforms.Compose(
        base
        + [
            ResizePad(image_size),
            SafeRotate(180),
            transforms.RandomApply(
                [transforms.RandomResizedCrop(image_size, scale=(0.95, 1.0), ratio=(1.0, 1.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    clinical_train = transforms.Compose(
        base
        + [
            ResizePad(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            normalize,
        ]
    )

    evaluation = transforms.Compose(base + [ResizePad(image_size), transforms.ToTensor(), normalize])
    return dermoscopic_train, evaluation, clinical_train, evaluation


class Milk10kDualDataset(Dataset):
    """Dataset yielding paired clinical and dermoscopic tensors."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        clinical_transform=None,
        dermoscopic_transform=None,
        include_labels: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.clinical_paths = self.dataframe["Clinical"].tolist()
        self.dermoscopic_paths = self.dataframe["Dermoscopic"].tolist()
        self.lesion_ids = self.dataframe["LesionID"].astype(str).tolist()
        self.labels = (
            self.dataframe["ClassId"].fillna(-1).astype(int).tolist()
            if "ClassId" in self.dataframe.columns
            else [-1] * len(self.dataframe)
        )
        self.clinical_transform = clinical_transform
        self.dermoscopic_transform = dermoscopic_transform
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        clinical = Image.open(self.clinical_paths[index]).convert("RGB")
        dermoscopic = Image.open(self.dermoscopic_paths[index]).convert("RGB")

        if self.clinical_transform is not None:
            clinical = self.clinical_transform(clinical)
        if self.dermoscopic_transform is not None:
            dermoscopic = self.dermoscopic_transform(dermoscopic)

        if self.include_labels:
            return (clinical, dermoscopic), self.labels[index]
        return (clinical, dermoscopic), self.lesion_ids[index]


def collate_dual(batch):
    """Collate paired images and integer labels."""
    pairs, targets = zip(*batch)
    clinical_images, dermoscopic_images = zip(*pairs)
    return (
        torch.stack(clinical_images),
        torch.stack(dermoscopic_images),
    ), torch.tensor(targets, dtype=torch.long)


def compute_rfs_weights(dataframe: pd.DataFrame, tau: float = 0.003, cap: float = 3.0) -> Tensor:
    """Compute the square-root frequency sampler weights from the reference notebook."""
    labels = dataframe["ClassId"].fillna(-1).astype(int).to_numpy()
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        raise ValueError("No labeled samples are available for weighted sampling.")

    counts = np.bincount(valid_labels, minlength=len(CLASS_NAMES))
    frequencies = counts / len(valid_labels)

    weights = []
    for label in labels:
        if label < 0:
            weights.append(0.0)
        else:
            weights.append(min(np.sqrt(tau / max(frequencies[label], 1e-9)), cap))

    return torch.tensor(weights, dtype=torch.double)
