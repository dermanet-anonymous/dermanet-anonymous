# DermaNet

Anonymous implementation accompanying a double-blind submission on paired dermoscopic--clinical skin lesion classification.

## Scope

This repository contains a cleaned, modular version of the core MILK10k implementation used as the reference source for this release. It includes:

- two EfficientNetV2-XL feature-extraction streams for clinical and dermoscopic images;
- feature extraction at backbone levels 2, 3, and 4;
- global learned depth weights for each modality;
- a `1024 -> 512 -> 256` fusion MLP;
- a melanocytic (`MEL`, `NV`) versus non-melanocytic hierarchy with three prediction heads;
- image-level training-time modality dropout;
- soft probability stitching for final class probabilities;
- weighted sampling, logit adjustment, and geometric test-time augmentation.

The release intentionally excludes raw datasets, restricted labels, checkpoints, experiment logs, submissions, and leaderboard-specific thresholding or rescue heuristics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Expected data layout

```text
data/
└── MILK10k_Training_Input/
    ├── MILK10k_Training_GroundTruth.csv
    └── MILK10k_Training_Input/
        ├── lesion_001/
        │   ├── clinical_image.jpg
        │   └── dermoscopic_image.jpg
        └── ...
```

Within each lesion folder, the implementation identifies the clinical image as the image with the smaller parsed ISIC number and the dermoscopic image as the image with the larger parsed ISIC number, following the reference workflow.

## Training

```bash
python scripts/train_milk10k.py --config configs/milk10k_reference.yaml
```

## Inference

```bash
python scripts/infer_milk10k.py \
  --config configs/milk10k_reference.yaml \
  --checkpoint checkpoints/best_milk10k.pth \
  --input-root data/MILK10k_Test_Input/MILK10k_Test_Input \
  --output outputs/milk10k_probabilities.csv
```

## Reproducibility notes

- The default configuration preserves the reference notebook's training settings, including `IMG_SIZE=576`, batch size `8`, gradient accumulation `8`, AdamW, cosine annealing, and modality-dropout probability `0.15`.
- The model's depth aggregation uses **global learned weights**, not sample-specific attention weights.
- Final output probabilities use **soft hierarchical stitching**. The repository exports raw stitched probabilities and does not apply competition-specific post-processing.
- Dataset files, challenge test labels, and trained weights are not redistributed.

## Anonymous review

Author names, affiliations, institutional paths, personal contact information, private experiment identifiers, and restricted benchmark material have been omitted for double-blind review.
