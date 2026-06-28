# DermaNet

Anonymous implementation for paired dermoscopic--clinical skin lesion classification.

## Overview

DermaNet combines:

- Dual EfficientNetV2 encoders for dermoscopic and clinical images
- Multi-level features from backbone levels 2, 3, and 4
- Learned feature fusion between image modalities
- Training-time modality dropout
- Melanocytic vs. non-melanocytic hierarchical prediction
- Soft probability stitching / hard routing for final class predictions

## Main Files

    configs/                 Training settings
    src/data/                Dataset loading and image transforms
    src/models/              DermaNet model, fusion, hierarchy, modality dropout
    src/training/            Training loop
    src/evaluation/          Metrics and test-time augmentation
    scripts/train_milk10k.py Train the model
    scripts/infer_milk10k.py Run inference from a saved checkpoint

## Installation

    pip install -r requirements.txt

## Training

Edit dataset paths and settings in:

    configs/milk10k_reference.yaml

Then run:

    python scripts/train_milk10k.py --config configs/milk10k_reference.yaml

The best checkpoint is saved according to the validation macro-F1.

## Inference

    python scripts/infer_milk10k.py --config configs/milk10k_reference.yaml --checkpoint checkpoints/best_milk10k.pth --input-root path/to/test/images --output outputs/predictions.csv

## Data

Datasets, private labels, challenge files, checkpoints, and submission files are not included. Users must obtain the datasets from their original providers.

## Anonymous Review

Author names, affiliations, private paths, experiment logs, and identifying metadata have been removed for double-blind review.
