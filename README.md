# DermaNet

Anonymous implementation accompanying the double-blind submission:

**DermaNet: Multi-Level Feature Fusion, Modality Dropout, and Clinically Guided Prediction for Paired Dermoscopic--Clinical Image Classification**

## Overview

DermaNet is a framework for skin lesion classification from paired dermoscopic and clinical images.

The implementation includes:

- Dual EfficientNetV2 backbones for dermoscopic and clinical inputs
- Multi-level feature extraction from intermediate backbone stages
- Learned depth-wise feature aggregation
- Feature-level fusion of dermoscopic and clinical embeddings
- Training-time modality dropout
- Clinically guided melanocytic/non-melanocytic prediction
- Hard hierarchical routing and soft probability stitching
- Evaluation utilities for MILK10k and ISIC2019

## Repository Status

This repository is provided for anonymous review.

Dataset files, challenge test labels, trained checkpoints, private experiment outputs, and restricted benchmark materials are not distributed. Users must obtain datasets through their original providers and follow the applicable access terms.

## Installation

pip install -r requirements.txt

## Reproducibility

Detailed setup, dataset layout, training, and evaluation instructions will be provided in the `docs/` directory.

## Data Availability

This repository does not redistribute:

* MILK10k images or challenge files
* ISIC2019 images or labels
* Private train/validation/test manifests
* Trained checkpoints
* Challenge submissions

## Anonymous Review

Author names, affiliations, institutional paths, personal contact details, and identifying experiment metadata have been omitted for double-blind review.

