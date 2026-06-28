"""Reproducibility utilities for DermaNet experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Global random seed.
        deterministic: When True, prefer deterministic CUDA operations.
            This can reduce training speed.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
