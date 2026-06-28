"""Random-seed utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set the reference workflow's random seeds and CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
