"""Seeding and device selection.

One function, called once, at the top of every entry point. Seeding scattered
across a codebase is how "it was reproducible yesterday" happens.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch, and optionally force deterministic kernels.

    `deterministic=True` costs some throughput but means an overfit-8 run that
    diverges is a real bug and not cuDNN algorithm selection.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(prefer: str | None = None) -> torch.device:
    """cuda if available, else cpu. This box is 2x RTX 3090 on a cu118 build."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
