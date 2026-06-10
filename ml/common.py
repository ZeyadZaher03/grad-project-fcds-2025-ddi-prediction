# ml/common.py
"""Shared constants and helpers for the DDI severity pipeline."""
from __future__ import annotations
import re
import random
import numpy as np

# Fixed class order used EVERYWHERE (training, metrics, serving metadata).
CLASSES = ["None", "Minor", "Moderate", "Major"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEVERITY_CLASSES = ["Minor", "Moderate", "Major"]  # excludes "None"

# DDInter "Level" -> our class. "Unknown" maps to None (dropped from positives).
LEVEL_MAP = {"Minor": "Minor", "Moderate": "Moderate", "Major": "Major"}


def normalize_name(name: str) -> str:
    """Lowercase and strip everything except a-z0-9 for name-based joins."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
