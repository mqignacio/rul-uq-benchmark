"""Utility functions for the QRNN-UQ-RUL project."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        logger.info("Using device: cuda")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("Using device: mps")
        return torch.device("mps")
    logger.info("Using device: cpu")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility (numpy, torch)."""
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create all required output directories if they don't exist."""
    from .config import MODEL_DIR, PREPROCESSED_DIR, RESULTS_DIR

    for d in [MODEL_DIR, PREPROCESSED_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for sub in ["qrnn", "gpr", "xgb", "xgb_quantile"]:
        (MODEL_DIR / sub).mkdir(parents=True, exist_ok=True)
    for sub in ["eval", "pred", "appendix"]:
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)
