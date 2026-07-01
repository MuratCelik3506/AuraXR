"""Shared project paths for AuraXR Python tools."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OAKINK_CANON = PROCESSED_DIR / "oakink_canonical"
HOT3D_CANON = PROCESSED_DIR / "hot3d_canonical"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
