"""Compatibility entry point for the main AuraXR grasp model."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import GraspModel, TemporalGeometryConditionedGraspModel, grasp_loss  # noqa: E402

__all__ = ["GraspModel", "TemporalGeometryConditionedGraspModel", "grasp_loss"]
