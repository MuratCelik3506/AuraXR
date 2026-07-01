"""A5 -- heuristic quality_label for grasp samples.

quality_raw = w1*contact_ratio - w2*clip(pen/max_pen,0,1) - w3*clip(dist/max_dist,0,1)
quality_label = clip(quality_raw, 0, 1)

This is the offline heuristic label used to supervise `quality_score` (MSE head),
distinct from the Unity-physics `success_label` (BCE head, see docs/C, B7).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.mano_fk import fingertip_positions  # noqa: E402
from model.model_io import (  # noqa: E402
    CONTACT_THRESHOLD_M,
    QUALITY_MAX_DIST_M,
    QUALITY_MAX_PENETRATION_M,
    QUALITY_W_CONTACT,
    QUALITY_W_DIST,
    QUALITY_W_PENETRATION,
)


def compute_quality_label(
    finger_aa45: torch.Tensor,
    obj_pts: torch.Tensor,
    dist: torch.Tensor,
    *,
    fingertips_wrist: torch.Tensor | None = None,
) -> torch.Tensor:
    """finger_aa45: (...,45), obj_pts: (...,N,3), dist: (...,1) or (...,) wrist-object distance.

    fingertips_wrist: optional (...,5,3) pre-computed fingertip positions already in wrist
    frame. When provided (e.g. from stored fk_joints or ground-truth hand_j), the simplified
    mano_fk.py FK is bypassed so the quality signal is geometrically correct.

    Returns quality_label in [0,1], shape (...,).
    """
    if fingertips_wrist is not None:
        tips = fingertips_wrist  # (...,5,3)
    else:
        tips = fingertip_positions(finger_aa45)  # (...,5,3)
    tip_obj_dists = torch.cdist(tips, obj_pts)  # (...,5,N)
    nearest = tip_obj_dists.min(dim=-1).values  # (...,5)
    contact_ratio = (nearest < CONTACT_THRESHOLD_M).float().mean(dim=-1)  # (...,)

    centroid = obj_pts.mean(dim=-2, keepdim=True)
    tip_to_centroid = (tips - centroid).norm(dim=-1)  # (...,5)
    obj_radius = (obj_pts - centroid).norm(dim=-1).mean(dim=-1, keepdim=True)  # (...,1)
    penetration = torch.relu(obj_radius - tip_to_centroid).max(dim=-1).values  # (...,)

    dist = dist.reshape(*dist.shape[: penetration.dim()]) if dist.dim() > penetration.dim() else dist

    pen_term = (penetration / QUALITY_MAX_PENETRATION_M).clamp(0.0, 1.0)
    dist_term = (dist.abs() / QUALITY_MAX_DIST_M).clamp(0.0, 1.0)

    quality_raw = (
        QUALITY_W_CONTACT * contact_ratio - QUALITY_W_PENETRATION * pen_term - QUALITY_W_DIST * dist_term
    )
    return quality_raw.clamp(0.0, 1.0)


def compute_quality_label_np(finger_aa45: np.ndarray, obj_pts: np.ndarray, dist: np.ndarray) -> np.ndarray:
    out = compute_quality_label(
        torch.from_numpy(finger_aa45.astype(np.float32)),
        torch.from_numpy(obj_pts.astype(np.float32)),
        torch.from_numpy(np.asarray(dist, dtype=np.float32)),
    )
    return out.numpy()
