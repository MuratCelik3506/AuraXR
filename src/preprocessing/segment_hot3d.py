"""A4 -- HOT3D temporal grasp segmentation into approach/grasp/transition phases.

Phase rule (per frame), using fields already present in the canonical contract
(rel_pos/dist, finger_aa45, contact_flag, rel_vel):
  - GRASP: in contact AND average MCP flexion >= GRASP_FINGER_MCP_DEG
           AND |rel_vel| <= GRASP_VELOCITY_THRESHOLD_M_S (stable hold)
  - APPROACH: not GRASP AND dist <= APPROACH_DIST_THRESHOLD_M
  - else: APPROACH is also used as the default "far from object" phase, since the
    object split (A7) only needs grasp-relevant windows; everything else is
    flagged as approach so downstream window sampling can filter on phase.
  - TRANSITION: any frame within +/-TRANSITION_WINDOW_FRAMES of an
    approach<->grasp phase boundary, overriding the rule above.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.model_io import (  # noqa: E402
    APPROACH_DIST_THRESHOLD_M,
    GRASP_FINGER_MCP_DEG,
    GRASP_VELOCITY_THRESHOLD_M_S,
    MANO_FINGER_RANGES,
    TRANSITION_WINDOW_FRAMES,
)

PHASE_APPROACH = 0
PHASE_GRASP = 1
PHASE_TRANSITION = 2

_MCP_SLICES = [MANO_FINGER_RANGES[name] for name in (
    "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp", "thumb_cmc",
)]


def _mean_mcp_flexion_deg(finger_aa45: np.ndarray) -> np.ndarray:
    """(...,45) -> (...,) mean MCP joint rotation magnitude across the 5 fingers, degrees."""
    mags = []
    for lo, hi in _MCP_SLICES:
        joint = finger_aa45[..., lo:hi]
        mags.append(np.linalg.norm(joint, axis=-1))
    return np.mean(np.stack(mags, axis=-1), axis=-1) * (180.0 / np.pi)


def segment_sequence(
    dist: np.ndarray,
    finger_aa45: np.ndarray,
    contact_flag: np.ndarray,
    rel_vel: np.ndarray,
) -> np.ndarray:
    """All inputs (T,*) aligned per-frame. Returns segment_id (T,) int64."""
    n = len(dist)
    dist = np.asarray(dist).reshape(n)
    contact_flag = np.asarray(contact_flag).reshape(n)
    rel_vel = np.asarray(rel_vel).reshape(n, -1)
    speed = np.linalg.norm(rel_vel, axis=-1)
    mcp_deg = _mean_mcp_flexion_deg(finger_aa45)

    in_contact = contact_flag > 0.5
    is_grasp = in_contact & (mcp_deg >= GRASP_FINGER_MCP_DEG) & (speed <= GRASP_VELOCITY_THRESHOLD_M_S)
    is_approach = (~is_grasp) & (dist <= APPROACH_DIST_THRESHOLD_M)

    phase = np.full(n, PHASE_APPROACH, dtype=np.int64)
    phase[is_grasp] = PHASE_GRASP
    phase[is_approach & ~is_grasp] = PHASE_APPROACH

    boundaries = np.flatnonzero(np.diff(phase) != 0)
    segment_id = phase.copy()
    for b in boundaries:
        lo = max(0, b - TRANSITION_WINDOW_FRAMES + 1)
        hi = min(n, b + TRANSITION_WINDOW_FRAMES + 1)
        segment_id[lo:hi] = PHASE_TRANSITION
    return segment_id


def segment_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return segment_sequence(data["dist"], data["finger_aa45"], data["contact_flag"], data["rel_vel"])


def main() -> None:
    import argparse

    from utils.paths import HOT3D_CANON

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HOT3D_CANON)
    parser.add_argument("--write", action="store_true", help="overwrite segment_id field in each seq_*.npz")
    args = parser.parse_args()

    files = sorted(args.root.glob("seq_*.npz"))
    for path in files:
        segment_id = segment_npz(path)
        counts = {int(p): int((segment_id == p).sum()) for p in np.unique(segment_id)}
        print(f"{path.name}: {counts}")
        if args.write:
            data = dict(np.load(path, allow_pickle=True))
            data["segment_id"] = segment_id
            np.savez(path, **data)


if __name__ == "__main__":
    main()
