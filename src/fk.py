"""fk.py — Forward kinematics for UmeTrack 22-joint hand skeleton.

Used by evaluate.py / evaluate_onnx.py to compute MPJPE in mm and PCK curves,
in addition to per-joint angle MAE.

Topology (extracted from any HOT3D umetrack_hand_user_profile.json):
    5 finger chains + 1 wrist landmark chain.
    Each joint is single-DOF, rotates around joint_rotation_axes[j] by joint_angles[j].
    joint_rest_positions[j] is in the wrist (world) frame at rest pose, in mm.
    Parent-local offset = rest_positions[j] - rest_positions[parent[j]].
    Joints 20–21 are placeholders (always 0 in HOT3D).

Canonical skeleton: we load ONE representative HOT3D user profile and use it for
all evaluation. Cross-user shape variation contributes uniformly to both pred and
gt, so MPJPE remains comparable. This is a limitation but documented in §5.

API:
    skel = load_canonical_skeleton(data_root)
    pos  = forward_kinematics(joint_angles_22, skel)   # (22, 3) mm, wrist-local
    err  = mpjpe(pred_22, tgt_22, skel)                # scalar mm
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTIVE_JOINTS = list(range(20))   # joints 20–21 are placeholders (always 0)


@dataclass
class Skeleton:
    """UmeTrack hand skeleton parameters (single user / canonical)."""
    rest_positions: np.ndarray   # (22, 3) mm — local offset to parent
    rotation_axes:  np.ndarray   # (22, 3) unit vectors
    parent:         np.ndarray   # (22,) int, 255 = root
    chains:         list[list[int]]  # parent-first joint order per chain

    @property
    def n_joints(self) -> int:
        return self.rest_positions.shape[0]


# ── Skeleton loader ──────────────────────────────────────────────────────────

def _build_chains(parent: np.ndarray, first_child: np.ndarray) -> list[list[int]]:
    """Return parent-first ordered joint indices per kinematic chain."""
    chains = []
    for root in range(len(parent)):
        if parent[root] != 255:
            continue
        chain = [root]
        cur = int(first_child[root])
        while cur != 255 and cur < len(parent):
            chain.append(cur)
            cur = int(first_child[cur])
        chains.append(chain)
    return chains


def load_skeleton_from_zip(hand_zip: Path) -> Skeleton:
    """Load one user's hand model from a Hot3DQuest hand_data.zip."""
    with zipfile.ZipFile(hand_zip, "r") as zf:
        with zf.open("umetrack_hand_user_profile.json") as f:
            hm = json.load(f)["hand_model"]
    return Skeleton(
        rest_positions=np.array(hm["joint_rest_positions"],   dtype=np.float64),
        rotation_axes =np.array(hm["joint_rotation_axes"],    dtype=np.float64),
        parent        =np.array(hm["joint_parent"],           dtype=np.int32),
        chains        =_build_chains(
            np.array(hm["joint_parent"],      dtype=np.int32),
            np.array(hm["joint_first_child"], dtype=np.int32),
        ),
    )


def load_canonical_skeleton(workspace_root: Path) -> Skeleton:
    """Find the first available HOT3D Quest3 hand_data.zip and use its user profile.

    Cross-user variation in finger length is small (~5–10%) and contributes
    uniformly to both pred and gt positions, so MPJPE remains comparable.
    """
    for split in ("train", "test"):
        split_dir = workspace_root / "data" / "quest3" / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            for prefix in ("Hot3DQuest_v4.0.0_", "Hot3DAria_v4.0.0_"):
                hz = seq_dir / f"{prefix}{seq_dir.name}_hand_data.zip"
                if hz.exists():
                    return load_skeleton_from_zip(hz)
    raise FileNotFoundError(
        f"No HOT3D hand_data.zip found under {workspace_root}/data/raw/hot3d/quest3/."
    )


# ── Forward kinematics ──────────────────────────────────────────────────────

def _axis_angle_to_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula. axis is unit (22,3 already normalized in profile)."""
    c, s = np.cos(theta), np.sin(theta)
    ux, uy, uz = axis
    one_c = 1.0 - c
    return np.array([
        [c + ux*ux*one_c,    ux*uy*one_c - uz*s, ux*uz*one_c + uy*s],
        [uy*ux*one_c + uz*s, c + uy*uy*one_c,    uy*uz*one_c - ux*s],
        [uz*ux*one_c - uy*s, uz*uy*one_c + ux*s, c + uz*uz*one_c   ],
    ], dtype=np.float64)


def forward_kinematics(joint_angles: np.ndarray, skel: Skeleton) -> np.ndarray:
    """Compute world (= wrist-local) positions of all 22 joints in mm.

    UmeTrack convention: rest_positions are wrist-frame absolute positions at
    rest pose. Parent-local offset = rest[j] - rest[parent[j]] (or rest[j]
    itself for chain roots). Rotation at joint j is applied around its rest
    position in parent's *current* frame and propagates to all descendants.

    Args:
        joint_angles: (22,) radians. Joints 20–21 ignored.
        skel: canonical Skeleton.

    Returns:
        positions: (22, 3) mm, in the wrist frame.
    """
    n = skel.n_joints
    world_T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    out_pos = np.zeros((n, 3), dtype=np.float64)

    for chain in skel.chains:
        for j in chain:
            p = int(skel.parent[j])
            if p != 255 and p < n:
                parent_T = world_T[p]
                local_offset = skel.rest_positions[j] - skel.rest_positions[p]
            else:
                parent_T = np.eye(4)
                local_offset = skel.rest_positions[j]

            local_T = np.eye(4, dtype=np.float64)
            local_T[:3, 3]  = local_offset
            local_T[:3, :3] = _axis_angle_to_rot(
                skel.rotation_axes[j], float(joint_angles[j])
            )
            world_T[j] = parent_T @ local_T
            out_pos[j] = world_T[j][:3, 3]
    return out_pos


def forward_kinematics_batch(joint_angles_batch: np.ndarray, skel: Skeleton) -> np.ndarray:
    """Vectorized FK over a batch. (N, 22) angles → (N, 22, 3) positions in mm.

    Implementation note: the kinematic chain has serial dependencies, so we loop
    over joints (22 iterations) but operate on the full batch each step using
    broadcasted Rodrigues. Avoids per-sample Python overhead.
    """
    N = joint_angles_batch.shape[0]
    n = skel.n_joints
    world_T = np.tile(np.eye(4, dtype=np.float64), (N, n, 1, 1))   # (N, n, 4, 4)
    out_pos = np.zeros((N, n, 3), dtype=np.float64)

    for chain in skel.chains:
        for j in chain:
            axis = skel.rotation_axes[j]
            theta = joint_angles_batch[:, j].astype(np.float64)   # (N,)
            c = np.cos(theta);  s = np.sin(theta);  one_c = 1.0 - c
            ux, uy, uz = axis
            R = np.empty((N, 3, 3), dtype=np.float64)
            R[:, 0, 0] = c + ux*ux*one_c
            R[:, 0, 1] = ux*uy*one_c - uz*s
            R[:, 0, 2] = ux*uz*one_c + uy*s
            R[:, 1, 0] = uy*ux*one_c + uz*s
            R[:, 1, 1] = c + uy*uy*one_c
            R[:, 1, 2] = uy*uz*one_c - ux*s
            R[:, 2, 0] = uz*ux*one_c - uy*s
            R[:, 2, 1] = uz*uy*one_c + ux*s
            R[:, 2, 2] = c + uz*uz*one_c
            p = int(skel.parent[j])
            if p != 255 and p < n:
                parent_T = world_T[:, p]
                local_offset = skel.rest_positions[j] - skel.rest_positions[p]
            else:
                parent_T = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
                local_offset = skel.rest_positions[j]
            local_T = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            local_T[:, :3, :3] = R
            local_T[:, :3,  3] = local_offset
            world_T[:, j] = np.einsum("nij,njk->nik", parent_T, local_T)
            out_pos[:, j] = world_T[:, j, :3, 3]
    return out_pos


# ── Metrics ─────────────────────────────────────────────────────────────────

# Fingertip joint indices (last joint of each finger chain)
FINGERTIP_JOINTS = [3, 7, 11, 15, 19]   # Thumb DIP, Index DIP, Middle DIP, Ring DIP, Pinky DIP

# Finger groups for per-finger breakdown
FINGER_JOINT_GROUPS = {
    "Thumb":  [0,  1,  2,  3],
    "Index":  [4,  5,  6,  7],
    "Middle": [8,  9, 10, 11],
    "Ring":   [12, 13, 14, 15],
    "Pinky":  [16, 17, 18, 19],
}


def mpjpe(pred_angles: np.ndarray, tgt_angles: np.ndarray, skel: Skeleton) -> dict:
    """Compute MPJPE and related 3D-space metrics in mm.

    Args:
        pred_angles, tgt_angles: (N, 22) radians.
        skel: canonical Skeleton.

    Returns dict with:
        overall_mm:    mean per-joint position error over 20 active joints
        fingertip_mm:  MPJPE restricted to 5 fingertip joints (most visible)
        per_finger_mm: dict {finger_name: mm}
        per_joint_mm:  (22,) per-joint position error
        pck@thresholds: AUC of PCK curve from 0 to 50 mm
    """
    pred_pos = forward_kinematics_batch(pred_angles, skel)   # (N, 22, 3)
    tgt_pos  = forward_kinematics_batch(tgt_angles,  skel)
    err = np.linalg.norm(pred_pos - tgt_pos, axis=-1)         # (N, 22) mm

    per_joint_mm = err.mean(axis=0)                            # (22,)
    overall_mm   = float(per_joint_mm[ACTIVE_JOINTS].mean())
    fingertip_mm = float(per_joint_mm[FINGERTIP_JOINTS].mean())

    per_finger_mm = {
        f: float(per_joint_mm[idx].mean())
        for f, idx in FINGER_JOINT_GROUPS.items()
    }

    # PCK@thresh: fraction of frames where all-active-joint error < thresh
    thresholds = np.arange(0, 51, 2)  # 0..50 mm in 2 mm steps
    pck = []
    err_active = err[:, ACTIVE_JOINTS]                         # (N, 20)
    max_active = err_active.max(axis=1)                        # worst joint per frame
    for t in thresholds:
        pck.append(float((max_active <= t).mean()))
    # AUC (trapezoid) normalized to [0, 1] (auc / max_threshold)
    auc_pck = float(np.trapezoid(pck, thresholds) / thresholds.max())

    return {
        "overall_mpjpe_mm":      overall_mm,
        "fingertip_mpjpe_mm":    fingertip_mm,
        "per_finger_mpjpe_mm":   per_finger_mm,
        "per_joint_mpjpe_mm":    per_joint_mm.tolist(),
        "pck_thresholds_mm":     thresholds.tolist(),
        "pck_values":            pck,
        "pck_auc_normalized":    auc_pck,
    }


def mpjpe_phase_split(
    pred_angles: np.ndarray,
    tgt_angles:  np.ndarray,
    distances:   np.ndarray,
    skel:        Skeleton,
    grip_thresh: float = 0.10,
) -> dict:
    """Split MPJPE by interaction phase (grip vs pre-shape)."""
    grip_mask = distances < grip_thresh
    out = {}
    if grip_mask.any():
        out["grip"] = mpjpe(pred_angles[grip_mask], tgt_angles[grip_mask], skel)
    else:
        out["grip"] = None
    pre_mask = ~grip_mask
    if pre_mask.any():
        out["pre_shape"] = mpjpe(pred_angles[pre_mask], tgt_angles[pre_mask], skel)
    else:
        out["pre_shape"] = None
    return out


# ── Differentiable Torch FK (training loss) ─────────────────────────────────

try:
    import torch
    import torch.nn as nn

    class TorchFK(nn.Module):
        """Differentiable forward kinematics for UMeTrack 22-joint hand.

        Mirrors `forward_kinematics_batch` (numpy) but in PyTorch so it can be used
        as a training loss. Skeleton parameters are stored as non-trainable buffers.

        Input:  joint_angles (B, 22) radians.
        Output: positions    (B, 22, 3) mm, in the wrist frame.
        """

        def __init__(self, skel: Skeleton):
            super().__init__()
            self.register_buffer("rest_positions", torch.tensor(skel.rest_positions, dtype=torch.float32))
            self.register_buffer("rotation_axes",  torch.tensor(skel.rotation_axes,  dtype=torch.float32))
            self.parent = skel.parent.astype(np.int32)
            self.chains = [list(c) for c in skel.chains]
            self.n_joints = int(skel.n_joints)

        def _axis_angle_rot(self, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            """Rodrigues per batch. axis: (3,)  theta: (B,)  → R: (B, 3, 3)."""
            B = theta.shape[0]
            c = torch.cos(theta); s = torch.sin(theta); one_c = 1.0 - c
            ux, uy, uz = axis[0], axis[1], axis[2]
            R = theta.new_zeros((B, 3, 3))
            R[:, 0, 0] = c + ux * ux * one_c
            R[:, 0, 1] = ux * uy * one_c - uz * s
            R[:, 0, 2] = ux * uz * one_c + uy * s
            R[:, 1, 0] = uy * ux * one_c + uz * s
            R[:, 1, 1] = c + uy * uy * one_c
            R[:, 1, 2] = uy * uz * one_c - ux * s
            R[:, 2, 0] = uz * ux * one_c - uy * s
            R[:, 2, 1] = uz * uy * one_c + ux * s
            R[:, 2, 2] = c + uz * uz * one_c
            return R

        def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
            B = joint_angles.shape[0]
            n = self.n_joints
            device, dtype = joint_angles.device, joint_angles.dtype
            eye4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4)
            # Dict of per-joint world transforms (avoids in-place writes for autograd).
            world_T: dict[int, torch.Tensor] = {}
            positions: list[torch.Tensor] = [None] * n  # type: ignore

            for chain in self.chains:
                for j in chain:
                    axis  = self.rotation_axes[j]
                    theta = joint_angles[:, j]
                    R = self._axis_angle_rot(axis, theta)               # (B, 3, 3)
                    p = int(self.parent[j])
                    if p != 255 and p < n and p in world_T:
                        parent_T = world_T[p]
                        local_offset = self.rest_positions[j] - self.rest_positions[p]
                    else:
                        parent_T = eye4
                        local_offset = self.rest_positions[j]
                    # Build local_T = [[R, t], [0, 1]] without in-place ops.
                    t = local_offset.unsqueeze(0).expand(B, 3).unsqueeze(-1)   # (B, 3, 1)
                    bottom = torch.tensor([[0., 0., 0., 1.]], device=device, dtype=dtype)
                    bottom = bottom.unsqueeze(0).expand(B, 1, 4)
                    top    = torch.cat([R, t], dim=-1)                         # (B, 3, 4)
                    local_T = torch.cat([top, bottom], dim=1)                  # (B, 4, 4)
                    Tj = torch.matmul(parent_T, local_T)
                    world_T[j] = Tj
                    positions[j] = Tj[:, :3, 3]

            # Fill any joints not in chains with zeros (placeholders 20/21).
            for j in range(n):
                if positions[j] is None:
                    positions[j] = torch.zeros(B, 3, device=device, dtype=dtype)
            return torch.stack(positions, dim=1)                               # (B, n, 3)

except ImportError:  # torch optional for some scripts
    pass


# ── Sanity self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent.parent
    skel = load_canonical_skeleton(root)
    print(f"Skeleton loaded: {skel.n_joints} joints, {len(skel.chains)} chains")
    for i, c in enumerate(skel.chains):
        print(f"  Chain {i}: {c}")
    # Zero pose
    zero_pos = forward_kinematics(np.zeros(22), skel)
    print(f"\nRest-pose joint positions (wrist-local, mm) — first 8:")
    for i in range(8):
        print(f"  J{i}: {zero_pos[i]}")

    # MPJPE of zero vs zero should be 0.0
    pred = np.zeros((4, 22))
    tgt  = np.zeros((4, 22))
    m = mpjpe(pred, tgt, skel)
    assert m["overall_mpjpe_mm"] < 1e-6, f"identity MPJPE not zero: {m['overall_mpjpe_mm']}"

    # Non-trivial: bend index PIP by 60° → check fingertip moves
    pred[:, 6] = np.deg2rad(60)   # Index PIP
    m = mpjpe(pred, tgt, skel)
    print(f"\nIndex PIP 60° bend:")
    print(f"  Overall MPJPE: {m['overall_mpjpe_mm']:.2f} mm")
    print(f"  Fingertip MPJPE: {m['fingertip_mpjpe_mm']:.2f} mm")
    print(f"  Per-finger: {m['per_finger_mpjpe_mm']}")
    print(f"  PCK AUC: {m['pck_auc_normalized']:.3f}")
    print("\nfk.py self-test passed.")
