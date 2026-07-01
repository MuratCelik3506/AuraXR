"""A6 -- training-time augmentation: random yaw rotation about the object's up axis,
plus small positional/point-cloud jitter.

Yaw only rotates wrist-relative quantities and the object point cloud; MANO finger
pose (finger_aa45) is defined in the wrist-local frame and is yaw-invariant, so it
is left untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import axis_angle_to_rot6d  # noqa: E402
from model.mano_fk import axis_angle_to_matrix  # noqa: E402


def random_yaw_matrix(rng: np.random.Generator) -> torch.Tensor:
    angle = float(rng.uniform(-np.pi, np.pi))
    axis_angle = torch.tensor([0.0, 0.0, angle], dtype=torch.float32)
    return axis_angle_to_matrix(axis_angle)


def augment_yaw(
    rel_pos: torch.Tensor,
    global_orient_aa: torch.Tensor,
    obj_pts: torch.Tensor,
    rng: np.random.Generator,
    point_noise_std: float = 0.003,
) -> dict[str, torch.Tensor]:
    """rel_pos:(3,), global_orient_aa:(3,), obj_pts:(N,3) -> yaw-rotated versions.

    Rotates wrist position and global orientation by a random yaw, and applies the
    same rotation to the object point cloud so the relative geometry is preserved,
    then adds small Gaussian jitter to the points.
    """
    R_yaw = random_yaw_matrix(rng)
    rel_pos_aug = R_yaw @ rel_pos
    R_orient = axis_angle_to_matrix(global_orient_aa)
    R_orient_aug = R_yaw @ R_orient
    rot6d_aug = torch.cat([R_orient_aug[:, 0], R_orient_aug[:, 1]], dim=-1)
    obj_pts_aug = obj_pts @ R_yaw.T
    noise = torch.from_numpy(rng.normal(0.0, point_noise_std, size=obj_pts_aug.shape).astype(np.float32))
    obj_pts_aug = obj_pts_aug + noise
    return {"rel_pos": rel_pos_aug, "rel_rot6d": rot6d_aug, "obj_pts": obj_pts_aug}


def augment_frame_feat(frame_feat: torch.Tensor, obj_pts: torch.Tensor, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """frame_feat: (...,13) = [rel_pos(3), rel_rot6d(6), rel_vel(3), dist(1)].

    Applies one shared random yaw to every frame in the window plus the point cloud;
    rel_vel and dist are rotation invariant in magnitude (dist) or rotated alongside
    rel_pos (rel_vel), so both are recomputed via the same R_yaw.
    """
    R_yaw = random_yaw_matrix(rng)
    rel_pos = frame_feat[..., 0:3]
    rot6d = frame_feat[..., 3:9]
    rel_vel = frame_feat[..., 9:12]
    dist = frame_feat[..., 12:13]

    rel_pos_aug = rel_pos @ R_yaw.T
    rel_vel_aug = rel_vel @ R_yaw.T
    a1, a2 = rot6d[..., 0:3], rot6d[..., 3:6]
    a1_aug = a1 @ R_yaw.T
    a2_aug = a2 @ R_yaw.T
    rot6d_aug = torch.cat([a1_aug, a2_aug], dim=-1)

    frame_feat_aug = torch.cat([rel_pos_aug, rot6d_aug, rel_vel_aug, dist], dim=-1)
    obj_pts_aug = obj_pts @ R_yaw.T
    return {"frame_feat": frame_feat_aug, "obj_pts": obj_pts_aug}
