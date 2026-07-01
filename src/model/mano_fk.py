"""Differentiable MANO forward kinematics for finger_aa45.

Implements the exact MANO kinematic chain (zero-beta template joints + per-joint
axis-angle rotations) so that fingertip positions are geometrically consistent with:
  - HOT3D stored fk_joints (computed by utils/mano_right.py, world frame)
  - OakInk hand_j ground-truth joints (camera frame, converted to world)
  - obj_pts_contact (object surface in MANO wrist frame)

The wrist global orient is NOT applied here; all outputs are in wrist frame
(wrist at origin), consistent with how obj_pts_contact is computed in the dataset
loaders (canonical->world->wrist via @ R_wrist).

Template joint positions (_J_TEMPLATE) are derived from MANO model at zero betas
(utils/mano_right.ManoRight at zero shape) and expressed relative to the wrist joint.
These are fixed constants that do not require the MANO .pkl at inference time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# MANO kinematic tree (zero-beta template, wrist-relative joint positions)
# Derived from utils/mano_right.ManoRight with zero betas; J[i] - J[0].
# Joint order: wrist(0), index(1-3), middle(4-6), ring(7-9), pinky(10-12), thumb(13-15)
# where each finger has 3 joints (proximal/MCP-equivalent, PIP, DIP).
# ---------------------------------------------------------------------------
_J_TEMPLATE = torch.tensor([
    # wrist (origin)
    [ 0.0000000,  0.0000000,  0.0000000],
    # index finger: MCP, PIP, DIP
    [-0.0881272, -0.0051603,  0.0206860],
    [-0.1207761, -0.0011910,  0.0229031],
    [-0.1429320,  0.0038940,  0.0228889],  # <- last = DIP/tip proxy
    # middle finger: MCP, PIP, DIP
    [-0.0946604, -0.0014789, -0.0033575],
    [-0.1258431,  0.0006824, -0.0089520],
    [-0.1487477,  0.0005137, -0.0128966],
    # ring finger (chain 7-9, more negative Z)
    [-0.0688869, -0.0099403, -0.0432093],
    [-0.0858914, -0.0098785, -0.0557081],
    [-0.1016683, -0.0105696, -0.0660400],
    # pinky finger (chain 10-12)
    [-0.0817355,  0.0024260, -0.0266732],
    [-0.1100498,  0.0044930, -0.0317717],
    [-0.1337703,  0.0028049, -0.0394055],
    # thumb (chain 13-15)
    [-0.0240897, -0.0155223,  0.0257929],
    [-0.0437229, -0.0082476,  0.0494924],
    [-0.0659407, -0.0136806,  0.0640165],
], dtype=torch.float32)

# Parent indices: -1 = root (wrist). Same as utils/mano_right.ManoRight.parents.
_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

# J_rel[i] = J[i] - J[parent[i]]  (precomputed, fixed)
_J_REL = _J_TEMPLATE.clone()
for _i in range(1, 16):
    _J_REL[_i] = _J_TEMPLATE[_i] - _J_TEMPLATE[_PARENTS[_i]]

# Fingertip = last joint in each finger chain (indices 3, 6, 9, 12, 15).
_TIP_INDICES = torch.tensor([3, 6, 9, 12, 15], dtype=torch.long)

# Keep these exports for quality_labels.py and model_io imports that reference them.
FINGERTIP_JOINT_INDICES = [2, 5, 8, 11, 14]   # 0-based within the 15 finger-joint array


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix via Rodrigues' formula."""
    angle = axis_angle.norm(dim=-1, keepdim=True)
    eps = 1e-8
    axis = axis_angle / angle.clamp_min(eps)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(
        *axis_angle.shape[:-1], 3, 3
    )
    angle_bc = angle[..., None]
    R = eye + angle_bc.sin() * K + (1 - angle_bc.cos()) * (K @ K)
    return torch.where(angle.squeeze(-1).unsqueeze(-1).unsqueeze(-1) < eps, eye, R)


def _make_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build 4×4 rigid transform from (...,3,3) R and (...,3) t without inplace ops."""
    *batch, _, _ = R.shape
    device, dtype = R.device, R.dtype
    bottom = torch.tensor([0., 0., 0., 1.], device=device, dtype=dtype)
    bottom = bottom.expand(*batch, 1, 4)
    t_col = t.unsqueeze(-1)                          # (...,3,1)
    top = torch.cat([R, t_col], dim=-1)              # (...,3,4)
    return torch.cat([top, bottom], dim=-2)          # (...,4,4)


def finger_aa45_to_joint_positions(finger_aa45: torch.Tensor) -> torch.Tensor:
    """(..., 45) MANO finger axis-angle -> (..., 16, 3) joint positions in wrist frame.

    Implements the MANO kinematic chain exactly:
      G[0]  = [I | J_rel[0]]
      G[i]  = G[parent[i]] @ [R[i] | J_rel[i]]   for i in 1..15
      output = G[i][:3, 3] - G[0][:3, 3]          (wrist-relative)

    All tensor operations are out-of-place so autograd backward works correctly.
    Wrist global orient is NOT applied; outputs are in MANO wrist frame.
    """
    *batch, _ = finger_aa45.shape
    device, dtype = finger_aa45.device, finger_aa45.dtype

    # Rotation matrices for joints 1-15.
    aa15 = finger_aa45.reshape(*batch, 15, 3)
    R_fingers = axis_angle_to_matrix(aa15)   # (*,15,3,3)

    # Identity rotation for wrist (joint 0).
    eye3 = torch.eye(3, device=device, dtype=dtype).expand(*batch, 3, 3)

    # Template relative positions on device.
    J_rel = _J_REL.to(device=device, dtype=dtype)  # (16,3)

    # Build G list: G[i] is the 4×4 world transform for joint i.
    # Each G[i] = G[parent[i]] @ T_local[i]  (out-of-place matmul).
    G: list[torch.Tensor] = []

    # Joint 0: wrist
    j0_t = J_rel[0].expand(*batch, 3)
    G.append(_make_T(eye3, j0_t))

    # Joints 1-15: finger chain
    for i in range(1, 16):
        R_i = R_fingers[..., i - 1, :, :]           # (*,3,3)
        t_i = J_rel[i].expand(*batch, 3)             # (*,3)
        T_local = _make_T(R_i, t_i)                  # (*,4,4)
        G.append(G[_PARENTS[i]] @ T_local)           # (*,4,4)

    # Stack and extract wrist-relative joint positions.
    G_stacked = torch.stack(G, dim=-3)               # (*,16,4,4)
    joints = G_stacked[..., :3, 3]                   # (*,16,3)
    joints = joints - joints[..., 0:1, :]            # wrist at origin
    return joints


def fingertip_positions(finger_aa45: torch.Tensor) -> torch.Tensor:
    """(..., 45) -> (..., 5, 3) fingertip positions in MANO wrist frame.

    Fingertip = last joint in each finger chain: indices [3,6,9,12,15]
    (index, middle, ring, pinky, thumb — matching HOT3D fk_joints convention).
    Same frame as obj_pts_contact so cdist is geometrically meaningful.
    """
    joints = finger_aa45_to_joint_positions(finger_aa45)   # (*,16,3)
    idx = _TIP_INDICES.to(finger_aa45.device)
    return joints.index_select(-2, idx)                     # (*,5,3)
