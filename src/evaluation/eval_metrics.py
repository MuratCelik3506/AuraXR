"""Offline evaluation metrics (docs/B-grasp-model.md B10).

1. Geodesic rotation error (main metric)
2. MPJPE / fingertip position error
3. Joint-limit violation rate
4. Contact ratio
5. Penetration depth
6. Confidence calibration (AUC-ROC, ECE)
7. Diversity score (CVAE)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import _LOWER, _UPPER  # noqa: E402
from model.mano_fk import axis_angle_to_matrix, fingertip_positions  # noqa: E402
from model.model_io import EVAL_CONTACT_THRESHOLD_M  # noqa: E402


def angle_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over MANO axis-angle components (debug/auxiliary only, B10)."""
    return (pred - target).abs().mean()


def pose_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


def geodesic_rotation_error(pred: torch.Tensor, target: torch.Tensor, degrees: bool = True) -> torch.Tensor:
    """B10.1 -- main metric. pred/target: (...,45) axis-angle -> per-joint geodesic
    error d(R_pred,R_gt) = arccos((trace(R_pred^T R_gt)-1)/2), averaged over all joints.
    """
    shape_prefix = pred.shape[:-1]
    R_pred = axis_angle_to_matrix(pred.reshape(*shape_prefix, 15, 3))
    R_gt = axis_angle_to_matrix(target.reshape(*shape_prefix, 15, 3))
    R_rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    if degrees:
        angle = angle * (180.0 / torch.pi)
    return angle.mean()


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """B10.2 -- mean per-joint position error via lightweight FK, in millimeters."""
    from model.mano_fk import finger_aa45_to_joint_positions

    pred_pts = finger_aa45_to_joint_positions(pred)
    gt_pts = finger_aa45_to_joint_positions(target)
    return ((pred_pts - gt_pts).norm(dim=-1).mean()) * 1000.0


def fingertip_position_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """B10.2 -- mean fingertip position error over the 5 fingertips, in millimeters."""
    pred_tips = fingertip_positions(pred)
    gt_tips = fingertip_positions(target)
    return ((pred_tips - gt_tips).norm(dim=-1).mean()) * 1000.0


def joint_limit_violation_rate(pred: torch.Tensor) -> torch.Tensor:
    """B10.3 -- fraction of joint-angle components outside anatomical limits."""
    lower = _LOWER.to(pred.device)
    upper = _UPPER.to(pred.device)
    violated = (pred < lower) | (pred > upper)
    return violated.float().mean()


def joint_limit_saturation_rate(pred: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Hard clamp kaldırıldıktan sonra artık anlamsız, ama geriye dönük uyumluluk için.
    Anatomik sınıra eps'ten yakın olan bileşenlerin oranı.
    Düşük = model sınırları zorlamıyor, doğal dağılım var."""
    lower = _LOWER.to(pred.device)
    upper = _UPPER.to(pred.device)
    near_lower = (pred - lower).abs() < eps
    near_upper = (upper - pred).abs() < eps
    return (near_lower | near_upper).float().mean()


def contact_ratio(pred: torch.Tensor, obj_pts: torch.Tensor, threshold_m: float = EVAL_CONTACT_THRESHOLD_M) -> torch.Tensor:
    """B10.4 -- fraction of fingertips within `threshold_m` of the nearest object point."""
    tips = fingertip_positions(pred)  # (B,5,3)
    dists = torch.cdist(tips, obj_pts).min(dim=-1).values  # (B,5)
    in_contact = (dists < threshold_m).float()
    return in_contact.mean()


def penetration_depth_mm(pred: torch.Tensor, obj_pts: torch.Tensor) -> torch.Tensor:
    """B10.5 -- mean fingertip penetration depth in mm using the same point-cloud proxy
    as the L_penetration training loss (no watertight mesh/SDF available)."""
    tips = fingertip_positions(pred)
    centroid = obj_pts.mean(dim=1, keepdim=True)
    tip_to_centroid = (tips - centroid).norm(dim=-1)
    obj_radius = (obj_pts - centroid).norm(dim=-1).mean(dim=-1, keepdim=True)
    penetration = torch.relu(obj_radius - tip_to_centroid)
    return penetration.mean() * 1000.0


def diversity_score(candidate_poses: torch.Tensor) -> torch.Tensor:
    """B10.5(diversity) -- mean pairwise distance between K CVAE samples per item.
    candidate_poses: (B,K,45)."""
    batch, k, _ = candidate_poses.shape
    if k < 2:
        return candidate_poses.new_tensor(0.0)
    diffs = candidate_poses[:, :, None, :] - candidate_poses[:, None, :, :]  # (B,K,K,45)
    dists = diffs.norm(dim=-1)  # (B,K,K)
    mask = ~torch.eye(k, dtype=torch.bool, device=candidate_poses.device)
    return dists[:, mask].mean()


def jitter_velocity(poses: torch.Tensor) -> torch.Tensor:
    """Mean frame-to-frame pose velocity magnitude for (..., T, 45)."""
    if poses.shape[-2] < 2:
        return poses.new_tensor(0.0)
    return (poses[..., 1:, :] - poses[..., :-1, :]).norm(dim=-1).mean()


def jitter_acceleration(poses: torch.Tensor) -> torch.Tensor:
    """Mean second difference magnitude for (..., T, 45)."""
    if poses.shape[-2] < 3:
        return poses.new_tensor(0.0)
    acc = poses[..., 2:, :] - 2 * poses[..., 1:-1, :] + poses[..., :-2, :]
    return acc.norm(dim=-1).mean()


def jitter_score(poses: torch.Tensor) -> torch.Tensor:
    """D4 -- jitter score = max velocity / mean velocity over a (..., T, 45) sequence.
    Higher = spikier, less stable motion."""
    if poses.shape[-2] < 2:
        return poses.new_tensor(0.0)
    vel = (poses[..., 1:, :] - poses[..., :-1, :]).norm(dim=-1)  # (..., T-1)
    mean_vel = vel.mean(dim=-1).clamp_min(1e-8)
    max_vel = vel.max(dim=-1).values
    return (max_vel / mean_vel).mean()


def geodesic_velocity(poses: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame geodesic rotation distance (ana jitter metriği, D4), (..., T, 45)."""
    if poses.shape[-2] < 2:
        return poses.new_tensor(0.0)
    return geodesic_rotation_error(poses[..., 1:, :], poses[..., :-1, :])


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Small dependency-free ROC AUC for binary labels."""
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += np.sum(value > neg) + 0.5 * np.sum(value == neg)
    return float(wins / (len(pos) * len(neg)))


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not np.any(mask):
            continue
        conf = scores[mask].mean()
        acc = labels[mask].mean()
        ece += np.abs(conf - acc) * mask.mean()
    return float(ece)


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Dependency-free Spearman rank correlation (D6: quality_score vs Unity success rate)."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    ra = _rankdata(a)
    rb = _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average ranks for ties
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def oracle_rank(success_probs: np.ndarray, true_success: np.ndarray) -> float:
    """D6 -- mean rank (1=best) of the argmax(success_prob)-selected candidate among
    the K candidates, ranked by true_success descending. Lower is better."""
    success_probs = np.asarray(success_probs)  # (K,)
    true_success = np.asarray(true_success)  # (K,)
    selected = int(np.argmax(success_probs))
    order = np.argsort(-true_success)
    rank = int(np.where(order == selected)[0][0]) + 1
    return float(rank)
