"""Temporal Geometry-Conditioned Grasp Model (docs/B-grasp-model.md).

One learned system, unified input interface (B1):
  frame_feat (B,T,13) -- OakInk uses T=1 (rel_vel=0), HOT3D/Unity use T>1.

Pipeline (B0): PointNet object encoder -> GRU temporal encoder (B3) -> fusion
with previous pose + object feature -> self-attention over 15 finger joints
(B4) -> CVAE decoder (B6) -> {finger_aa45, quality_score, success_prob} (B7).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.mano_fk import fingertip_positions  # noqa: E402
from model.model_io import (  # noqa: E402
    DEFAULT_N_POINTS,
    FINGER_POSE_DIM,
    HOT3D_FRAME_DIM,
    JOINT_TO_FINGER,
    LATENT_DIM,
    NUM_FINGER_JOINTS,
    OBJ_EMB_DIM,
    TRAIN_CONTACT_HINGE_M,
    WRIST_DIM,
)


class PointNetEncoder(nn.Module):
    """Mini PointNet object encoder (B2): (B,N,3) -> (B,out_dim)."""

    def __init__(self, out_dim: int = OBJ_EMB_DIM, in_dim: int = 3) -> None:
        super().__init__()
        hidden = out_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, obj_pts: torch.Tensor) -> torch.Tensor:
        x = self.point_mlp(obj_pts)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.out(pooled)


class FiLM(nn.Module):
    """Feature-wise linear modulation (B5): object feature scales/shifts activations."""

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return feat * (1.0 + gamma) + beta


class TemporalEncoder(nn.Module):
    """GRU temporal encoder over the unified frame_feat (B3).

    Input per docs: [rel_pos(3), rel_rot6d(6), rel_vel(3), dist(1), contact_flag(1)] = 14 dims.
    Causal (only past frames). T=1 (OakInk) runs the GRU for a single step.
    """

    def __init__(self, frame_dim: int = HOT3D_FRAME_DIM, hidden: int = 256, layers: int = 1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(frame_dim + 1, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)

    def forward(self, frame_feat: torch.Tensor, contact_flag: torch.Tensor | None = None) -> torch.Tensor:
        if contact_flag is None:
            contact_flag = frame_feat.new_zeros((*frame_feat.shape[:2], 1))
        x = torch.cat([frame_feat, contact_flag], dim=-1)
        x = self.input_proj(x)
        _, h = self.gru(x)
        return h[-1]


class JointSelfAttention(nn.Module):
    """Self-attention over the 15 MANO finger joints (B4).

    Each of the 15 tokens is built from a shared fusion context plus a learned
    per-joint identity embedding plus an embedding of that joint's previous
    pose -- without per-joint identity all 15 tokens start identical and
    attention degenerates to a no-op.
    """

    def __init__(self, dim: int = 128, num_joints: int = NUM_FINGER_JOINTS, num_heads: int = 4) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.dim = dim
        self.learned_joint_emb = nn.Parameter(torch.randn(num_joints, dim) * 0.02)
        self.prev_pose_proj = nn.Linear(3, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, fusion_out: torch.Tensor, prev_pose: torch.Tensor) -> torch.Tensor:
        """fusion_out: (B, dim). prev_pose: (B, 45) previous/canonical finger pose. -> (B, 15, dim)."""
        batch = fusion_out.shape[0]
        prev_pose_emb = self.prev_pose_proj(prev_pose.view(batch, self.num_joints, 3))
        tokens = (
            fusion_out[:, None, :].expand(batch, self.num_joints, self.dim)
            + self.learned_joint_emb[None, :, :]
            + prev_pose_emb
        )
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens


class ContextEncoder(nn.Module):
    """Builds the shared conditioning context fed into self-attention + CVAE."""

    def __init__(self, obj_dim: int = OBJ_EMB_DIM, hidden: int = 256) -> None:
        super().__init__()
        self.temporal_proj = nn.Linear(hidden, hidden)
        self.film = FiLM(obj_dim, hidden)
        self.backbone = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, temporal_feat: torch.Tensor, obj_emb: torch.Tensor) -> torch.Tensor:
        h = self.temporal_proj(temporal_feat)
        return self.backbone(self.film(h, obj_emb))


class GraspCVAE(nn.Module):
    """Conditional VAE decoder (B6) operating on attended per-joint tokens."""

    def __init__(self, joint_dim: int = 128, num_joints: int = NUM_FINGER_JOINTS, z_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        flat_dim = joint_dim * num_joints
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim + num_joints * 3, flat_dim // 2),
            nn.GELU(),
            nn.Linear(flat_dim // 2, z_dim * 2),
        )
        self.decoder_joint_head = nn.Sequential(
            nn.Linear(joint_dim + z_dim, joint_dim),
            nn.GELU(),
            nn.Linear(joint_dim, 3),
        )

    def encode(self, joint_tokens: torch.Tensor, target_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = joint_tokens.reshape(joint_tokens.shape[0], -1)
        mu, logvar = self.encoder(torch.cat([flat, target_pose], dim=-1)).chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, joint_tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """joint_tokens: (B, 15, joint_dim), z: (B, z_dim) -> (B, 45)."""
        batch = joint_tokens.shape[0]
        z_expanded = z[:, None, :].expand(batch, self.num_joints, self.z_dim)
        per_joint = self.decoder_joint_head(torch.cat([joint_tokens, z_expanded], dim=-1))  # (B,15,3)
        out = per_joint.reshape(batch, self.num_joints * 3)
        # Hard clamp kaldırıldı: joint_limit_loss (soft) gradient ile öğretir.
        # Clamp gradient'i keser ve violation_rate'i yapay sıfır gösterir.
        return out

    def forward_train(
        self, joint_tokens: torch.Tensor, target_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(joint_tokens, target_pose)
        pred = self.decode(joint_tokens, self.reparameterize(mu, logvar))
        return pred, mu, logvar

    def sample(self, joint_tokens: torch.Tensor, k: int = 1) -> torch.Tensor:
        batch = joint_tokens.shape[0]
        z = torch.randn(batch, k, self.z_dim, device=joint_tokens.device, dtype=joint_tokens.dtype)
        tokens_expanded = joint_tokens[:, None, :, :].expand(batch, k, self.num_joints, self.joint_dim)
        flat_tokens = tokens_expanded.reshape(batch * k, self.num_joints, self.joint_dim)
        flat_z = z.reshape(batch * k, self.z_dim)
        flat_pred = self.decode(flat_tokens, flat_z)
        return flat_pred.reshape(batch, k, -1)


class TemporalGeometryConditionedGraspModel(nn.Module):
    """Single main model for static (T=1) and temporal (T>1) grasp refinement."""

    def __init__(
        self,
        obj_dim: int = OBJ_EMB_DIM,
        hidden: int = 256,
        joint_dim: int = 128,
        z_dim: int = LATENT_DIM,
        encoder_type: str = "gru",  # "gru" | "mlp" — ablation A3: temporal encoder choice
        obj_encoder_type: str = "pointnet",  # "pointnet" | "bbox" | "none" — ablation A3
        use_attention: bool = True,   # ablation A3: self-attention over joints
        use_film: bool = True,        # ablation A3: FiLM conditioning vs concat
        use_success_head: bool = True,  # False when Phase 3 labels unavailable
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.joint_dim = joint_dim
        self.encoder_type = encoder_type
        self.obj_encoder_type = obj_encoder_type
        self.use_attention = use_attention
        self.use_film = use_film

        # Object encoder (ablation: pointnet / bbox / none)
        if obj_encoder_type == "pointnet":
            self.obj_encoder = PointNetEncoder(obj_dim)
        elif obj_encoder_type == "bbox":
            # 6-dim (min + max) + 3-dim aspect ratio = 9 dims -> obj_dim
            self.obj_encoder = nn.Sequential(nn.Linear(9, obj_dim), nn.GELU(), nn.Linear(obj_dim, obj_dim))
        else:  # "none"
            self.obj_encoder = None

        # Temporal encoder (ablation: gru / mlp)
        if encoder_type == "gru":
            self.temporal_encoder = TemporalEncoder(hidden=hidden)
        else:  # "mlp" — single-frame linear projection
            self.temporal_encoder = nn.Sequential(
                nn.Linear(HOT3D_FRAME_DIM + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )

        # Context encoder (FiLM vs concat ablation)
        if use_film:
            self.context_encoder = ContextEncoder(obj_dim=obj_dim, hidden=hidden)
        else:
            # Simple concat+linear instead of FiLM
            self.context_encoder = nn.Sequential(
                nn.Linear(hidden + obj_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
            )
        self.context_to_joint = nn.Linear(hidden, joint_dim)
        # Attention ablation: full self-attention vs independent per-joint MLP
        if use_attention:
            self.self_attn: nn.Module = JointSelfAttention(dim=joint_dim)
        else:
            self.self_attn = nn.Sequential(nn.Linear(joint_dim + 3, joint_dim), nn.GELU(), nn.Linear(joint_dim, joint_dim))
        self.cvae = GraspCVAE(joint_dim=joint_dim, z_dim=z_dim)

        # B7: two separate confidence heads -- heuristic quality (MSE) vs Unity success (BCE).
        # quality_head receives both joint_tokens AND pred_pose so it can score each
        # candidate differently when k>1 samples are drawn at inference time.
        # Known limitation: during training pred_pose comes from posterior z (CVAE encoder),
        # but at inference candidates come from prior z~N(0,I). The joint_tokens part
        # (deterministic context) still provides a meaningful discrimination signal.
        flat_joint_dim = joint_dim * NUM_FINGER_JOINTS
        self.quality_head = nn.Sequential(
            nn.Linear(flat_joint_dim + FINGER_POSE_DIM, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        # success_head requires Phase 3 Unity labels; skip when use_success_head=False.
        self.use_success_head = use_success_head
        if use_success_head:
            self.success_head: nn.Module | None = nn.Sequential(
                nn.Linear(flat_joint_dim + FINGER_POSE_DIM, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.success_head = None

    def encode_object(self, obj_pts: torch.Tensor) -> torch.Tensor:
        if self.obj_encoder is None:
            return obj_pts.new_zeros(obj_pts.shape[0], self.hidden)
        if self.obj_encoder_type == "bbox":
            mn = obj_pts.min(dim=1).values   # (B,3)
            mx = obj_pts.max(dim=1).values   # (B,3)
            aspect = (mx - mn) / (mx - mn).norm(dim=-1, keepdim=True).clamp(min=1e-6)
            bbox_feat = torch.cat([mn, mx, aspect], dim=-1)  # (B,9)
            return self.obj_encoder(bbox_feat)
        return self.obj_encoder(obj_pts)

    def build_context(
        self,
        frame_feat: torch.Tensor,
        obj_pts: torch.Tensor,
        contact_flag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """frame_feat: (B,T,13) unified interface (B1). T=1 for OakInk, T>1 for HOT3D/Unity."""
        if self.encoder_type == "gru":
            temporal = self.temporal_encoder(frame_feat, contact_flag)
        else:  # mlp: use last frame only
            if contact_flag is None:
                cf = frame_feat.new_zeros(*frame_feat.shape[:2], 1)
            else:
                cf = contact_flag
            last = torch.cat([frame_feat[:, -1, :], cf[:, -1, :]], dim=-1)  # (B, frame_dim+1)
            temporal = self.temporal_encoder(last)  # (B, hidden)
        obj_emb = self.encode_object(obj_pts)
        if self.use_film:
            return self.context_encoder(temporal, obj_emb)
        else:
            return self.context_encoder(torch.cat([temporal, obj_emb], dim=-1))

    def joint_tokens_from_context(self, context: torch.Tensor, prev_pose: torch.Tensor) -> torch.Tensor:
        fusion = self.context_to_joint(context)
        if self.use_attention:
            return self.self_attn(fusion, prev_pose)
        else:
            # Per-joint independent MLP: expand context + per-joint prev_pose
            batch = fusion.shape[0]
            per_joint_prev = prev_pose.view(batch, NUM_FINGER_JOINTS, 3)  # (B,15,3)
            fusion_exp = fusion[:, None, :].expand(batch, NUM_FINGER_JOINTS, self.joint_dim)
            joint_input = torch.cat([fusion_exp, per_joint_prev], dim=-1)  # (B,15, joint_dim+3)
            return self.self_attn(joint_input)

    def _heads(
        self,
        joint_tokens: torch.Tensor,
        pred_pose: torch.Tensor,
        compute_success: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flat = joint_tokens.reshape(joint_tokens.shape[0], -1)
        combined = torch.cat([flat, pred_pose], dim=-1)
        quality = self.quality_head(combined)
        success = (self.success_head(combined) if compute_success else None) if self.success_head is not None else None
        return quality, success

    def forward_train(
        self,
        frame_feat: torch.Tensor,
        obj_pts: torch.Tensor,
        target_pose: torch.Tensor,
        prev_pose: torch.Tensor | None = None,
        contact_flag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prev_pose is None:
            prev_pose = torch.zeros_like(target_pose)
        context = self.build_context(frame_feat, obj_pts, contact_flag)
        joint_tokens = self.joint_tokens_from_context(context, prev_pose)
        pred, mu, logvar = self.cvae.forward_train(joint_tokens, target_pose)
        quality, success = self._heads(joint_tokens, pred, compute_success=self.use_success_head)
        out: dict[str, torch.Tensor] = {"pred": pred, "mu": mu, "logvar": logvar, "quality_score": quality}
        if success is not None:
            out["success_prob"] = success
        return out

    @torch.no_grad()
    def infer(
        self,
        frame_feat: torch.Tensor,
        obj_pts: torch.Tensor,
        prev_pose: torch.Tensor | None = None,
        contact_flag: torch.Tensor | None = None,
        k: int = 1,
    ) -> dict[str, torch.Tensor]:
        if prev_pose is None:
            prev_pose = torch.zeros(frame_feat.shape[0], FINGER_POSE_DIM, device=frame_feat.device, dtype=frame_feat.dtype)
        context = self.build_context(frame_feat, obj_pts, contact_flag)
        joint_tokens = self.joint_tokens_from_context(context, prev_pose)
        return self._sample_and_select(joint_tokens, k)

    def _sample_and_select(self, joint_tokens: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
        candidates = self.cvae.sample(joint_tokens, k=max(1, k))  # (B,K,45)
        batch, n, _ = candidates.shape
        flat = joint_tokens.reshape(batch, -1)                     # (B, flat_joint_dim)
        flat_exp = flat[:, None, :].expand(batch, n, flat.shape[-1])  # (B,K,flat_joint_dim)

        # quality per-candidate: joint context + each candidate pose → distinct score per k.
        quality_input = torch.cat([flat_exp, candidates], dim=-1).reshape(batch * n, -1)
        quality = self.quality_head(quality_input).reshape(batch, n)  # (B,K)

        # D2: selection uses quality_score heuristic (Spearman 0.72 on OakInk).
        # success_head requires Unity physics labels (Phase 3); skip when unavailable.
        best_idx = quality.argmax(dim=1)
        arange = torch.arange(batch, device=candidates.device)
        selected = candidates[arange, best_idx]

        out: dict[str, torch.Tensor] = {
            "candidate_poses": candidates,
            "candidate_quality_scores": quality,
            "selected_pose": selected,
            "selected_quality_score": quality[arange, best_idx].unsqueeze(-1),
            "selected_candidate_index": best_idx,
        }
        if self.success_head is not None:
            success = self.success_head(quality_input).reshape(batch, n)
            out["candidate_success_probs"] = success
            out["selected_success_prob"] = success[arange, best_idx].unsqueeze(-1)
        return out

    # --- Backward-compatible static/temporal wrappers used by existing scripts ---

    def forward_static_train(
        self, wrist_feat: torch.Tensor, obj_pts: torch.Tensor, target_pose: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        frame_feat = wrist_feat_to_frame_feat(wrist_feat)
        out = self.forward_train(frame_feat, obj_pts, target_pose)
        out["conf"] = out["quality_score"]
        return out

    def forward_temporal_train(
        self,
        frame_feat: torch.Tensor,
        finger_hist: torch.Tensor,
        contact_flag: torch.Tensor,
        obj_pts: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prev_pose = finger_hist[:, -1, :]
        out = self.forward_train(frame_feat, obj_pts, target_pose, prev_pose=prev_pose, contact_flag=contact_flag)
        out["conf"] = out["quality_score"]  # backward compat alias
        return out

    @torch.no_grad()
    def infer_static(self, wrist_feat: torch.Tensor, obj_pts: torch.Tensor, k: int = 1) -> dict[str, torch.Tensor]:
        frame_feat = wrist_feat_to_frame_feat(wrist_feat)
        out = self.infer(frame_feat, obj_pts, k=k)
        out["candidate_conf"] = out.get("candidate_success_probs", out["candidate_quality_scores"])
        out["selected_conf"] = out.get("selected_success_prob", out["selected_quality_score"])
        return out

    @torch.no_grad()
    def infer_temporal(
        self,
        frame_feat: torch.Tensor,
        finger_hist: torch.Tensor,
        contact_flag: torch.Tensor,
        obj_pts: torch.Tensor,
        k: int = 1,
    ) -> dict[str, torch.Tensor]:
        prev_pose = finger_hist[:, -1, :]
        out = self.infer(frame_feat, obj_pts, prev_pose=prev_pose, contact_flag=contact_flag, k=k)
        out["candidate_conf"] = out.get("candidate_success_probs", out["candidate_quality_scores"])
        out["selected_conf"] = out.get("selected_success_prob", out["selected_quality_score"])
        return out


def wrist_feat_to_frame_feat(wrist_feat: torch.Tensor) -> torch.Tensor:
    """Adapts legacy wrist_feat(B,6) = [tsl(3), global_orient_aa(3)] into the unified
    frame_feat(B,1,13) interface (B1) so static (OakInk, T=1) batches flow through the
    same temporal encoder as HOT3D. rel_vel and dist are not available from wrist_feat
    alone so they are zero-filled (matches docs: OakInk rel_vel=0; dist needs the mesh
    pipeline and is computed upstream in dataset_oakink when available).

    WARNING: pos and rot6d here are in world/camera frame, NOT object-relative.
    This wrapper is for backward compatibility only. New code should use the
    object-relative frame_feat produced directly by OakInkStaticDataset.__getitem__.
    """
    batch = wrist_feat.shape[0]
    pos = wrist_feat[:, :3]
    aa = wrist_feat[:, 3:6]
    rot6d = axis_angle_to_rot6d(aa)
    rel_vel = torch.zeros_like(pos)
    dist = torch.zeros(batch, 1, device=wrist_feat.device, dtype=wrist_feat.dtype)
    frame = torch.cat([pos, rot6d, rel_vel, dist], dim=-1)
    return frame[:, None, :]


def axis_angle_to_rot6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """(...,3) axis-angle -> (...,6) 6D continuous rotation representation (first two columns of R)."""
    from model.mano_fk import axis_angle_to_matrix

    R = axis_angle_to_matrix(axis_angle)
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


# Backward-compatible short name used by scripts.
GraspModel = TemporalGeometryConditionedGraspModel


_LOWER = torch.tensor([
    -0.3, -0.4, -0.2, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1,
    -0.3, -0.3, -0.2, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1,
    -0.3, -0.4, -0.2, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1,
    -0.3, -0.5, -0.2, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1,
    -0.5, -0.3, -0.5, -0.2, -0.2, -0.2, -0.1, -0.1, -0.1,
], dtype=torch.float32)

_UPPER = torch.tensor([
    1.6, 0.4, 0.2, 1.7, 0.1, 0.1, 1.5, 0.1, 0.1,
    1.6, 0.3, 0.2, 1.7, 0.1, 0.1, 1.5, 0.1, 0.1,
    1.6, 0.4, 0.2, 1.7, 0.1, 0.1, 1.5, 0.1, 0.1,
    1.6, 0.5, 0.2, 1.7, 0.1, 0.1, 1.5, 0.1, 0.1,
    1.2, 0.3, 0.5, 0.8, 0.2, 0.2, 1.2, 0.1, 0.1,
], dtype=torch.float32)


def joint_limit_loss(pred: torch.Tensor) -> torch.Tensor:
    lower = _LOWER.to(pred.device)
    upper = _UPPER.to(pred.device)
    return (F.relu(lower - pred).pow(2) + F.relu(pred - upper).pow(2)).mean()


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def contact_penetration_loss(pred_pose: torch.Tensor, obj_pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """L_contact / L_penetration — eğitim sinyali, gerçek fizik ölçümü değil.

    L_contact: fingertip → nearest surface point hinge loss (TRAIN_CONTACT_HINGE_M).

    L_penetration: nearest-surface distance based. A fingertip is considered inside
    the object when it is closer to the nearest surface point than a small epsilon —
    i.e., `nearest_dist < 0`. Since obj_pts is a surface point cloud (not a solid),
    we approximate penetration as the fingertip being embedded past the nearest surface
    by using a sign proxy: if the distance to the nearest neighbor drops below half the
    average inter-point spacing, the fingertip is likely inside. In practice, for
    normalized point clouds at 1024 points, a 3mm threshold reliably distinguishes
    close-contact from penetration without requiring SDF computation.

    Bu proxy bir eğitim yönlendiricisidir — gerçek penetration ölçümü
    Unity PhysX'ten (Physics.ComputePenetration) gelir ve success_label'a dahildir.
    """
    tips = fingertip_positions(pred_pose)  # (B,5,3)
    dists = torch.cdist(tips, obj_pts)     # (B,5,N)
    nearest = dists.min(dim=-1).values     # (B,5)
    l_contact = F.relu(nearest - TRAIN_CONTACT_HINGE_M).mean()
    # Nearest-surface penetration: penalise fingertips too close to surface from inside.
    # Use a tight threshold (3mm) — below this the tip is likely penetrating.
    _PENETRATION_THRESHOLD_M = 0.003
    l_penetration = F.relu(_PENETRATION_THRESHOLD_M - nearest).mean()
    return l_contact, l_penetration


def grasp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    quality_score: torch.Tensor,
    obj_pts: torch.Tensor | None = None,
    quality_label: torch.Tensor | None = None,
    success_prob: torch.Tensor | None = None,
    success_label: torch.Tensor | None = None,
    prev_target_pose: torch.Tensor | None = None,
    prev_pred_pose: torch.Tensor | None = None,
    prev2_target_pose: torch.Tensor | None = None,
    prev2_pred_pose: torch.Tensor | None = None,
    kl_weight: float = 1e-2,
    limit_weight: float = 1.0,
    contact_weight: float = 0.3,
    penetration_weight: float = 0.1,  # proxy sinyal; gerçek ölçüm Unity'den
    quality_weight: float = 0.1,
    success_weight: float = 0.1,
    vel_weight: float = 0.0,
    acc_weight: float = 0.0,
    tip_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """B11 loss summary.

    Stage 1 (OakInk):  L = L_recon + kl*L_KL + limit*L_limit + contact*L_contact
                          + penetration*L_penetration + quality*L_quality + tip*L_tip
    Stage 2 (HOT3D):   Stage 1 + vel*L_vel + acc*L_acc   (vel/acc only meaningful for T>1)
    Stage 3 (calib):   success*L_success only (call with backbone frozen upstream)
    """
    rec = F.mse_loss(pred, target)
    kl = kl_loss(mu, logvar)
    limit = joint_limit_loss(pred)
    total = rec + kl_weight * kl + limit_weight * limit
    info = {
        "rec": float(rec.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "limit": float(limit.detach().cpu()),
    }

    if tip_weight > 0:
        tips_pred = fingertip_positions(pred)    # (B, 5, 3) wrist frame
        tips_gt   = fingertip_positions(target)  # (B, 5, 3) wrist frame
        l_tip = F.mse_loss(tips_pred, tips_gt)
        total = total + tip_weight * l_tip
        info["tip"] = float(l_tip.detach().cpu())

    if obj_pts is not None:
        l_contact, l_penetration = contact_penetration_loss(pred, obj_pts)
        total = total + contact_weight * l_contact + penetration_weight * l_penetration
        info["contact"] = float(l_contact.detach().cpu())
        info["penetration"] = float(l_penetration.detach().cpu())

    if quality_label is not None:
        l_quality = F.mse_loss(quality_score, quality_label)
        total = total + quality_weight * l_quality
        info["quality"] = float(l_quality.detach().cpu())

    if success_label is not None and success_prob is not None:
        l_success = F.binary_cross_entropy(success_prob, success_label)
        total = total + success_weight * l_success
        info["success"] = float(l_success.detach().cpu())

    if prev_target_pose is not None and prev_pred_pose is not None:
        # L_vel: predicted velocity must match ground-truth velocity (B8), not just be small.
        gt_vel = target - prev_target_pose
        pred_vel = pred - prev_pred_pose
        l_vel = F.mse_loss(pred_vel, gt_vel)
        total = total + vel_weight * l_vel
        info["vel"] = float(l_vel.detach().cpu())

        if prev2_target_pose is not None and prev2_pred_pose is not None:
            gt_acc = target - 2 * prev_target_pose + prev2_target_pose
            pred_acc = pred - 2 * prev_pred_pose + prev2_pred_pose
            l_acc = F.mse_loss(pred_acc, gt_acc)
            total = total + acc_weight * l_acc
            info["acc"] = float(l_acc.detach().cpu())

    info["total"] = float(total.detach().cpu())
    return total, info


if __name__ == "__main__":
    model = GraspModel()
    b = 2
    out = model.forward_static_train(
        torch.randn(b, WRIST_DIM),
        torch.randn(b, DEFAULT_N_POINTS, 3),
        torch.randn(b, FINGER_POSE_DIM),
    )
    print({k: tuple(v.shape) for k, v in out.items()})
    temp = model.forward_temporal_train(
        torch.randn(b, 16, HOT3D_FRAME_DIM),
        torch.randn(b, 16, FINGER_POSE_DIM),
        torch.zeros(b, 16, 1),
        torch.randn(b, DEFAULT_N_POINTS, 3),
        torch.randn(b, FINGER_POSE_DIM),
    )
    print({k: tuple(v.shape) for k, v in temp.items()})
