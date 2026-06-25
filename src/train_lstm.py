"""Train the AuraXR SDF-LSTM model.

SDF-LSTM training:
  - HOT3D input: temporal windows (T=16), each 29-dim (25 core + 4 SDF local)
  - ARCTIC/DexYCB input: single-frame contact-pose windows (T=1)
  - Object embedding: pre-computed 32-dim PCA embedding (lookup by BOP ID)
  - Loss: Huber(mano_pose_15) + 0.3×geodesic(wrist_rot) + 0.1×BCE(contact)
  - Multi-source: HOT3D (full dynamics) + ARCTIC + DexYCB (contact augmentation)

Run (single source, backward-compatible):
    .venv/bin/python3 src/train_lstm.py \\
        --data_dir data/processed/hot3d_mano/right/ \\
        --output_dir checkpoints/lstm_right/

Run (multi-source):
    .venv/bin/python3 src/train_lstm.py \\
        --hot3d_dir  data/processed/hot3d_mano/right \\
        --arctic_dir data/processed/arctic_mano/right \\
        --dexycb_dir data/processed/dexycb_mano/right \\
        --output_dir checkpoints/lstm_right/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

sys.path.insert(0, str(Path(__file__).parent))
from model import SDFLSTMModel, SDFTransformerModel

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_T      = 8    # ~0.27s at HOT3D 30fps; halves LSTM unroll for faster training
WINDOW_STRIDE = 4    # 75% overlap; improves gradient coverage near window edges
BATCH_SIZE    = 256
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
PATIENCE      = 20

POSE_LOSS_W    = 1.0
WRIST_LOSS_W   = 0.3
CONTACT_LOSS_W = 0.1
CONTACT_FRAME_EXTRA_W = 3.0
POSE_REG_W     = 0.01
FINGERTIP_LOSS_W = 0.05

WRIST_DIMS = slice(11, 17)
WRIST_OBJ_DIMS = slice(25, 28)   # raw wrist-in-object-frame position (metres)
SS_START_EPOCH = 10
SS_END_EPOCH = 80
SS_MAX_PROB = 0.50

# Canonical fingertip offsets from wrist in hand-local frame (metres, right-hand).
# Derived from MANO zero-pose FK at canonical shape. Used to approximate fingertip
# positions without running smplx during training.
_FINGERTIP_OFFSETS = torch.tensor([
    [ 0.055, -0.018,  0.025],   # thumb
    [ 0.082,  0.008,  0.002],   # index
    [ 0.086,  0.025,  0.000],   # middle
    [ 0.078,  0.038,  0.000],   # ring
    [ 0.063,  0.048, -0.002],   # pinky
], dtype=torch.float32)  # (5, 3)


@dataclass(frozen=True)
class SourceConfig:
    name: str
    window_t: int
    train_fraction: float
    loss_weight: float
    contact_loss_w: float


SOURCE_CONFIGS = {
    # HOT3D is the only source with real approach dynamics.
    "hot3d":  SourceConfig("hot3d",  WINDOW_T, 0.70, 1.00, CONTACT_LOSS_W),
    # ARCTIC/DexYCB frames are all contact=1, so contact BCE teaches grasp pose.
    # Lower weight than HOT3D because these are static single-frame snapshots.
    "arctic": SourceConfig("arctic", 1,        0.25, 0.25, 0.05),
    "dexycb": SourceConfig("dexycb", 1,        0.05, 0.50, 0.10),
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TemporalWindowDataset(Dataset):
    """Sliding windows over temporal sequences from dataset_mano.h5.

    norm_stats: if provided externally (e.g. HOT3D master stats), those are used
    instead of the file's own meta. Allows multi-source training on a common scale.
    """

    def __init__(
        self,
        h5_path: Path,
        split: str,
        embed_matrix: np.ndarray,   # (N_obj, 32) SDF embeddings indexed by bop_ids
        bop_ids: np.ndarray,         # (N_obj,) sorted BOP IDs
        window_t: int = WINDOW_T,
        stride: int = WINDOW_STRIDE,
        norm_stats: dict | None = None,
    ):
        super().__init__()
        self.window_t = window_t
        self.embed_matrix = torch.from_numpy(embed_matrix.astype(np.float32))
        self.bop_id_to_idx = {int(bid): i for i, bid in enumerate(bop_ids)}
        self.norm_stats = norm_stats

        with h5py.File(h5_path, "r") as f:
            g = f[split]
            self.features    = torch.from_numpy(g["features"][:])
            self.raw_features = self.features.clone()
            self.sdf_feats   = torch.from_numpy(g["sdf_features"][:])
            self.obj_ids     = g["obj_id"][:]
            self.targets     = torch.from_numpy(g["targets"][:])
            self.wrist_rot   = torch.from_numpy(g["wrist_rot_6d"][:])
            self.seq_ids     = g["sequence_id"][:]
            self.frame_idx   = g["frame_index"][:]
            self.is_mirror   = g["is_mirror"][:] if "is_mirror" in g else np.zeros_like(self.seq_ids)
            contact_name     = "contact_v2" if "contact_v2" in g else "contact"
            self.contact     = torch.from_numpy(g[contact_name][:]).float()

            # Use provided norm_stats or fall back to file's embedded meta
            if norm_stats is None:
                if "meta" not in f.attrs:
                    raise KeyError(
                        f"{h5_path} has no 'meta' attribute. "
                        "Re-run the builder to regenerate, or pass norm_stats explicitly."
                    )
                norm_stats = json.loads(f.attrs["meta"])
            self.norm_stats = norm_stats

        feat_mean = torch.tensor(norm_stats["feature_mean"], dtype=torch.float32)
        feat_std  = torch.tensor(norm_stats["feature_std"],  dtype=torch.float32)
        sdf_mean  = torch.tensor(norm_stats["sdf_mean"],     dtype=torch.float32)
        sdf_std   = torch.tensor(norm_stats["sdf_std"],      dtype=torch.float32)
        tgt_mean  = torch.tensor(norm_stats["target_mean"],  dtype=torch.float32)
        tgt_std   = torch.tensor(norm_stats["target_std"],   dtype=torch.float32)

        if self.features.shape[1] < feat_mean.shape[0]:
            pad = feat_mean.shape[0] - self.features.shape[1]
            self.features = F.pad(self.features, (0, pad))
            self.raw_features = F.pad(self.raw_features, (0, pad))
        elif self.features.shape[1] > feat_mean.shape[0]:
            raise ValueError(
                f"{h5_path} feature dim {self.features.shape[1]} exceeds "
                f"normalization dim {feat_mean.shape[0]}"
            )

        self.features  = (self.features  - feat_mean) / feat_std
        self.sdf_feats = (self.sdf_feats - sdf_mean)  / sdf_std
        self.targets   = (self.targets   - tgt_mean)  / tgt_std
        self.tgt_mean = tgt_mean
        self.tgt_std  = tgt_std
        self.wrist_input_mean = feat_mean[WRIST_DIMS]
        self.wrist_input_std = feat_std[WRIST_DIMS]
        self.feat_dim = int(self.features.shape[1] + self.sdf_feats.shape[1])

        # Build sliding window index grouped by (sequence_id, is_mirror)
        self._seq_windows: list[list[int]] = []
        seq_to_frames: dict[tuple[int, int], list[int]] = {}
        for i in range(len(self.seq_ids)):
            key = (int(self.seq_ids[i]), int(self.is_mirror[i]))
            seq_to_frames.setdefault(key, []).append(i)

        for key, positions in seq_to_frames.items():
            positions.sort(key=lambda i: self.frame_idx[i])
            L = len(positions)
            for start in range(0, L - window_t + 1, stride):
                self._seq_windows.append(positions[start:start + window_t])

    def __len__(self):
        return len(self._seq_windows)

    def __getitem__(self, idx: int):
        positions = self._seq_windows[idx]
        feat_seq  = self.features[positions]
        sdf_seq   = self.sdf_feats[positions]
        inp_seq   = torch.cat([feat_seq, sdf_seq], dim=-1)
        tgt_seq   = self.targets[positions]
        wrist_seq = self.wrist_rot[positions]
        cont_seq  = self.contact[positions]

        bop_id    = int(self.obj_ids[positions[-1]])
        embed_idx = self.bop_id_to_idx.get(bop_id, 0)
        obj_emb   = self.embed_matrix[embed_idx]              # (32,)
        obj_id_t  = torch.tensor(bop_id, dtype=torch.long)

        # Raw (unnormalized) wrist-in-object-frame position for fingertip SDF loss.
        if self.raw_features.shape[1] > WRIST_OBJ_DIMS.stop:
            raw_wrist_obj = self.raw_features[positions, WRIST_OBJ_DIMS]  # (T, 3) metres
        else:
            raw_wrist_obj = torch.zeros(len(positions), 3)

        return inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq, obj_id_t, raw_wrist_obj


def load_norm_stats(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return json.loads(f.attrs["meta"])


def build_multi_source_datasets(
    args,
    embed_matrix: np.ndarray,
    bop_ids: np.ndarray,
) -> tuple[Dataset, list[tuple[str, Dataset]]]:
    """Build train and validation datasets for HOT3D plus optional sources."""
    train_sources: list[tuple[SourceConfig, Dataset]] = []
    val_sources: list[tuple[str, Dataset]] = []

    hot3d_h5 = Path(args.hot3d_dir) / "dataset_mano.h5"
    if not hot3d_h5.exists():
        raise FileNotFoundError(f"HOT3D h5 not found: {hot3d_h5}")

    master_norm = load_norm_stats(hot3d_h5)

    cfg = SOURCE_CONFIGS["hot3d"]
    hot3d_train = TemporalWindowDataset(hot3d_h5, "train", embed_matrix, bop_ids,
                                         window_t=cfg.window_t, stride=WINDOW_STRIDE,
                                         norm_stats=master_norm)
    hot3d_val   = TemporalWindowDataset(hot3d_h5, "val",   embed_matrix, bop_ids,
                                         window_t=cfg.window_t, stride=WINDOW_STRIDE,
                                         norm_stats=master_norm)
    train_sources.append((cfg, hot3d_train))
    val_sources.append(("hot3d", hot3d_val))
    print(f"HOT3D  train={len(hot3d_train)} windows  val={len(hot3d_val)} windows")

    if getattr(args, "arctic_dir", None):
        arctic_h5 = Path(args.arctic_dir) / "dataset_mano.h5"
        if arctic_h5.exists():
            cfg = SOURCE_CONFIGS["arctic"]
            arctic_train = TemporalWindowDataset(arctic_h5, "train", embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            arctic_val   = TemporalWindowDataset(arctic_h5, "val",   embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            train_sources.append((cfg, arctic_train))
            val_sources.append(("arctic", arctic_val))
            print(f"ARCTIC train={len(arctic_train)} windows  val={len(arctic_val)} windows")
        else:
            print(f"[SKIP] ARCTIC h5 not found: {arctic_h5}")

    if getattr(args, "dexycb_dir", None):
        dexycb_h5 = Path(args.dexycb_dir) / "dataset_mano.h5"
        if dexycb_h5.exists():
            cfg = SOURCE_CONFIGS["dexycb"]
            dexycb_train = TemporalWindowDataset(dexycb_h5, "train", embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            dexycb_val   = TemporalWindowDataset(dexycb_h5, "val",   embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            train_sources.append((cfg, dexycb_train))
            val_sources.append(("dexycb", dexycb_val))
            print(f"DexYCB train={len(dexycb_train)} windows  val={len(dexycb_val)} windows")
        else:
            print(f"[SKIP] DexYCB h5 not found: {dexycb_h5}")

    return train_sources, val_sources


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to a rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1,
                     dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_angle_rad(pred_6d: torch.Tensor, tgt_6d: torch.Tensor) -> torch.Tensor:
    """Per-sample angular error in radians. Input shape: (..., 6)."""
    r_pred = rot6d_to_matrix(pred_6d)
    r_tgt = rot6d_to_matrix(tgt_6d)
    r_diff = r_pred.transpose(-2, -1) @ r_tgt
    trace = r_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def geodesic_loss(pred_6d: torch.Tensor, tgt_6d: torch.Tensor) -> torch.Tensor:
    return rotation_angle_rad(pred_6d, tgt_6d).mean()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights / weights.mean().clamp_min(1e-6)
    return (values * weights).mean()


def fingertip_sdf_loss(
    pred_wrist_6d: torch.Tensor,     # (B, T, 6)
    raw_wrist_obj: torch.Tensor,     # (B, T, 3) metres, unnormalized
    contact_label: torch.Tensor,     # (B, T)
    obj_ids: torch.Tensor,           # (B,) BOP IDs
    sdf_grids: dict[int, torch.Tensor],   # bop_id → (1, R, R, R) on device
    sdf_bounds: dict[int, torch.Tensor],  # bop_id → (2, 3) on device
) -> torch.Tensor:
    """Penalize fingertip distance from object surface on contact frames.

    Uses differentiable trilinear SDF query (F.grid_sample) so gradients flow
    back to pred_wrist_6d, pushing the wrist rotation to orient fingers toward
    the object surface.
    """
    B, T, _ = pred_wrist_6d.shape
    device = pred_wrist_6d.device
    contact_mask = contact_label > 0.5   # (B, T)
    if not contact_mask.any():
        return pred_wrist_6d.sum() * 0.0
    if raw_wrist_obj.abs().max() < 1e-6:
        return pred_wrist_6d.sum() * 0.0

    offs = _FINGERTIP_OFFSETS.to(device)   # (5, 3)
    # Wrist rotation matrices: (B, T, 3, 3)
    rot = rot6d_to_matrix(pred_wrist_6d.reshape(B * T, 6)).reshape(B, T, 3, 3)
    # rot @ offs.T: (B, T, 3, 3) @ (3, 5) → (B, T, 3, 5) → permute → (B, T, 5, 3)
    tips_local = (rot @ offs.T).permute(0, 1, 3, 2)
    tips_obj = raw_wrist_obj.unsqueeze(2) + tips_local   # (B, T, 5, 3)

    total = pred_wrist_6d.sum() * 0.0
    count = 0
    for b in range(B):
        bid = int(obj_ids[b].item())
        if bid not in sdf_grids:
            continue
        ct = contact_mask[b]   # (T,)
        if not ct.any():
            continue
        # Skip if wrist_in_obj is all zeros (dataset didn't provide it)
        if raw_wrist_obj[b].abs().max() < 1e-6:
            continue

        grid   = sdf_grids[bid]    # (1, 1, R, R, R)
        bounds = sdf_bounds[bid]   # (2, 3)
        lo, hi = bounds[0], bounds[1]

        tips = tips_obj[b][ct]     # (n_ct, 5, 3)
        n_ct = tips.shape[0]

        # Normalise positions to [-1, 1] for F.grid_sample
        gc = 2.0 * (tips - lo) / (hi - lo + 1e-6) - 1.0   # (n_ct, 5, 3)
        # grid_sample expects (N, D_out, H_out, W_out, 3) for 5D input
        gc = gc.reshape(1, 1, n_ct * 5, 1, 3)
        sdf_vals = F.grid_sample(
            grid, gc, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, 1, 1, n_ct*5, 1) → squeeze
        sdf_vals = sdf_vals.reshape(-1)   # (n_ct*5,)

        total = total + F.relu(sdf_vals).mean()
        count += 1

    return total / max(count, 1)


def normalize_wrist_for_input(
    wrist_6d: torch.Tensor,
    wrist_input_mean: torch.Tensor,
    wrist_input_std: torch.Tensor,
) -> torch.Tensor:
    return (wrist_6d - wrist_input_mean) / wrist_input_std.clamp_min(1e-6)


def forward_sequence_scheduled_sampling(
    model,
    inp_seq: torch.Tensor,
    obj_emb: torch.Tensor,
    ss_prob: float,
    wrist_input_mean: torch.Tensor,
    wrist_input_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frame-by-frame training forward with scheduled wrist input replacement."""
    B, T, _ = inp_seq.shape
    h, c = model.initial_state(B, device=inp_seq.device)
    poses, wrists, contacts = [], [], []
    prev_wrist = None

    for t in range(T):
        frame = inp_seq[:, t, :].clone()
        if prev_wrist is not None and ss_prob > 0.0:
            mask = torch.rand(B, device=frame.device) < ss_prob
            if mask.any():
                prev_in = normalize_wrist_for_input(
                    prev_wrist.detach(), wrist_input_mean, wrist_input_std)
                frame[mask, WRIST_DIMS] = prev_in[mask]

        pose, wrist, contact, h, c = model(frame, obj_emb, h, c)
        poses.append(pose)
        wrists.append(wrist)
        contacts.append(contact)
        prev_wrist = wrist

    return (
        torch.stack(poses, dim=1),
        torch.stack(wrists, dim=1),
        torch.stack(contacts, dim=1),
    )


def forward_sequence_autoregressive(
    model,
    inp_seq: torch.Tensor,
    obj_emb: torch.Tensor,
    wrist_input_mean: torch.Tensor,
    wrist_input_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inference-like validation: feed previous predicted wrist into next frame."""
    return forward_sequence_scheduled_sampling(
        model, inp_seq, obj_emb, 1.0, wrist_input_mean, wrist_input_std)


def get_ss_prob(epoch: int) -> float:
    if epoch <= SS_START_EPOCH:
        return 0.0
    span = max(1, SS_END_EPOCH - SS_START_EPOCH)
    progress = min(epoch - SS_START_EPOCH, span)
    return SS_MAX_PROB * progress / span


def temporal_loss(
    pred_pose,
    pred_wrist,
    pred_contact,
    tgt_pose,
    tgt_wrist,
    contact_label,
    contact_loss_w: float = CONTACT_LOSS_W,
    total_weight: float = 1.0,
    contact_pos_weight: float = 1.0,
):
    T = contact_label.shape[-1]
    # Late frames (closer to the grasp moment) are weighted up to 2x more.
    time_weight = torch.linspace(1.0, 2.0, steps=T, device=contact_label.device)
    frame_weight = (1.0 + CONTACT_FRAME_EXTRA_W * contact_label) * time_weight

    pose_frame = F.huber_loss(pred_pose, tgt_pose, delta=0.5, reduction="none").mean(-1)
    pose_loss = weighted_mean(pose_frame, frame_weight)

    wrist_angle = rotation_angle_rad(pred_wrist, tgt_wrist)
    wrist_loss = weighted_mean(wrist_angle, frame_weight)

    if contact_loss_w > 0:
        cw = torch.where(
            contact_label > 0.5,
            torch.full_like(contact_label, contact_pos_weight),
            torch.ones_like(contact_label),
        )
        contact_loss = F.binary_cross_entropy(
            pred_contact.squeeze(-1).clamp(1e-6, 1 - 1e-6), contact_label, weight=cw)
    else:
        contact_loss = pred_contact.sum() * 0.0

    # Penalise PCA coefficients outside ±3 to prevent anatomically invalid poses.
    pose_reg_loss = F.relu(pred_pose.abs() - 3.0).mean()

    total = (POSE_LOSS_W * pose_loss
           + WRIST_LOSS_W * wrist_loss
           + contact_loss_w * contact_loss
           + POSE_REG_W * pose_reg_loss)
    total = total * total_weight
    return total, pose_loss.detach(), wrist_loss.detach(), contact_loss.detach()


def evaluate_temporal(
    model,
    loader,
    device,
    contact_loss_w: float = CONTACT_LOSS_W,
    autoregressive: bool = False,
    wrist_input_mean: torch.Tensor | None = None,
    wrist_input_std: torch.Tensor | None = None,
    contact_pos_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total_loss = total_pose_err = total_wrist_deg = total_final_wrist_deg = 0.0
    total_contact_wrist_deg = total_contact_weight = 0.0
    total_jitter_deg = total_contact_bce = n = 0
    contact_recall_num = contact_recall_den = 0
    with torch.no_grad():
        for batch in loader:
            inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq = batch[:5]
            inp_seq   = inp_seq.to(device)
            obj_emb   = obj_emb.to(device)
            tgt_seq   = tgt_seq.to(device)
            wrist_seq = wrist_seq.to(device)
            cont_seq  = cont_seq.to(device)
            if autoregressive:
                if wrist_input_mean is None or wrist_input_std is None:
                    raise ValueError("autoregressive evaluation requires wrist input normalization stats")
                pp, pw, pc = forward_sequence_autoregressive(
                    model, inp_seq, obj_emb, wrist_input_mean, wrist_input_std)
            else:
                pp, pw, pc = model.forward_sequence(inp_seq, obj_emb)
            loss, *_ = temporal_loss(pp, pw, pc, tgt_seq, wrist_seq, cont_seq,
                                     contact_loss_w=contact_loss_w,
                                     contact_pos_weight=contact_pos_weight)
            B = inp_seq.shape[0]
            wrist_rad = rotation_angle_rad(pw, wrist_seq)
            final_wrist_rad = rotation_angle_rad(pw[:, -1, :], wrist_seq[:, -1, :])
            contact_mask = cont_seq > 0.5
            if contact_mask.any():
                contact_wrist_deg = torch.rad2deg(wrist_rad[contact_mask]).sum()
                contact_count = int(contact_mask.sum().item())
            else:
                contact_wrist_deg = pw.sum() * 0.0
                contact_count = 0
            if pw.shape[1] > 1:
                jitter_rad = rotation_angle_rad(pw[:, 1:, :], pw[:, :-1, :]).mean()
            else:
                jitter_rad = pw.sum() * 0.0
            if contact_loss_w > 0:
                cw = torch.where(
                    cont_seq > 0.5,
                    torch.full_like(cont_seq, contact_pos_weight),
                    torch.ones_like(cont_seq),
                )
                contact_bce = F.binary_cross_entropy(pc.squeeze(-1), cont_seq, weight=cw)
            else:
                contact_bce = pc.sum() * 0.0

            pred_contact_pos = pc.squeeze(-1) > 0.5
            gt_contact_pos = cont_seq > 0.5
            contact_recall_num += int((pred_contact_pos & gt_contact_pos).sum().item())
            contact_recall_den += int(gt_contact_pos.sum().item())

            total_loss += loss.item() * B
            total_pose_err += (pp[:, -1, :] - tgt_seq[:, -1, :]).abs().mean().item() * B
            total_wrist_deg += torch.rad2deg(wrist_rad).mean().item() * B
            total_final_wrist_deg += torch.rad2deg(final_wrist_rad).mean().item() * B
            total_contact_wrist_deg += contact_wrist_deg.item()
            total_contact_weight += contact_count
            total_jitter_deg += torch.rad2deg(jitter_rad).item() * B
            total_contact_bce += contact_bce.item() * B
            n += B
    return {
        "loss": total_loss / n,
        "pose_l1_final": total_pose_err / n,
        "wrist_deg": total_wrist_deg / n,
        "final_wrist_deg": total_final_wrist_deg / n,
        "contact_wrist_deg": (
            total_contact_wrist_deg / total_contact_weight
            if total_contact_weight > 0 else float("nan")
        ),
        "jitter_deg": total_jitter_deg / n,
        "contact_bce": total_contact_bce / n,
        "contact_recall": contact_recall_num / max(contact_recall_den, 1),
    }


def _next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def build_source_schedule(train_loaders: dict[str, DataLoader]) -> list[str]:
    """Create one epoch schedule with HOT3D as the anchor and aux sources sampled by ratio."""
    if "hot3d" not in train_loaders:
        return [name for name, loader in train_loaders.items() for _ in range(len(loader))]

    hot3d_steps = len(train_loaders["hot3d"])
    hot3d_fraction = SOURCE_CONFIGS["hot3d"].train_fraction
    schedule = ["hot3d"] * hot3d_steps

    for name, loader in train_loaders.items():
        if name == "hot3d":
            continue
        cfg = SOURCE_CONFIGS[name]
        # Small contact-pose datasets are intentionally oversampled by cycling
        # their loader, so their configured fraction is preserved.
        steps = int(round(hot3d_steps * cfg.train_fraction / hot3d_fraction))
        schedule.extend([name] * steps)

    rng = np.random.default_rng()
    rng.shuffle(schedule)
    return schedule


def compute_sequence_yaw_bins(dataset: TemporalWindowDataset, n_bins: int = 8) -> list[int]:
    bins: list[int] = []
    for positions in dataset._seq_windows:
        dir_world = dataset.raw_features[positions, 0:3].float()
        mean_dir = F.normalize(dir_world.mean(0), dim=0, eps=1e-6)
        yaw = torch.atan2(mean_dir[2], mean_dir[0])
        bin_idx = int((yaw + torch.pi) / (2 * torch.pi) * n_bins) % n_bins
        bins.append(bin_idx)
    return bins


def build_yaw_balanced_sampler(dataset: TemporalWindowDataset, n_bins: int = 8) -> WeightedRandomSampler:
    bins = torch.tensor(compute_sequence_yaw_bins(dataset, n_bins), dtype=torch.long)
    counts = torch.bincount(bins, minlength=n_bins).float().clamp_min(1)
    weights = (1.0 / counts)[bins]
    weights = weights / weights.mean()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loops
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    epochs     = getattr(args, "epochs",     EPOCHS)
    patience   = getattr(args, "patience",   PATIENCE)
    batch_size = getattr(args, "batch_size", BATCH_SIZE)
    print(f"Training SDF-LSTM on {device}")

    embed_matrix = np.load("data/models/sdf_grids/sdf_embed_matrix.npy")
    bop_ids      = np.load("data/models/sdf_grids/sdf_bop_ids.npy")

    use_multi = bool(getattr(args, "hot3d_dir", None))
    if use_multi:
        train_sources, val_sources = build_multi_source_datasets(args, embed_matrix, bop_ids)
        print("Train sources: " + ", ".join(
            f"{cfg.name}=windows:{len(ds)} T:{cfg.window_t} loss_w:{cfg.loss_weight} contact_w:{cfg.contact_loss_w}"
            for cfg, ds in train_sources
        ))
    else:
        h5_path = Path(args.data_dir) / "dataset_mano.h5"
        train_ds = TemporalWindowDataset(h5_path, "train", embed_matrix, bop_ids)
        val_ds   = TemporalWindowDataset(h5_path, "val",   embed_matrix, bop_ids)
        val_sources = [("val", val_ds)]
        print(f"Train: {len(train_ds)} windows  Val: {len(val_ds)} windows")

    if use_multi:
        train_loaders = {}
        for cfg, ds in train_sources:
            sampler = build_yaw_balanced_sampler(ds) if (
                getattr(args, "balanced_yaw_sampling", False) and cfg.name == "hot3d"
            ) else None
            train_loaders[cfg.name] = DataLoader(
                ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                num_workers=4, persistent_workers=True, pin_memory=False)
        source_cfgs = {cfg.name: cfg for cfg, _ in train_sources}
        train_loader = None
    else:
        sampler = build_yaw_balanced_sampler(train_ds) if getattr(args, "balanced_yaw_sampling", False) else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                                  sampler=sampler, num_workers=6,
                                  persistent_workers=True, pin_memory=False)
        train_loaders = {}
        source_cfgs = {}
    val_loaders  = [(name, DataLoader(ds, batch_size=batch_size, shuffle=False,
                                      num_workers=2, persistent_workers=True))
                    for name, ds in val_sources]

    model_feat_dim = (train_sources[0][1] if use_multi else train_ds).feat_dim
    model_type = getattr(args, "model_type", "lstm")
    oa_sdf = getattr(args, "orientation_aware_sdf", False)
    if model_type == "transformer":
        model = SDFTransformerModel(
            feat_dim=model_feat_dim,
            orientation_aware_sdf=oa_sdf,
        ).to(device)
    else:
        model = SDFLSTMModel(
            feat_dim=model_feat_dim,
            orientation_aware_sdf=oa_sdf,
        ).to(device)
    print(f"{model.__class__.__name__} params: {model.count_params():,}")

    stats_ds = train_sources[0][1] if use_multi else train_ds
    wrist_input_mean = stats_ds.wrist_input_mean.to(device)
    wrist_input_std = stats_ds.wrist_input_std.to(device)

    # Compute contact pos_weight from HOT3D training data to address class imbalance.
    contact_pos = float(stats_ds.contact.sum().item())
    contact_neg = float(len(stats_ds.contact)) - contact_pos
    contact_pos_weight = min(contact_neg / max(contact_pos, 1.0), 20.0)
    print(f"Contact pos_weight: {contact_pos_weight:.1f}  "
          f"(pos={int(contact_pos)}, neg={int(contact_neg)})")

    # Load SDF grids for fingertip proximity loss.
    sdf_grid_dir = Path("data/models/sdf_grids")
    sdf_grids_cpu: dict[int, torch.Tensor] = {}
    sdf_bounds_cpu: dict[int, torch.Tensor] = {}
    for path in sorted(sdf_grid_dir.glob("bop*.npz")):
        bid = int(path.stem[3:])
        d = np.load(str(path))
        sdf_grids_cpu[bid] = torch.from_numpy(d["grid"]).unsqueeze(0).unsqueeze(0).float()  # (1,1,R,R,R)
        sdf_bounds_cpu[bid] = torch.from_numpy(d["bounds"]).float()  # (2,3)
    sdf_grids_dev  = {k: v.to(device) for k, v in sdf_grids_cpu.items()}
    sdf_bounds_dev = {k: v.to(device) for k, v in sdf_bounds_cpu.items()}
    print(f"Loaded {len(sdf_grids_dev)} SDF grids for fingertip loss")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        print("  torch.compile: skipped (MPS fallback)")

    lr = getattr(args, "lr", LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=5e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=5)

    best_metric = float("inf")
    patience_counter = 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train.log"
    logger = setup_logger(log_path)
    logger.info(f"=== Training start  epochs={epochs}  batch={batch_size}  lr={lr}  device={device} ===")
    if use_multi:
        src_info = ", ".join(f"{cfg.name}({len(ds)} windows)"
                             for cfg, ds in train_sources)
    else:
        src_info = f"single({len(train_ds)} windows)"
    logger.info(f"Sources: {src_info}")
    print(f"\nLog file: {log_path.resolve()}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = n = 0
        total_pose_loss = total_wrist_loss = total_contact_loss = total_ft_loss = 0.0
        step = 0
        epoch_start = time.time()
        ss_prob = get_ss_prob(epoch)
        if use_multi:
            schedule = build_source_schedule(train_loaders)
            loader_iters = {name: iter(loader) for name, loader in train_loaders.items()}
            batch_iter = tqdm(schedule, desc=f"Epoch {epoch}/{epochs}", leave=False)
        else:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        source_seen: dict[str, int] = {}
        for item in batch_iter:
            if use_multi:
                source_name = item
                batch, loader_iters[source_name] = _next_batch(
                    loader_iters[source_name], train_loaders[source_name])
                inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq, obj_id, raw_wrist_obj = batch
                cfg = source_cfgs[source_name]
                source_seen[source_name] = source_seen.get(source_name, 0) + inp_seq.shape[0]
            else:
                source_name = "single"
                cfg = SourceConfig("single", WINDOW_T, 1.0, 1.0, CONTACT_LOSS_W)
                inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq, obj_id, raw_wrist_obj = item

            inp_seq       = inp_seq.to(device, non_blocking=True)
            obj_emb       = obj_emb.to(device, non_blocking=True)
            tgt_seq       = tgt_seq.to(device, non_blocking=True)
            wrist_seq     = wrist_seq.to(device, non_blocking=True)
            cont_seq      = cont_seq.to(device, non_blocking=True)
            obj_id        = obj_id.to(device, non_blocking=True)
            raw_wrist_obj = raw_wrist_obj.to(device, non_blocking=True)

            noise_scale = float(getattr(args, "input_noise", 0.0))
            if noise_scale > 0.0:
                continuous_dims = [i for i in range(min(11, inp_seq.shape[-1]))]
                inp_seq[:, :, continuous_dims] += noise_scale * torch.randn_like(inp_seq[:, :, continuous_dims])
                inp_seq[:, :, WRIST_DIMS] += (noise_scale * 0.25) * torch.randn_like(inp_seq[:, :, WRIST_DIMS])

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                if ss_prob > 0.0:
                    pj, pw, pc = forward_sequence_scheduled_sampling(
                        model, inp_seq, obj_emb, ss_prob, wrist_input_mean, wrist_input_std)
                else:
                    pj, pw, pc = model.forward_sequence(inp_seq, obj_emb)

            pj = pj.float(); pw = pw.float(); pc = pc.float()
            loss, pose_l, wrist_l, contact_l = temporal_loss(
                pj, pw, pc, tgt_seq, wrist_seq, cont_seq,
                contact_loss_w=cfg.contact_loss_w,
                total_weight=cfg.loss_weight,
                contact_pos_weight=contact_pos_weight)
            ft_loss = fingertip_sdf_loss(
                pw, raw_wrist_obj, cont_seq, obj_id, sdf_grids_dev, sdf_bounds_dev)
            loss = loss + FINGERTIP_LOSS_W * ft_loss * cfg.loss_weight

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            B_size = inp_seq.shape[0]
            total_loss         += loss.item()    * B_size
            total_pose_loss    += pose_l.item()  * B_size
            total_wrist_loss   += wrist_l.item() * B_size
            total_contact_loss += contact_l.item()* B_size
            total_ft_loss      += ft_loss.item() * B_size
            n    += B_size
            step += 1

            if step % 50 == 0:
                logger.info(
                    f"E{epoch:03d} step {step:04d} | "
                    f"src={source_name if use_multi else 'single'}  "
                    f"total={loss.item():.4f}  "
                    f"pose={pose_l.item():.4f}  "
                    f"wrist={wrist_l.item():.4f}  "
                    f"contact={contact_l.item():.4f}  "
                    f"fingertip={ft_loss.item():.4f}  "
                    f"ss={ss_prob:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        train_loss    = total_loss         / n
        epoch_pose    = total_pose_loss    / n
        epoch_wrist   = total_wrist_loss   / n
        epoch_contact = total_contact_loss / n
        epoch_ft      = total_ft_loss      / n
        epoch_time    = time.time() - epoch_start

        logger.info(
            f"E{epoch:03d} EPOCH_TRAIN | "
            f"total={train_loss:.4f}  pose={epoch_pose:.4f}  "
            f"wrist={epoch_wrist:.4f}  contact={epoch_contact:.4f}  "
            f"fingertip={epoch_ft:.4f}  "
            f"ss={ss_prob:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={epoch_time:.0f}s"
        )

        # Evaluate each val source separately (skip on non-val epochs)
        val_every = getattr(args, "val_every", 2)
        if epoch % val_every != 0 and epoch != epochs:
            print(f"Epoch {epoch:4d} | train={train_loss:.4f}  [val skipped]  "
                  f"ss={ss_prob:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            if epoch <= 5:
                warmup_scheduler.step()
            continue

        val_results = {}
        for name, vloader in val_loaders:
            contact_w = SOURCE_CONFIGS.get(name, SOURCE_CONFIGS["hot3d"]).contact_loss_w
            tf_metrics = evaluate_temporal(model, vloader, device, contact_loss_w=contact_w,
                                           contact_pos_weight=contact_pos_weight)
            ar_metrics = evaluate_temporal(
                model, vloader, device, contact_loss_w=contact_w, autoregressive=True,
                wrist_input_mean=wrist_input_mean, wrist_input_std=wrist_input_std,
                contact_pos_weight=contact_pos_weight)
            val_results[name] = {"tf": tf_metrics, "ar": ar_metrics}

        # Primary metric: HOT3D val loss (or first source if HOT3D not named)
        primary_name = "hot3d" if "hot3d" in val_results else list(val_results.keys())[0]
        primary_loss = val_results[primary_name]["ar"]["loss"]

        # Runtime-aware checkpoint score: final wrist error + contact-frame wrist error.
        # Lower is better. Falls back to 2x final_wrist_deg when no contact frames exist.
        primary_ar = val_results[primary_name]["ar"]
        cw_deg = primary_ar["contact_wrist_deg"]
        if not np.isfinite(cw_deg):
            cw_deg = primary_ar["final_wrist_deg"]
        primary_score = primary_ar["final_wrist_deg"] + 0.5 * cw_deg

        if epoch <= 5:
            warmup_scheduler.step()
        else:
            scheduler.step(primary_loss)

        val_summary = "  ".join(
            f"{name}_tf_loss={m['tf']['loss']:.4f} {name}_ar_loss={m['ar']['loss']:.4f} "
            f"{name}_ar_pose={m['ar']['pose_l1_final']:.4f} "
            f"{name}_ar_wrist={m['ar']['wrist_deg']:.1f}° "
            f"{name}_ar_final={m['ar']['final_wrist_deg']:.1f}° "
            f"{name}_ar_contact_wrist={m['ar']['contact_wrist_deg']:.1f}° "
            f"{name}_ar_jitter={m['ar']['jitter_deg']:.1f}° "
            f"{name}_ar_contact_recall={m['ar']['contact_recall']:.2f} "
            for name, m in val_results.items()
        )
        print(f"Epoch {epoch:4d} | train={train_loss:.4f}  {val_summary}  "
              f"ss={ss_prob:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        if use_multi:
            print("  source_samples=" + ", ".join(
                f"{k}:{v}" for k, v in sorted(source_seen.items())
            ))

        logger.info(f"E{epoch:03d} EPOCH_VAL  | {val_summary.strip()}")

        if primary_score < best_metric:
            best_metric = primary_score
            patience_counter = 0
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"epoch": epoch, "model": raw.state_dict(),
                        "val_results": val_results,
                        "ss_prob": ss_prob,
                        "best_metric": best_metric,
                        "best_metric_name": f"{primary_name}_ar_final_wrist_deg+contact_wrist_deg"},
                       out_dir / "best.pt")
            msg = (f"✓ NEW BEST  score={best_metric:.2f}°  "
                   f"final_wrist={primary_ar['final_wrist_deg']:.1f}°  "
                   f"contact_wrist={cw_deg:.1f}°  "
                   f"contact_recall={primary_ar['contact_recall']:.2f}")
            print(f"  {msg}")
            logger.info(f"E{epoch:03d} {msg}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                msg = f"Early stopping at epoch {epoch} (patience={patience})"
                print(msg)
                logger.info(msg)
                break

    msg = f"Training complete. Best {primary_name} score: {best_metric:.2f}°"
    print(msg)
    logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=Path, default=None,
                   help="Path to directory containing dataset_mano.h5 (single-source mode)")
    p.add_argument("--hot3d_dir",  type=Path, default=None,
                   help="HOT3D processed dir (multi-source mode, provides master norm stats)")
    p.add_argument("--arctic_dir", type=Path, default=None,
                   help="ARCTIC processed dir (optional augmentation)")
    p.add_argument("--dexycb_dir", type=Path, default=None,
                   help="DexYCB processed dir (optional augmentation)")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--hand",       default="shared", choices=["right", "left", "shared", "canonical"])
    p.add_argument("--epochs",     type=int,  default=EPOCHS)
    p.add_argument("--patience",   type=int,  default=PATIENCE)
    p.add_argument("--batch_size", type=int,  default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--val_every",  type=int,   default=2, help="Validate every N epochs")
    p.add_argument("--balanced_yaw_sampling", action="store_true",
                   help="Use inverse-frequency sampling over approach-direction yaw bins.")
    p.add_argument("--orientation_aware_sdf", action="store_true", default=True,
                   help="Inject dir_obj_local into object embedding fusion (default: on).")
    p.add_argument("--no_orientation_aware_sdf", dest="orientation_aware_sdf", action="store_false",
                   help="Disable orientation-aware SDF injection.")
    p.add_argument("--input_noise", type=float, default=0.0,
                   help="Selective Gaussian noise for continuous input dims during training.")
    p.add_argument("--model_type", default="transformer", choices=["lstm", "transformer"],
                   help="Model architecture (default: transformer).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.hot3d_dir is None and args.data_dir is None:
        raise SystemExit("Specify either --data_dir (single) or --hot3d_dir (multi-source)")
    train_lstm(args)
