"""Run the SDF-LSTM in a synthetic 3D hand-object scene and export MP4.

This is a Unity-free smoke test for the deployed inference path:
  - builds a synthetic wrist trajectory around a selected HOT3D BOP object
  - feeds normalized frame features through SDFLSTMModel.forward(...)
  - autoregressively feeds the previous predicted wrist rotation
  - renders predicted MANO hand joints plus the target object in 3D

Example:
    .venv/bin/python3 src/simulate_lstm_3d_video.py \\
        --hand right --object bottle --out results/synthetic_right_bottle.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from grip_categories import OBJ_BBOX, OBJ_NAMES, object_features
from mano_fk import MANOForwardKinematics
from model import SDFLSTMModel
from sdf_utils import SDFDatabase


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

OBJECT_PRESETS = {
    "bottle": 13,
    "cup": 9,
    "can": 10,
    "plate": 3,
    "spoon": 4,
    "mouse": 31,
    "phone": 24,
    "marker": 32,
    "vase": 16,
    "remote": 33,
}

WRIST_DIMS = slice(11, 17)
IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def infer_feat_dim(state_dict: dict[str, torch.Tensor]) -> int:
    w = state_dict.get("feat_proj.0.weight")
    return int(w.shape[1]) if w is not None else 29


def infer_orientation_aware(state_dict: dict[str, torch.Tensor]) -> bool:
    w = state_dict.get("obj_inj.0.weight")
    return bool(w is not None and int(w.shape[1]) == 99)


def load_meta(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return json.loads(f.attrs["meta"])


def resolve_bop_id(name_or_id: str) -> int:
    if name_or_id.lower() in OBJECT_PRESETS:
        return OBJECT_PRESETS[name_or_id.lower()]
    try:
        return int(name_or_id)
    except ValueError as exc:
        options = ", ".join(OBJECT_PRESETS)
        raise SystemExit(f"Unknown object '{name_or_id}'. Use one of: {options}, or a BOP ID.") from exc


def build_trajectory(frames: int, radius: float, start_dist: float, end_dist: float, mode: str) -> np.ndarray:
    """Return wrist position in object-local metres, shape (T, 3)."""
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    dist = start_dist + (end_dist - start_dist) * (3 * t**2 - 2 * t**3)

    if mode == "front":
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        z = -dist
    elif mode == "arc":
        angle = np.deg2rad(-35.0 + 70.0 * t)
        x = np.sin(angle) * radius
        y = 0.02 * np.sin(2.0 * np.pi * t)
        z = -dist
    elif mode == "side":
        x = -dist
        y = np.zeros_like(t)
        z = np.full_like(t, radius)
    else:
        raise SystemExit(f"Unknown trajectory '{mode}'.")

    return np.stack([x, y, z], axis=1).astype(np.float32)


def build_raw_features(
    wrist_pos_obj: np.ndarray,
    prev_wrist_pos_obj: np.ndarray,
    bop_id: int,
    sdf_db: SDFDatabase,
    fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build unnormalized core features and SDF features for one frame."""
    delta = -wrist_pos_obj
    dist = float(np.linalg.norm(delta))
    direction = delta / (dist + 1e-8)
    vel_world = (wrist_pos_obj - prev_wrist_pos_obj) * float(fps)
    approach_spd = float(np.dot(vel_world, direction))
    obj_vel = np.zeros(3, dtype=np.float32)
    grip_oh, bbox = object_features(bop_id)

    core = np.concatenate([
        direction,           # [0-2] dir_world, object frame == world frame
        direction,           # [3-5] dir_obj_local
        [dist],              # [6]
        [approach_spd],      # [7]
        obj_vel,             # [8-10]
        IDENTITY_6D,         # [11-16], overwritten autoregressively after frame 0
        [1.0],               # [17] hand confidence
        grip_oh,             # [18-21]
        bbox,                # [22-24]
        wrist_pos_obj,       # [25-27]
    ]).astype(np.float32)
    return core, sdf_db.query(bop_id, wrist_pos_obj)


def normalize_inputs(core: np.ndarray, sdf: np.ndarray, meta: dict, feat_dim: int) -> torch.Tensor:
    feat_mean = np.asarray(meta["feature_mean"], dtype=np.float32)
    feat_std = np.asarray(meta["feature_std"], dtype=np.float32)
    sdf_mean = np.asarray(meta["sdf_mean"], dtype=np.float32)
    sdf_std = np.asarray(meta["sdf_std"], dtype=np.float32)

    core_n = (core - feat_mean) / np.maximum(feat_std, 1e-6)
    sdf_n = (sdf - sdf_mean) / np.maximum(sdf_std, 1e-6)
    full = np.concatenate([core_n, sdf_n]).astype(np.float32)
    if full.shape[0] > feat_dim:
        full = full[:feat_dim]
    elif full.shape[0] < feat_dim:
        full = np.pad(full, (0, feat_dim - full.shape[0]))
    return torch.from_numpy(full).unsqueeze(0)


@torch.no_grad()
def run_model(args, model, meta, embed_matrix, bop_ids, sdf_db, device):
    bop_id = resolve_bop_id(args.object)
    positions = build_trajectory(args.frames, args.arc_radius, args.start_dist, args.end_dist, args.trajectory)
    bop_to_idx = {int(bid): i for i, bid in enumerate(bop_ids)}
    obj_emb = torch.from_numpy(embed_matrix[bop_to_idx.get(bop_id, 0)].astype(np.float32)).to(device).unsqueeze(0)

    feat_dim = model.feat_proj[0].in_features
    feat_mean = torch.tensor(meta["feature_mean"], dtype=torch.float32, device=device)
    feat_std = torch.tensor(meta["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    tgt_mean = np.asarray(meta["target_mean"], dtype=np.float32)
    tgt_std = np.asarray(meta["target_std"], dtype=np.float32)

    h, c = model.initial_state(batch_size=1, device=device)
    pred_pose, pred_wrist, pred_contact = [], [], []
    prev_wrist_raw = None

    for i, pos in enumerate(positions):
        prev_pos = positions[max(i - 1, 0)]
        core, sdf = build_raw_features(pos, prev_pos, bop_id, sdf_db, args.fps)
        frame = normalize_inputs(core, sdf, meta, feat_dim).to(device)
        if prev_wrist_raw is not None and frame.shape[1] >= WRIST_DIMS.stop:
            frame[:, WRIST_DIMS] = (prev_wrist_raw - feat_mean[WRIST_DIMS]) / feat_std[WRIST_DIMS]

        pose, wrist, contact, h, c = model(frame, obj_emb, h, c)
        pred_pose.append(pose.squeeze(0).cpu().numpy())
        pred_wrist.append(wrist.squeeze(0).cpu().numpy())
        pred_contact.append(float(contact.item()))
        prev_wrist_raw = wrist.detach()

    pose_raw = np.stack(pred_pose) * tgt_std + tgt_mean
    return bop_id, positions, pose_raw, np.stack(pred_wrist), np.asarray(pred_contact, dtype=np.float32)


def draw_box(ax, bbox_m: np.ndarray):
    b = bbox_m * 1000.0
    corners = np.array([
        [-b[0], -b[1], -b[2]], [b[0], -b[1], -b[2]], [b[0], b[1], -b[2]], [-b[0], b[1], -b[2]],
        [-b[0], -b[1], b[2]], [b[0], -b[1], b[2]], [b[0], b[1], b[2]], [-b[0], b[1], b[2]],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="#777777", lw=1.5, alpha=0.75)


def render_video(joints_m: np.ndarray, positions_m: np.ndarray, contacts: np.ndarray, bop_id: int, out: Path, fps: int):
    joints_mm = joints_m * 1000.0 + positions_m[:, None, :] * 1000.0
    pos_mm = positions_m * 1000.0
    all_pts = np.concatenate([joints_mm.reshape(-1, 3), pos_mm], axis=0)
    center = all_pts.mean(axis=0)
    span = max(float(np.ptp(all_pts, axis=0).max()) * 0.62, 220.0)
    lo, hi = center - span, center + span

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_box(ax, OBJ_BBOX.get(bop_id, np.array([0.04, 0.04, 0.04], dtype=np.float32)))
    ax.scatter([0], [0], [0], s=70, c="#666666", depthshade=False)
    trail, = ax.plot([], [], [], color="#d35400", lw=1.5, alpha=0.65)

    lines = []
    for i, j in HAND_EDGES:
        line, = ax.plot(
            [joints_mm[0, i, 0], joints_mm[0, j, 0]],
            [joints_mm[0, i, 1], joints_mm[0, j, 1]],
            [joints_mm[0, i, 2], joints_mm[0, j, 2]],
            color="#1f77b4",
            lw=2.2,
        )
        lines.append(line)
    scatter = ax.scatter(joints_mm[0, :, 0], joints_mm[0, :, 1], joints_mm[0, :, 2], c="#1f77b4", s=18, depthshade=False)
    title = ax.set_title("")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=22, azim=42)

    def update(frame: int):
        pts = joints_mm[frame]
        for k, (i, j) in enumerate(HAND_EDGES):
            lines[k].set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
            lines[k].set_3d_properties([pts[i, 2], pts[j, 2]])
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        trail.set_data(pos_mm[:frame + 1, 0], pos_mm[:frame + 1, 1])
        trail.set_3d_properties(pos_mm[:frame + 1, 2])
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        name = OBJ_NAMES.get(bop_id, f"bop_{bop_id}")
        title.set_text(f"{name} | frame {frame + 1}/{len(joints_mm)} | contact={contacts[frame]:.2f}")
        return lines + [scatter, trail, title]

    update(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=len(joints_mm), interval=1000 // fps, blit=False)
    ani.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=2200))
    plt.close(fig)


def procedural_hand_joints(pose: np.ndarray, wrist: np.ndarray, hand: str) -> np.ndarray:
    """Fallback 21-joint hand in metres when smplx/MANO is unavailable.

    This is not anatomically exact; it is a visual smoke-test skeleton whose
    finger curl is driven by the model's pose output.
    """
    T = pose.shape[0]
    joints = np.zeros((T, 21, 3), dtype=np.float32)
    x_sign = -1.0 if hand == "left" else 1.0
    finger_roots = {
        1: np.array([0.028 * x_sign, 0.000, 0.005], dtype=np.float32),
        5: np.array([0.010 * x_sign, 0.000, 0.010], dtype=np.float32),
        9: np.array([-0.026 * x_sign, 0.000, 0.000], dtype=np.float32),
        13: np.array([-0.010 * x_sign, 0.000, 0.006], dtype=np.float32),
        17: np.array([0.042 * x_sign, -0.014, -0.022], dtype=np.float32),
    }
    lengths = {
        1: [0.036, 0.025, 0.019],
        5: [0.041, 0.028, 0.020],
        9: [0.032, 0.021, 0.017],
        13: [0.036, 0.024, 0.018],
        17: [0.026, 0.021, 0.017],
    }
    pose_scale = np.tanh(pose.reshape(T, 5, 3).mean(axis=2))

    for t in range(T):
        joints[t, 0] = 0.0
        for f_idx, root_idx in enumerate([1, 5, 9, 13, 17]):
            curl = 0.35 + 0.85 / (1.0 + np.exp(-pose_scale[t, f_idx]))
            direction = np.array([0.0, 0.18 * curl, 1.0], dtype=np.float32)
            if root_idx == 17:
                direction = np.array([0.42 * x_sign, -0.10, 0.70], dtype=np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            joints[t, root_idx] = finger_roots[root_idx]
            cur = joints[t, root_idx].copy()
            for seg, length in enumerate(lengths[root_idx], start=1):
                cur = cur + direction * length
                joints[t, root_idx + seg] = cur

    # Apply predicted wrist rotation when it is numerically valid.
    rot = rot6d_to_matrix_np(wrist)
    return np.einsum("tij,tkj->tki", rot, joints)


def rot6d_to_matrix_np(r: np.ndarray) -> np.ndarray:
    a1 = r[:, :3]
    a2 = r[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Unity-free 3D SDF-LSTM smoke test video.")
    p.add_argument("--hand", default="right", choices=["left", "right"])
    p.add_argument("--ckpt", default=None, type=Path)
    p.add_argument("--data", default=None, type=Path, help="dataset_mano.h5 used only for normalization metadata")
    p.add_argument("--object", default="bottle")
    p.add_argument("--trajectory", default="arc", choices=["front", "arc", "side"])
    p.add_argument("--frames", default=120, type=int)
    p.add_argument("--fps", default=15, type=int)
    p.add_argument("--start_dist", default=0.42, type=float)
    p.add_argument("--end_dist", default=0.055, type=float)
    p.add_argument("--arc_radius", default=0.12, type=float)
    p.add_argument("--out", default=None, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent
    if args.ckpt is None:
        args.ckpt = root / f"checkpoints/lstm_{args.hand}/best.pt"
    if args.data is None:
        args.data = root / f"data/processed/hot3d_mano/{args.hand}/dataset_mano.h5"
    if args.out is None:
        args.out = root / f"results/synthetic_{args.hand}_{args.object}_{args.trajectory}.mp4"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    feat_dim = infer_feat_dim(ckpt["model"])
    model = SDFLSTMModel(
        feat_dim=feat_dim,
        orientation_aware_sdf=infer_orientation_aware(ckpt["model"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = load_meta(args.data)
    embed_matrix = np.load(root / "data/models/sdf_grids/sdf_embed_matrix.npy")
    bop_ids = np.load(root / "data/models/sdf_grids/sdf_bop_ids.npy")
    sdf_db = SDFDatabase(root / "data/models/sdf_grids")

    bop_id, positions, pose, wrist, contacts = run_model(args, model, meta, embed_matrix, bop_ids, sdf_db, device)
    try:
        fk = MANOForwardKinematics(hand=args.hand)
        joints_m = fk(pose, wrist)
        print("Hand renderer: MANO FK")
    except Exception as exc:
        print(f"[WARN] MANO FK unavailable ({exc}). Using procedural skeleton fallback.")
        joints_m = procedural_hand_joints(pose, wrist, args.hand)
    render_video(joints_m, positions, contacts, bop_id, args.out, args.fps)

    print(f"Saved: {args.out}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Object: {OBJ_NAMES.get(bop_id, f'bop_{bop_id}')} ({bop_id})")
    print(f"Contact probability range: {contacts.min():.3f} - {contacts.max():.3f}")


if __name__ == "__main__":
    main()
