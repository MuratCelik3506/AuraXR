"""build_dataset_mano.py — HOT3D → HDF5 MANO dataset for SDF-LSTM training.

Replaces build_dataset_temporal.py. Identical feature pipeline (25+4=29-dim),
but target changes from UMeTrack 22-dim joint angles → MANO 15-dim PCA pose.

HOT3D MANO format (mano_hand_pose_trajectory.jsonl):
  "pose"  : [15 floats]  — MANO PCA hand pose components (num_pca_comps=15)
  "betas" : [10 floats]  — MANO shape parameters (per-subject, NOT used in training)
  "wrist_xform": same as UMeTrack (t_xyz, q_wxyz)

HDF5 schema (output: dataset_mano.h5):
  {train,val}/
    features      (N, 25)  — core spatial feature (same as temporal pipeline)
    sdf_features  (N, 4)   — SDF local feature
    obj_id        (N,)     — BOP object ID (int32)
    targets       (N, 15)  — MANO PCA pose  ← changed from (N, 22) UMeTrack
    betas         (N, 10)  — MANO shape (stored, not used in loss)
    wrist_rot_6d  (N, 6)   — 6D wrist rotation (unchanged)
    distances     (N,)
    sequence_id   (N,)
    frame_index   (N,)
    is_mirror     (N,)
    contact       (N,)
  attrs["meta"] JSON: norm stats + architecture config

Run:
    .venv/bin/python3 src/build_dataset_mano.py \\
        --data_dir data/raw/hot3d/quest3/ \\
        --output_dir data/processed/hot3d_mano/left/ \\
        --hand left \\
        --sdf_dir data/models/sdf_grids/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from grip_categories import object_features
from hot3d_utils import (
    FEATURE_DIM,
    HAND_KEY,
    build_uid_to_bop,
    find_sequences,
    mirror_feature,
    mirror_wrist_rot,
    quat_conjugate,
    read_dynamic_objects,
    read_metadata,
    rotate_vec,
    wrist_rot_to_6d,
    zip_paths,
)
from sdf_utils import SDFDatabase, SDF_FEATURE_DIM

assert FEATURE_DIM == 25

CONTACT_DIST_M    = 0.10  # 10 cm wrist-to-object-center
CONTACT_V2_DIST_M = 0.12  # slightly wider proxy; train_lstm.py prefers this if present
MANO_POSE_DIM     = 15    # MANO PCA components (HOT3D stores 15)
MANO_BETAS_DIM    = 10    # MANO shape parameters
TOTAL_FEATURE_DIM = FEATURE_DIM + SDF_FEATURE_DIM   # 29


# ── MANO trajectory reader ─────────────────────────────────────────────────────

def read_mano_trajectory(hand_zip_path: Path) -> dict[int, dict[str, dict]]:
    """Parse mano_hand_pose_trajectory.jsonl from hand_data.zip.

    Returns: {timestamp_ns → {hand_id_str → {pose, betas, wrist_xform}}}
    hand_id_str: "0" = left, "1" = right (same as UMeTrack convention)
    """
    result: dict[int, dict[str, dict]] = {}
    with zipfile.ZipFile(hand_zip_path, "r") as zf:
        with zf.open("mano_hand_pose_trajectory.jsonl") as f:
            for line in f:
                entry = json.loads(line)
                ts = int(entry["timestamp_ns"])
                result[ts] = {k: v for k, v in entry["hand_poses"].items()}
    return result


# ── Mirror augmentation for MANO pose ─────────────────────────────────────────
# MANO PCA components encode hand deformations. For left↔right mirroring,
# components that correspond to abduction (x-axis rotations) flip sign.
# We approximate: negate odd-indexed components (empirically captures abduction).
# For exact mirror, run: python src/build_dataset_mano.py --compute_mirror_mask
_MANO_MIRROR_MASK: np.ndarray | None = None


def _default_mirror_mask() -> np.ndarray:
    """Approximate MANO pose mirror mask (sign per PCA component).
    Components 0,2,4,... are typically symmetric (flexion) → keep +1.
    Components 1,3,5,... are typically anti-symmetric (abduction) → flip -1.
    """
    mask = np.ones(MANO_POSE_DIM, dtype=np.float32)
    mask[1::2] = -1.0
    return mask


def _load_mirror_mask(mask_path: Path | None) -> np.ndarray:
    if mask_path is not None and mask_path.exists():
        data = json.loads(mask_path.read_text())
        return np.array(data["mirror_mask"], dtype=np.float32)
    return _default_mirror_mask()


def mirror_mano_pose(pose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply sign mask to 15-dim MANO PCA pose for left↔right augmentation."""
    return pose * mask


def parse_yaw_aug(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def yaw_matrix(deg: float) -> np.ndarray:
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def yaw_augment_sequence(frames: list[dict], yaw_deg: float, seq_id: int) -> list[dict]:
    """Rotate world-frame features around HOT3D world-up.

    Object-local direction and canonical-relative wrist 6D remain unchanged under
    a rigid global yaw of both wrist and object frames.
    """
    ry = yaw_matrix(yaw_deg)
    out = []
    for f in frames:
        feat = f["feature"].copy()
        feat[0:3] = ry @ feat[0:3]
        feat[8:11] = ry @ feat[8:11]
        out.append({
            **f,
            "feature": feat.astype(np.float32),
            "seq_id": np.int32(seq_id),
            "is_mirror": np.uint8(2),
            "augmentation": f"yaw_{yaw_deg:g}",
        })
    return out


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_sequence_frames(
    seq_dir: Path,
    hand_key: str,
    seq_id: int,
    sdf_db: SDFDatabase | None,
    mirror_mask: np.ndarray,
    add_wrist_obj_pos: bool = False,
) -> list[dict]:
    """Extract frames from one sequence in temporal order.

    Identical to build_dataset_temporal.extract_sequence_frames() except:
      - reads mano_hand_pose_trajectory.jsonl
      - frame["target"]  = pose (15,)    MANO PCA
      - frame["betas"]   = betas (10,)   MANO shape
      - no hand_confidence field (MANO trajectory doesn't carry it)
    """
    hand_zip, gt_zip = zip_paths(seq_dir)
    if hand_zip is None:
        return []

    try:
        mano_traj  = read_mano_trajectory(hand_zip)
        obj_by_ts  = read_dynamic_objects(gt_zip)
        metadata   = read_metadata(gt_zip)
    except KeyError:
        # mano_hand_pose_trajectory.jsonl not present in this zip (e.g. older dumps)
        return []
    except Exception as e:
        print(f"    [SKIP] {seq_dir.name}: {e}")
        return []

    uid_to_bop     = build_uid_to_bop(metadata)
    frames: list[dict] = []
    frame_idx      = 0
    prev_wrist_pos: np.ndarray | None = None
    prev_ts: int | None = None
    prev_obj_state: dict[int, tuple[np.ndarray, int]] = {}

    for ts, hand_poses in sorted(mano_traj.items()):
        if hand_key not in hand_poses:
            continue
        if ts not in obj_by_ts:
            continue

        pose_data    = hand_poses[hand_key]
        wrist_pos    = np.array(pose_data["wrist_xform"]["t_xyz"],  dtype=np.float32)
        wrist_q_wxyz = np.array(pose_data["wrist_xform"]["q_wxyz"], dtype=np.float32)
        mano_pose    = np.array(pose_data["pose"],  dtype=np.float32)   # (15,)
        mano_betas   = np.array(pose_data["betas"], dtype=np.float32)   # (10,)

        # Wrist velocity
        if prev_wrist_pos is not None and prev_ts is not None:
            dt = (ts - prev_ts) * 1e-9
            vel_world = (wrist_pos - prev_wrist_pos) / dt if dt > 0 else np.zeros(3, dtype=np.float32)
        else:
            vel_world = np.zeros(3, dtype=np.float32)
        prev_wrist_pos = wrist_pos
        prev_ts = ts

        # Per-object velocities
        cur_obj_vel: dict[int, np.ndarray] = {}
        cur_obj_state: dict[int, tuple[np.ndarray, int]] = {}
        for obj in obj_by_ts[ts]:
            bop_id = uid_to_bop.get(obj["object_uid"])
            if bop_id is None:
                continue
            pos = obj["pos_world"]
            cur_obj_state[bop_id] = (pos, ts)
            if bop_id in prev_obj_state:
                p_prev, ts_prev = prev_obj_state[bop_id]
                dt_obj = (ts - ts_prev) * 1e-9
                cur_obj_vel[bop_id] = ((pos - p_prev) / dt_obj).astype(np.float32) if dt_obj > 0 else np.zeros(3, np.float32)
            else:
                cur_obj_vel[bop_id] = np.zeros(3, dtype=np.float32)

        # Nearest object
        min_dist    = float("inf")
        best_core   = None
        best_dir    = None
        best_bop_id = 0
        best_obj_q  = None
        best_obj_pos = None

        for obj in obj_by_ts[ts]:
            bop_id = uid_to_bop.get(obj["object_uid"])
            if bop_id is None:
                continue
            delta = obj["pos_world"] - wrist_pos
            dist  = float(np.linalg.norm(delta))
            if dist < min_dist:
                min_dist     = dist
                direction    = delta / (dist + 1e-8)
                q_obj_inv    = quat_conjugate(obj["quat_world"])
                dir_obj_loc  = rotate_vec(q_obj_inv, direction)
                approach_spd = float(np.dot(vel_world, direction))
                obj_vel      = cur_obj_vel.get(bop_id, np.zeros(3, np.float32))
                wrist_rot_in = wrist_rot_to_6d(wrist_q_wxyz, direction)
                wrist_in_obj = rotate_vec(q_obj_inv, wrist_pos - obj["pos_world"])
                grip_oh, bbox = object_features(bop_id)
                # hand_confidence slot (dim 17) → set to 1.0 (MANO traj has no conf)
                parts = [
                    direction,       # [0-2]
                    dir_obj_loc,     # [3-5]
                    [dist],          # [6]
                    [approach_spd],  # [7]
                    obj_vel,         # [8-10]
                    wrist_rot_in,    # [11-16]
                    [1.0],           # [17] hand_confidence placeholder
                    grip_oh,         # [18-21]
                    bbox,            # [22-24]
                ]
                if add_wrist_obj_pos:
                    parts.append(wrist_in_obj)  # [25-27]
                best_core = np.concatenate(parts).astype(np.float32)
                best_dir     = direction
                best_bop_id  = bop_id
                best_obj_q   = obj["quat_world"]
                best_obj_pos = obj["pos_world"]

        prev_obj_state = cur_obj_state
        if best_core is None:
            continue

        # SDF feature
        sdf_feat = np.zeros(SDF_FEATURE_DIM, dtype=np.float32)
        if sdf_db is not None and best_obj_q is not None:
            q_inv        = quat_conjugate(best_obj_q)
            wrist_in_obj = rotate_vec(q_inv, wrist_pos - best_obj_pos)
            sdf_feat     = sdf_db.query(best_bop_id, wrist_in_obj)

        rot6d_target = wrist_rot_to_6d(wrist_q_wxyz, best_dir)

        frames.append({
            "feature":      best_core,
            "sdf_feature":  sdf_feat,
            "obj_id":       np.int32(best_bop_id),
            "target":       mano_pose,         # (15,) MANO PCA
            "betas":        mano_betas,        # (10,) stored for optional use
            "wrist_rot_6d": rot6d_target,
            "distance":     np.float32(min_dist),
            "seq_id":       np.int32(seq_id),
            "frame_index":  np.int32(frame_idx),
            "is_mirror":    np.uint8(0),
            "contact":      np.uint8(min_dist < CONTACT_DIST_M),
            "contact_v2":   np.uint8(min_dist < CONTACT_V2_DIST_M),
        })
        frame_idx += 1

    return frames


def mirror_sequence(frames: list[dict], mirrored_seq_id: int, mirror_mask: np.ndarray) -> list[dict]:
    """Mirror a complete sequence for left↔right augmentation."""
    return [{
        "feature":      mirror_feature(f["feature"]),
        "sdf_feature":  f["sdf_feature"].copy(),
        "obj_id":       f["obj_id"],
        "target":       mirror_mano_pose(f["target"], mirror_mask),
        "betas":        f["betas"].copy(),
        "wrist_rot_6d": mirror_wrist_rot(f["wrist_rot_6d"]),
        "distance":     f["distance"],
        "seq_id":       np.int32(mirrored_seq_id),
        "frame_index":  f["frame_index"],
        "is_mirror":    np.uint8(1),
        "contact":      f["contact"],
        "contact_v2":   f["contact_v2"],
    } for f in frames]


# ── Normalisation stats ────────────────────────────────────────────────────────

def feature_names(feature_dim: int) -> list[str]:
    names = [
        "dir_world_x", "dir_world_y", "dir_world_z",
        "dir_obj_local_x", "dir_obj_local_y", "dir_obj_local_z",
        "distance", "approach_speed",
        "obj_vel_x", "obj_vel_y", "obj_vel_z",
        "wrist_rot6d_0", "wrist_rot6d_1", "wrist_rot6d_2",
        "wrist_rot6d_3", "wrist_rot6d_4", "wrist_rot6d_5",
        "hand_confidence",
        "grip_power", "grip_precision", "grip_palmar", "grip_pinch",
        "bbox_x", "bbox_y", "bbox_z",
    ]
    if feature_dim == FEATURE_DIM + 3:
        names += ["wrist_obj_x", "wrist_obj_y", "wrist_obj_z"]
    return names


def compute_norm_stats(train_frames: list[dict]) -> dict:
    """Compute z-score stats from train-real frames only."""
    real = [f for f in train_frames if f["is_mirror"] == 0]
    if not real:
        raise ValueError("No train-real frames — cannot compute norm stats")

    def stats(arr: np.ndarray):
        m, s = arr.mean(0), arr.std(0)
        s = np.where(s < 1e-8, 1.0, s)
        return m.tolist(), s.tolist()

    feature_arr = np.stack([f["feature"] for f in real])
    feat_mean,  feat_std  = stats(feature_arr)
    sdf_mean,   sdf_std   = stats(np.stack([f["sdf_feature"]  for f in real]))
    tgt_mean,   tgt_std   = stats(np.stack([f["target"]       for f in real]))
    rot_mean,   rot_std   = stats(np.stack([f["wrist_rot_6d"] for f in real]))

    feature_dim = int(feature_arr.shape[1])
    total_input_dim = feature_dim + SDF_FEATURE_DIM
    return {
        "feature_mean":   feat_mean,  "feature_std":   feat_std,
        "sdf_mean":       sdf_mean,   "sdf_std":       sdf_std,
        "target_mean":    tgt_mean,   "target_std":    tgt_std,
        "wrist_rot_mean": rot_mean,   "wrist_rot_std": rot_std,
        "architecture": {
            "input_dim":       feature_dim,
            "sdf_input_dim":   SDF_FEATURE_DIM,
            "total_input_dim": total_input_dim,
            "output_dim":      MANO_POSE_DIM,   # 15 (changed from 22)
            "output_type":     "mano_pca",
            "mano_pose_dim":   MANO_POSE_DIM,
            "version":         6 if feature_dim > FEATURE_DIM else 5,
        },
        "feature_names": feature_names(feature_dim),
        "feature_version": "mano_v6_wrist_obj" if feature_dim > FEATURE_DIM else "mano_v5",
        "norm_stats_source": "train_real_only",
    }


# ── HDF5 writer ────────────────────────────────────────────────────────────────

def write_hdf5(output_path: Path, train_frames: list[dict], val_frames: list[dict], meta: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def seq_lengths(frames):
        counts = defaultdict(int)
        for f in frames:
            counts[int(f["seq_id"])] += 1
        return {int(k): int(v) for k, v in sorted(counts.items())}

    meta["train_sequence_lengths"] = seq_lengths(train_frames)
    meta["val_sequence_lengths"]   = seq_lengths(val_frames)

    with h5py.File(output_path, "w") as hf:
        hf.attrs["meta"] = json.dumps(meta)
        for split, frames in [("train", train_frames), ("val", val_frames)]:
            g = hf.create_group(split)
            kw = dict(compression="gzip")
            g.create_dataset("features",     data=np.stack([f["feature"]      for f in frames]), **kw)
            g.create_dataset("sdf_features", data=np.stack([f["sdf_feature"]  for f in frames]), **kw)
            g.create_dataset("obj_id",       data=np.array([f["obj_id"]        for f in frames], np.int32),   **kw)
            g.create_dataset("targets",      data=np.stack([f["target"]       for f in frames]), **kw)  # (N,15)
            g.create_dataset("betas",        data=np.stack([f["betas"]        for f in frames]), **kw)  # (N,10)
            g.create_dataset("wrist_rot_6d", data=np.stack([f["wrist_rot_6d"] for f in frames]), **kw)
            g.create_dataset("distances",    data=np.array([f["distance"]      for f in frames], np.float32), **kw)
            g.create_dataset("sequence_id",  data=np.array([f["seq_id"]        for f in frames], np.int32),  **kw)
            g.create_dataset("frame_index",  data=np.array([f["frame_index"]   for f in frames], np.int32),  **kw)
            g.create_dataset("is_mirror",    data=np.array([f["is_mirror"]     for f in frames], np.uint8),  **kw)
            g.create_dataset("contact",      data=np.array([f["contact"]       for f in frames], np.uint8),  **kw)
            g.create_dataset("contact_v2",   data=np.array([f["contact_v2"]    for f in frames], np.uint8),  **kw)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True, type=Path, nargs="+")
    p.add_argument("--output_dir",  required=True, type=Path)
    p.add_argument("--hand",        default="right", choices=["right", "left"])
    p.add_argument("--val_frac",    default=0.15, type=float)
    p.add_argument("--seed",        default=42, type=int)
    p.add_argument("--no_mirror",   action="store_true")
    p.add_argument("--mirror_mask", default=None, type=Path,
                   help="JSON file with mirror_mask array (15,). Auto-approximated if absent.")
    p.add_argument("--sdf_dir",     default="data/models/sdf_grids", type=Path)
    p.add_argument("--add_wrist_obj_pos", action="store_true",
                   help="Append wrist position in object coordinates to core features (+3 dims).")
    p.add_argument("--yaw_aug", default="", type=str,
                   help="Comma-separated train-only global yaw augmentations in degrees, e.g. '45,-45,90,-90'.")
    p.add_argument("--dry_run",     action="store_true",
                   help="Parse first 5 sequences only, print stats, do not write HDF5.")
    return p.parse_args()


def main():
    args = parse_args()
    hand_key     = HAND_KEY[args.hand]
    mirror_mask  = _load_mirror_mask(args.mirror_mask)
    mirror_label = "OFF" if args.no_mirror else f"ON (mask={mirror_mask.tolist()})"
    yaw_angles = parse_yaw_aug(args.yaw_aug)
    print(f"build_dataset_mano: hand={args.hand}  mirror={mirror_label} "
          f"wrist_obj={args.add_wrist_obj_pos} yaw_aug={yaw_angles}")

    # SDF database
    sdf_db = None
    sdf_dir = Path(args.sdf_dir)
    if sdf_dir.exists() and list(sdf_dir.glob("bop*.npz")):
        print(f"Loading SDF database from {sdf_dir} …")
        sdf_db = SDFDatabase(sdf_dir)
        print(f"  {len(sdf_db)} SDF grids loaded")
    else:
        print(f"[WARN] SDF grids not found at {sdf_dir} — sdf_features will be zeros")

    # Discover sequences
    sequences = []
    for d in args.data_dir:
        found = find_sequences(d, split="train")
        print(f"  {d}: {len(found)} sequences")
        sequences.extend(found)
    if not sequences:
        print("[ERROR] No sequences found.")
        return

    if args.dry_run:
        sequences = sequences[:5]
        print(f"[DRY RUN] Processing first {len(sequences)} sequences …")

    # Train/val split at sequence level
    rng = random.Random(args.seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * args.val_frac))
    val_set = {str(s) for s in shuffled[:n_val]}

    train_seqs: list[list[dict]] = []
    val_seqs:   list[list[dict]] = []
    skipped = 0

    for seq_id, seq_dir in enumerate(tqdm(sequences, desc="Sequences")):
        frames = extract_sequence_frames(
            seq_dir, hand_key, seq_id, sdf_db, mirror_mask,
            add_wrist_obj_pos=args.add_wrist_obj_pos)
        if not frames:
            skipped += 1
            continue
        if str(seq_dir) in val_set:
            val_seqs.append(frames)
            if not args.no_mirror:
                val_seqs.append(mirror_sequence(frames, seq_id + len(sequences), mirror_mask))
        else:
            train_seqs.append(frames)
            if not args.no_mirror:
                train_seqs.append(mirror_sequence(frames, seq_id + len(sequences), mirror_mask))
            next_aug_seq_id = seq_id + 2 * len(sequences)
            for yaw in yaw_angles:
                train_seqs.append(yaw_augment_sequence(frames, yaw, next_aug_seq_id))
                next_aug_seq_id += len(sequences)

    rng_tr = random.Random(args.seed + 1)
    rng_tr.shuffle(train_seqs)
    train_frames = [f for seq in train_seqs for f in seq]
    val_frames   = [f for seq in val_seqs   for f in seq]

    print(f"\nExtracted: {len(train_frames)} train / {len(val_frames)} val frames")
    n_contact = sum(1 for f in train_frames if f["contact"])
    print(f"  Contact frames (dist<{CONTACT_DIST_M*100:.0f}cm): {n_contact} ({100*n_contact/max(1,len(train_frames)):.1f}%)")
    if train_frames:
        poses = np.stack([f["target"] for f in train_frames[:1000]])
        print(f"  MANO pose range: [{poses.min():.3f}, {poses.max():.3f}]  (sanity: should be ~[-2, 2])")

    if args.dry_run:
        print("[DRY RUN] Done — no HDF5 written.")
        return

    meta = compute_norm_stats(train_frames)
    meta["feature_dim"] = meta["architecture"]["input_dim"]
    meta["augmentation_flags"] = {
        "mirror": not args.no_mirror,
        "yaw_angles_deg": yaw_angles,
        "add_wrist_obj_pos": args.add_wrist_obj_pos,
    }
    meta["normalization_policy"] = "mean_std_train_real_only"
    out  = args.output_dir / "dataset_mano.h5"
    print(f"\nWriting {out} …")
    write_hdf5(out, train_frames, val_frames, meta)
    print(f"Done. {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
