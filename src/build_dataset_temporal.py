"""build_dataset_temporal.py — HOT3D → HDF5 temporal dataset for SDF-LSTM training.

Differences from build_dataset.py (v3, stateless):
  - Frame ordering preserved within sequences (sorted by timestamp)
  - frame_index stored per frame for temporal window reconstruction
  - SDF features added: [sdf_value(1), sdf_gradient(3)] queried at wrist pos in obj frame
  - nearest_bop_id stored for SDF embedding lookup
  - Mirror augmentation applied at SEQUENCE level (all frames flipped together)
  - Output: dataset_temporal.h5

Feature layout (29 dims):
  [0-7]   spatial (same as v3: dir_world, dir_obj_local, dist, approach_speed)
  [8-10]  obj_vel
  [11-16] wrist_rot_6d
  [17-20] grip_onehot   (hand_confidence slot replaced by grip[0])
  [21-28] bbox(3) + sdf_local(4) + padding → actually:
          let's keep 25-dim core + 4-dim sdf = 29-dim total

Actual layout:
  [0-24]  core 25-dim feature (same as v3, dim[17]=hand_conf kept for compat)
  [25-28] sdf_local: sdf_value(1), sdf_grad_xyz(3)

DataLoader interface:
  seq_id + frame_index → reconstruct temporal order
  Sequence lengths stored in attrs["sequence_lengths"] as JSON

Run:
    .venv/bin/python3 src/build_dataset_temporal.py \\
        --data_dir data/raw/hot3d/quest3/ \\
        --output_dir data/left_temporal/ \\
        --hand left \\
        --sdf_dir data/models/sdf_grids/

Output (output_dir/dataset_temporal.h5):
  train/features      (N, 25)  — core feature (backward compat)
  train/sdf_features  (N, 4)   — SDF local feature
  train/obj_id        (N,)     — BOP object ID (int32)
  train/targets       (N, 22)
  train/wrist_rot_6d  (N, 6)
  train/distances     (N,)
  train/sequence_id   (N,)     — which sequence (int32)
  train/frame_index   (N,)     — position within sequence, 0-based (int32)
  train/is_mirror     (N,)     — 0=real, 1=mirror
  train/contact       (N,)     — 1 if distance < CONTACT_DIST_M
  attrs["meta"]  JSON: norm stats + architecture config + sequence_lengths
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from grip_categories import OBJ_GRIP, OBJ_BBOX, object_features
from hot3d_utils import (
    FEATURE_DIM,
    HAND_KEY,
    build_uid_to_bop,
    find_sequences,
    mirror_feature,
    mirror_joints,
    mirror_wrist_rot,
    quat_conjugate,
    read_dynamic_objects,
    read_metadata,
    read_umetrack_trajectory,
    rotate_vec,
    wrist_rot_to_6d,
    zip_paths,
)
from sdf_utils import SDFDatabase, SDF_FEATURE_DIM

assert FEATURE_DIM == 25

CONTACT_DIST_M = 0.10  # 10 cm wrist-to-object-center — captures grip frames
CONTACT_V2_DIST_M = 0.12  # M1 minimum proxy; real FK/SDF contact comes later
TOTAL_FEATURE_DIM = FEATURE_DIM + SDF_FEATURE_DIM  # 25 + 4 = 29


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True, type=Path, nargs="+")
    p.add_argument("--output_dir",  required=True, type=Path)
    p.add_argument("--hand",        default="right", choices=["right", "left"])
    p.add_argument("--val_frac",    default=0.15, type=float)
    p.add_argument("--seed",        default=42, type=int)
    p.add_argument("--no_mirror",   action="store_true",
                   help="Disable mirror augmentation.")
    p.add_argument("--sdf_dir",     default="data/models/sdf_grids", type=Path,
                   help="Directory with pre-computed SDF grids (bop##.npz).")
    return p.parse_args()


def extract_sequence_frames(
    seq_dir: Path,
    hand_key: str,
    seq_id: int,
    sdf_db: SDFDatabase,
) -> list[dict]:
    """Extract frames from one sequence in TEMPORAL ORDER.

    Returns list of frame dicts sorted by timestamp, each containing:
        feature      (25,) — core spatial features (same as v3)
        sdf_feature  (4,)  — SDF at wrist position
        obj_id       int   — BOP ID of nearest object
        target       (22,) — UMeTrack joint angles
        wrist_rot_6d (6,)
        distance     float
        frame_index  int   — 0-based position in this sequence
        seq_id       int
        is_mirror    uint8
        contact      uint8
        contact_v2   uint8 — minimum sanity proxy, distance < CONTACT_V2_DIST_M
    """
    hand_zip, gt_zip = zip_paths(seq_dir)
    if hand_zip is None:
        return []

    try:
        ume_traj  = read_umetrack_trajectory(hand_zip)
        obj_by_ts = read_dynamic_objects(gt_zip)
        metadata  = read_metadata(gt_zip)
    except Exception as e:
        print(f"    [SKIP] {seq_dir.name}: {e}")
        return []

    uid_to_bop     = build_uid_to_bop(metadata)
    frames         = []
    frame_idx      = 0
    prev_wrist_pos: np.ndarray | None = None
    prev_ts: int | None = None
    prev_obj_state: dict[int, tuple[np.ndarray, int]] = {}

    for ts, hand_poses in sorted(ume_traj.items()):
        if hand_key not in hand_poses:
            continue
        if ts not in obj_by_ts:
            continue

        pose         = hand_poses[hand_key]
        wrist_pos    = np.array(pose["wrist_xform"]["t_xyz"],  dtype=np.float32)
        wrist_q_wxyz = np.array(pose["wrist_xform"]["q_wxyz"], dtype=np.float32)
        joint_angles = np.array(pose["joint_angles"],          dtype=np.float32)
        hand_conf    = float(pose.get("hand_confidence", 1.0))

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
                if dt_obj > 0:
                    cur_obj_vel[bop_id] = ((pos - p_prev) / dt_obj).astype(np.float32)
                else:
                    cur_obj_vel[bop_id] = np.zeros(3, dtype=np.float32)
            else:
                cur_obj_vel[bop_id] = np.zeros(3, dtype=np.float32)

        # Nearest object
        min_dist     = float("inf")
        best_core    = None
        best_dir     = None
        best_bop_id  = 0
        best_obj_q   = None
        best_obj_pos = None

        for obj in obj_by_ts[ts]:
            bop_id = uid_to_bop.get(obj["object_uid"])
            if bop_id is None:
                continue

            delta = obj["pos_world"] - wrist_pos
            dist  = float(np.linalg.norm(delta))

            if dist < min_dist:
                min_dist      = dist
                direction     = delta / (dist + 1e-8)
                q_obj_inv     = quat_conjugate(obj["quat_world"])
                dir_obj_loc   = rotate_vec(q_obj_inv, direction)
                approach_spd  = float(np.dot(vel_world, direction))
                obj_vel       = cur_obj_vel.get(bop_id, np.zeros(3, dtype=np.float32))
                wrist_rot_in  = wrist_rot_to_6d(wrist_q_wxyz, direction)
                grip_oh, bbox = object_features(bop_id)
                best_core = np.concatenate([
                    direction,        # [0-2]
                    dir_obj_loc,      # [3-5]
                    [dist],           # [6]
                    [approach_spd],   # [7]
                    obj_vel,          # [8-10]
                    wrist_rot_in,     # [11-16]
                    [hand_conf],      # [17]
                    grip_oh,          # [18-21]
                    bbox,             # [22-24]
                ]).astype(np.float32)
                best_dir     = direction
                best_bop_id  = bop_id
                best_obj_q   = obj["quat_world"]
                best_obj_pos = obj["pos_world"]

        prev_obj_state = cur_obj_state

        if best_core is None:
            continue

        # SDF feature: query at wrist position in object local frame
        sdf_feat = np.zeros(SDF_FEATURE_DIM, dtype=np.float32)
        if sdf_db is not None and best_obj_q is not None:
            q_inv = quat_conjugate(best_obj_q)
            wrist_in_obj = rotate_vec(q_inv, wrist_pos - best_obj_pos)
            sdf_feat = sdf_db.query(best_bop_id, wrist_in_obj)

        rot6d_target = wrist_rot_to_6d(wrist_q_wxyz, best_dir)

        frames.append({
            "feature":      best_core,
            "sdf_feature":  sdf_feat,
            "obj_id":       np.int32(best_bop_id),
            "target":       joint_angles,
            "wrist_rot_6d": rot6d_target,
            "distance":     np.float32(min_dist),
            "seq_id":       np.int32(seq_id),
            "frame_index":  np.int32(frame_idx),
            "is_mirror":    np.uint8(0),
            "contact":      np.uint8(min_dist < CONTACT_DIST_M),
            "contact_v2":    np.uint8(min_dist < CONTACT_V2_DIST_M),
        })
        frame_idx += 1

    return frames


def mirror_sequence(frames: list[dict], mirrored_seq_id: int | None = None) -> list[dict]:
    """Mirror a complete sequence (applied at sequence level for temporal consistency)."""
    mirrored = []
    for f in frames:
        mirrored.append({
            "feature":      mirror_feature(f["feature"]),
            "sdf_feature":  f["sdf_feature"].copy(),        # SDF is symmetric
            "obj_id":       f["obj_id"],
            "target":       mirror_joints(f["target"]),
            "wrist_rot_6d": mirror_wrist_rot(f["wrist_rot_6d"]),
            "distance":     f["distance"],
            "seq_id":       np.int32(mirrored_seq_id if mirrored_seq_id is not None else int(f["seq_id"])),
            "frame_index":  f["frame_index"],
            "is_mirror":    np.uint8(1),
            "contact":      f["contact"],
            "contact_v2":    f.get("contact_v2", f["contact"]),
        })
    return mirrored


def compute_norm_stats(train_frames: list[dict]) -> dict:
    """Compute normalization from train-real frames only.

    Validation frames and mirrored augmentation are deliberately excluded to
    avoid leakage and augmentation-induced distribution shifts in the stats.
    """
    real = [f for f in train_frames if f["is_mirror"] == 0]
    if not real:
        raise ValueError("Cannot compute normalization stats: no train-real frames")
    features    = np.stack([f["feature"]      for f in real])
    sdf_feats   = np.stack([f["sdf_feature"]  for f in real])
    targets     = np.stack([f["target"]       for f in real])
    wrist_rots  = np.stack([f["wrist_rot_6d"] for f in real])

    def stats(arr):
        m, s = arr.mean(0), arr.std(0)
        s = np.where(s < 1e-8, 1.0, s)
        return m.tolist(), s.tolist()

    feat_mean, feat_std   = stats(features)
    sdf_mean,  sdf_std    = stats(sdf_feats)
    tgt_mean,  tgt_std    = stats(targets)
    rot_mean,  rot_std    = stats(wrist_rots)

    return {
        "feature_mean":     feat_mean,
        "feature_std":      feat_std,
        "sdf_mean":         sdf_mean,
        "sdf_std":          sdf_std,
        "target_mean":      tgt_mean,
        "target_std":       tgt_std,
        "wrist_rot_mean":   rot_mean,
        "wrist_rot_std":    rot_std,
        "architecture": {
            "input_dim":     FEATURE_DIM,
            "sdf_input_dim": SDF_FEATURE_DIM,
            "total_input_dim": TOTAL_FEATURE_DIM,
            "output_dim":    22,
            "version":       4,           # temporal version
        },
        "norm_stats_source": "train_real_only",
    }


def write_hdf5(output_path: Path, train_frames: list, val_frames: list, meta: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute sequence lengths for temporal indexing
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
            g.create_dataset("features",      data=np.stack([f["feature"]      for f in frames]), compression="gzip")
            g.create_dataset("sdf_features",  data=np.stack([f["sdf_feature"]  for f in frames]), compression="gzip")
            g.create_dataset("obj_id",        data=np.array([f["obj_id"]        for f in frames], np.int32),  compression="gzip")
            g.create_dataset("targets",       data=np.stack([f["target"]       for f in frames]), compression="gzip")
            g.create_dataset("wrist_rot_6d",  data=np.stack([f["wrist_rot_6d"] for f in frames]), compression="gzip")
            g.create_dataset("distances",     data=np.array([f["distance"]      for f in frames], np.float32), compression="gzip")
            g.create_dataset("sequence_id",   data=np.array([f["seq_id"]        for f in frames], np.int32),  compression="gzip")
            g.create_dataset("frame_index",   data=np.array([f["frame_index"]   for f in frames], np.int32),  compression="gzip")
            g.create_dataset("is_mirror",     data=np.array([f["is_mirror"]     for f in frames], np.uint8),  compression="gzip")
            g.create_dataset("contact",       data=np.array([f["contact"]       for f in frames], np.uint8),  compression="gzip")
            g.create_dataset("contact_v2",    data=np.array([f.get("contact_v2", f["contact"]) for f in frames], np.uint8), compression="gzip")


def main():
    args = parse_args()
    hand_key = HAND_KEY[args.hand]
    print(f"build_dataset_temporal: hand={args.hand}  mirror={'OFF' if args.no_mirror else 'ON'}")

    # Load SDF database if available
    sdf_db = None
    if args.sdf_dir.exists() and list(args.sdf_dir.glob("bop*.npz")):
        print(f"Loading SDF database from {args.sdf_dir} …")
        sdf_db = SDFDatabase(args.sdf_dir)
        print(f"  {len(sdf_db)} SDF grids loaded")
    else:
        print(f"[WARN] SDF grids not found at {args.sdf_dir} — sdf_features will be zeros")

    # Find sequences
    sequences = []
    for d in args.data_dir:
        found = find_sequences(d, split="train")
        print(f"  {d}: {len(found)} sequences")
        sequences.extend(found)
    if not sequences:
        print("[ERROR] No sequences found.")
        return

    # Train/val split at sequence level
    rng = random.Random(args.seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * args.val_frac))
    val_set = {str(s) for s in shuffled[:n_val]}

    # Extract frames PER SEQUENCE (temporal order preserved)
    train_seqs: list[list[dict]] = []
    val_seqs:   list[list[dict]] = []
    skipped = 0

    for seq_id, seq_dir in enumerate(tqdm(sequences, desc="Sequences")):
        frames = extract_sequence_frames(seq_dir, hand_key, seq_id, sdf_db)
        if not frames:
            skipped += 1
            continue
        if str(seq_dir) in val_set:
            val_seqs.append(frames)
            if not args.no_mirror:
                val_seqs.append(mirror_sequence(frames, mirrored_seq_id=seq_id + len(sequences)))
        else:
            train_seqs.append(frames)
            if not args.no_mirror:
                train_seqs.append(mirror_sequence(frames, mirrored_seq_id=seq_id + len(sequences)))

    # Flatten to frame list — sequences in order, frames within each sequence in order
    # (sequences themselves are shuffled for diversity in mini-batches)
    rng_tr = random.Random(args.seed + 1)
    rng_tr.shuffle(train_seqs)
    train_frames = [f for seq in train_seqs for f in seq]
    val_frames   = [f for seq in val_seqs   for f in seq]

    print(f"\nExtracted: {len(train_frames)} train frames ({len(train_seqs)} seqs, {skipped} skipped)")
    contact_tr = sum(1 for f in train_frames if f["contact"])
    contact_v2_tr = sum(1 for f in train_frames if f.get("contact_v2", f["contact"]))
    print(f"  Contact frames (dist<{CONTACT_DIST_M*100:.0f}cm): {contact_tr} ({100*contact_tr/max(1,len(train_frames)):.1f}%)")
    print(f"  Contact_v2 proxy (dist<{CONTACT_V2_DIST_M*100:.0f}cm): {contact_v2_tr} ({100*contact_v2_tr/max(1,len(train_frames)):.1f}%)")

    meta = compute_norm_stats(train_frames)

    out = args.output_dir / "dataset_temporal.h5"
    print(f"\nWriting {out} …")
    write_hdf5(out, train_frames, val_frames, meta)
    size_mb = out.stat().st_size / 1e6
    print(f"Done. {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
