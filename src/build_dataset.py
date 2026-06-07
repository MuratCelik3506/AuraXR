"""build_dataset.py — extract HOT3D frames and build the training dataset.

Feature layout (15 dims):
  spatial (8): [dir_world(3), dir_obj_local(3), dist(1), approach_speed(1)]
  object  (7): [grip_oh(4), bbox_x, bbox_y, bbox_z]

Target layout:
  /targets       (N, 22): UME joint angles (radians)
  /wrist_rot_6d  (N,  6): wrist rotation as 6D continuous representation
                           — first two columns of rotation matrix of q_rel,
                             where q_rel = canonical^{-1} ⊗ q_wrist (Unity frame)

Design notes:
  - hand_confidence < 0.70 → frame discarded (noisy tracking).
  - dir_world: world-frame unit vector (NOT wrist-local: canonical frame → always (0,0,1);
    real wrist quat → HOT3D/Unity tracking system mismatch).
  - dir_obj_local: delta rotated into object-local frame via q_wo.
  - approach_speed: dot(wrist_velocity, dir_world) — positions only, no rotation risk.
  - wrist_rot_6d: encodes palm orientation relative to approach direction in Unity frame.
  - ume_traj is an unordered dict → sorted() mandatory for velocity computation.

Run:
    python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/right/ --hand right
    python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/left/  --hand left

Output (output_dir/dataset.h5):
  /train/features     (N_train, 15) float32
  /train/targets      (N_train, 22) float32
  /train/wrist_rot_6d (N_train,  6) float32
  ...
  attrs["meta"]    JSON with norm stats + architecture config
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from grip_categories import OBJ_GRIP, OBJ_BBOX, object_features
from hot3d_utils import (
    HAND_KEY,
    build_uid_to_bop,
    find_sequences,
    quat_conjugate,
    read_dynamic_objects,
    read_metadata,
    read_umetrack_trajectory,
    rotate_vec,
    wrist_rot_to_6d,
    zip_paths,
)

SPATIAL_DIM  = 8   # dir_world(3) + dir_obj_local(3) + dist(1) + approach_speed(1)
OBJECT_DIM   = 7   # grip_oh(4) + bbox(3)
FEATURE_DIM  = SPATIAL_DIM + OBJECT_DIM   # 15

MAX_DISTANCE    = 0.40   # metres — frames beyond this are skipped
GRIP_THRESHOLD  = 0.10   # metres — "grip" label below this
GRIP_OVERSAMPLE = 10     # grip frames repeated this many times to balance training data
MIN_CONFIDENCE  = 0.70   # hand_confidence threshold — frames below are discarded


def parse_args():
    p = argparse.ArgumentParser(description="Build AuraXR dataset (15-dim features).")
    p.add_argument("--data_dir",     required=True, type=Path)
    p.add_argument("--output_dir",   required=True, type=Path)
    p.add_argument("--hand",         default="right", choices=["right", "left"])
    p.add_argument("--max_distance", default=MAX_DISTANCE, type=float)
    p.add_argument("--val_frac",     default=0.15, type=float)
    p.add_argument("--seed",         default=42, type=int)
    p.add_argument("--oversample",    default=GRIP_OVERSAMPLE, type=int,
                   help="Grip frame repeat factor for training (default: 10).")
    return p.parse_args()


def extract_frames(seq_dir: Path, hand_key: str, max_distance: float) -> list[dict]:
    """Extract frames with 15-dim features: world-frame + object-local dir + approach speed."""
    hand_zip, gt_zip = zip_paths(seq_dir)
    if hand_zip is None:
        return []

    try:
        ume_traj   = read_umetrack_trajectory(hand_zip)
        obj_by_ts  = read_dynamic_objects(gt_zip)
        metadata   = read_metadata(gt_zip)
    except Exception as e:
        print(f"    [SKIP] {seq_dir.name}: {e}")
        return []

    uid_to_bop = build_uid_to_bop(metadata)
    frames = []
    prev_wrist_pos: np.ndarray | None = None
    prev_ts: int | None = None

    for ts, hand_poses in sorted(ume_traj.items()):
        if hand_key not in hand_poses:
            continue
        if ts not in obj_by_ts:
            continue

        pose = hand_poses[hand_key]

        # Aşama 1.1: discard low-quality tracking frames
        if pose.get("hand_confidence", 1.0) < MIN_CONFIDENCE:
            prev_wrist_pos = None  # break velocity continuity across bad frames
            prev_ts = None
            continue

        wrist_pos    = np.array(pose["wrist_xform"]["t_xyz"],   dtype=np.float32)
        wrist_q_wxyz = np.array(pose["wrist_xform"]["q_wxyz"],  dtype=np.float32)
        joint_angles = np.array(pose["joint_angles"],           dtype=np.float32)

        # Aşama 3.1: wrist velocity for approach_speed (positions only, no rotation)
        if prev_wrist_pos is not None and prev_ts is not None:
            dt = (ts - prev_ts) * 1e-9
            vel_world = (wrist_pos - prev_wrist_pos) / dt if dt > 0 else np.zeros(3, dtype=np.float32)
        else:
            vel_world = np.zeros(3, dtype=np.float32)
        prev_wrist_pos = wrist_pos
        prev_ts = ts

        # Find nearest object in this frame
        min_dist       = float("inf")
        best_feature   = None
        best_target    = None
        best_direction = None  # saved for wrist_rot_6d computation

        for obj in obj_by_ts[ts]:
            bop_id = uid_to_bop.get(obj["object_uid"])
            if bop_id is None:
                continue

            obj_pos = obj["pos_world"]
            delta   = obj_pos - wrist_pos
            dist    = float(np.linalg.norm(delta))

            if dist < min_dist:
                min_dist      = dist
                grip_oh, bbox = object_features(bop_id)
                direction     = delta / (dist + 1e-8)   # world-frame unit vector (kept as-is)

                # Aşama 2.2: object-local direction — which face of the object is approached
                q_obj_inv     = quat_conjugate(obj["quat_world"])
                dir_obj_local = rotate_vec(q_obj_inv, delta / (dist + 1e-8))

                # Aşama 3.1: project wrist velocity onto approach direction
                approach_speed = float(np.dot(vel_world, direction))

                # feature: dir_world(3) + dir_obj_local(3) + dist(1) + approach_speed(1) + grip_oh(4) + bbox(3) = 15
                best_feature = np.concatenate([
                    direction,        # 3  world-frame (unchanged)
                    dir_obj_local,    # 3  object-local (NEW)
                    [dist],           # 1
                    [approach_speed], # 1  (NEW)
                    grip_oh,          # 4
                    bbox,             # 3
                ]).astype(np.float32)
                best_target    = joint_angles
                best_direction = direction

        if best_feature is None or min_dist > max_distance:
            continue

        # Wrist rotation in 6D (Unity-frame, relative to approach direction)
        rot6d = wrist_rot_to_6d(wrist_q_wxyz, best_direction)

        label = b"grip" if min_dist < GRIP_THRESHOLD else b"pre_shape"
        frames.append({
            "feature":      best_feature,
            "target":       best_target,
            "wrist_rot_6d": rot6d,
            "distance":     np.float32(min_dist),
            "label":        label,
        })

    return frames


def compute_norm_stats(frames: list[dict]) -> dict:
    features   = np.stack([f["feature"]      for f in frames])
    targets    = np.stack([f["target"]       for f in frames])
    wrist_rots = np.stack([f["wrist_rot_6d"] for f in frames])

    feat_mean = features.mean(axis=0)
    feat_std  = features.std(axis=0)
    tgt_mean  = targets.mean(axis=0)
    tgt_std   = targets.std(axis=0)
    rot_mean  = wrist_rots.mean(axis=0)
    rot_std   = wrist_rots.std(axis=0)

    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    tgt_std  = np.where(tgt_std  < 1e-8, 1.0, tgt_std)
    rot_std  = np.where(rot_std  < 1e-8, 1.0, rot_std)

    return {
        "feature_mean":    feat_mean.tolist(),
        "feature_std":     feat_std.tolist(),
        "target_mean":     tgt_mean.tolist(),
        "target_std":      tgt_std.tolist(),
        "wrist_rot_mean":  rot_mean.tolist(),
        "wrist_rot_std":   rot_std.tolist(),
        "architecture": {
            "spatial_input_dim": SPATIAL_DIM,
            "object_input_dim":  OBJECT_DIM,
            "output_dim":        22,
            "hidden_dim":        256,
            "embedding_dim":     128,
            "version":           4,
        },
    }


def write_hdf5(output_path: Path, train_frames: list, val_frames: list, meta: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        hf.attrs["meta"] = json.dumps(meta)
        for split_name, frames in [("train", train_frames), ("val", val_frames)]:
            grp = hf.create_group(split_name)
            grp.create_dataset("features",     data=np.stack([f["feature"]      for f in frames]), compression="gzip")
            grp.create_dataset("targets",      data=np.stack([f["target"]       for f in frames]), compression="gzip")
            grp.create_dataset("wrist_rot_6d", data=np.stack([f["wrist_rot_6d"] for f in frames]), compression="gzip")
            grp.create_dataset("distances",    data=np.array([f["distance"]     for f in frames], dtype=np.float32), compression="gzip")
            dt = h5py.string_dtype()
            grp.create_dataset("labels",       data=np.array([f["label"] for f in frames], dtype=dt))


def main():
    args = parse_args()
    hand_key = HAND_KEY[args.hand]
    print(f"build_dataset: hand={args.hand}  feature_dim={FEATURE_DIM}  max_distance={args.max_distance}m")

    sequences = find_sequences(args.data_dir, split="train")
    if not sequences:
        print(f"[ERROR] No sequences found in {args.data_dir}/train/")
        return

    rng = random.Random(args.seed)
    shuffled = sequences.copy()
    rng.shuffle(shuffled)
    n_val       = max(1, int(len(shuffled) * args.val_frac))
    val_seqs    = set(str(s) for s in shuffled[:n_val])
    train_seqs  = set(str(s) for s in shuffled[n_val:])
    print(f"Sequences: {len(train_seqs)} train, {len(val_seqs)} val")

    train_frames: list[dict] = []
    val_frames:   list[dict] = []
    skipped = 0

    for seq_dir in tqdm(sequences, desc="Sequences"):
        frames = extract_frames(seq_dir, hand_key, args.max_distance)
        if not frames:
            skipped += 1
            continue
        if str(seq_dir) in val_seqs:
            val_frames.extend(frames)
        else:
            train_frames.extend(frames)

    print(f"\nExtracted: {len(train_frames)} train, {len(val_frames)} val  (skipped {skipped} seqs)")

    if not train_frames:
        print("[ERROR] No training frames. Check data_dir.")
        return

    grip_tr = sum(1 for f in train_frames if f["label"] == b"grip")
    grip_va = sum(1 for f in val_frames   if f["label"] == b"grip")
    print(f"Train — grip:{grip_tr}  pre_shape:{len(train_frames)-grip_tr}")
    print(f"Val   — grip:{grip_va}  pre_shape:{len(val_frames)-grip_va}")

    # Compute norm stats before oversampling to avoid bias from repeated grip frames
    meta = compute_norm_stats(train_frames)

    # Oversample grip frames in training split to balance class distribution
    grip_seeds_tr = [f for f in train_frames if f["label"] == b"grip"]
    train_frames.extend(grip_seeds_tr * (args.oversample - 1))
    print(f"Grip oversample ×{args.oversample}: {grip_tr} → {grip_tr * args.oversample} grip frames  total={len(train_frames)}")

    output_path = args.output_dir / "dataset.h5"
    print(f"\nWriting {output_path} …")
    write_hdf5(output_path, train_frames, val_frames, meta)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Done. {size_mb:.1f} MB")
    print(f"\nFeature stats (8 spatial dims):")
    labels = ["dir_x","dir_y","dir_z","obj_x","obj_y","obj_z","dist","approach_spd"]
    for i, (m, s) in enumerate(zip(meta["feature_mean"][:8], meta["feature_std"][:8])):
        print(f"  [{i}] {labels[i]:14s}  mean={m:+.4f}  std={s:.4f}")


if __name__ == "__main__":
    main()
