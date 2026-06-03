"""build_dataset_v2.py — Step 1 (v2): extract HOT3D frames with extended features.

Feature layout v2 (15 dims vs. 11 in v1):
  spatial (8): [dir_x, dir_y, dir_z, distance, wrist_qw, wrist_qx, wrist_qy, wrist_qz]
  object  (7): [grip_oh(4), bbox_x, bbox_y, bbox_z]

The added wrist quaternion (4 dims) encodes hand orientation in world frame,
which determines pronation/supination and palm-facing direction — critical
for predicting correct grip shape beyond just approach direction + distance.

Run:
    python build_dataset_v2.py --data_dir ../data/quest3/ --output_dir ../data/v2/right/ --hand right
    python build_dataset_v2.py --data_dir ../data/quest3/ --output_dir ../data/v2/left/  --hand left

Output (output_dir/dataset_v2.h5) — same structure as v1 dataset.h5 but features are (N, 15).
  /train/features  (N_train, 15) float32
  /train/targets   (N_train, 22) float32
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
    zip_paths,
)

SPATIAL_DIM  = 8   # dir(3) + dist(1) + wrist_quat(4)
OBJECT_DIM   = 7   # grip_oh(4) + bbox(3)
FEATURE_DIM  = SPATIAL_DIM + OBJECT_DIM   # 15

MAX_DISTANCE    = 0.40   # metres — frames beyond this are skipped
GRIP_THRESHOLD  = 0.10   # metres — "grip" label below this

# Approach augmentation distances (same as v1)
_APPROACH_DISTANCES = [0.30, 0.50, 0.70, 1.00, 1.50, 2.50]
_D_CONTACT = 0.15
_D_OPEN    = 0.80


def _approach_blend(dist: float) -> float:
    t = np.clip((dist - _D_CONTACT) / (_D_OPEN - _D_CONTACT), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return float(1.0 - t)


def augment_approach_samples(grip_frames: list) -> list:
    """Synthesise open-hand pre-grasp frames at larger distances from grip-range seeds.
    Same logic as v1 but preserves the wrist quaternion from the seed frame.
    """
    neutral = np.zeros(22, dtype=np.float32)
    aug = []
    for f in grip_frames:
        direction = f["feature"][:3]
        wrist_quat = f["feature"][4:8]     # w,x,y,z — preserved at all distances
        grip_oh    = f["feature"][8:12]
        bbox       = f["feature"][12:15]
        real_tgt   = f["target"]

        for d in _APPROACH_DISTANCES:
            blend   = _approach_blend(d)
            feature = np.concatenate([direction, [d], wrist_quat, grip_oh, bbox]).astype(np.float32)
            target  = (blend * real_tgt + (1.0 - blend) * neutral).astype(np.float32)
            aug.append({
                "feature":  feature,
                "target":   target,
                "distance": np.float32(d),
                "label":    b"approach",
            })
    return aug


def parse_args():
    p = argparse.ArgumentParser(description="Build AuraXR v2 dataset (15-dim features).")
    p.add_argument("--data_dir",     required=True, type=Path)
    p.add_argument("--output_dir",   required=True, type=Path)
    p.add_argument("--hand",         default="right", choices=["right", "left"])
    p.add_argument("--max_distance", default=MAX_DISTANCE, type=float)
    p.add_argument("--val_frac",     default=0.15, type=float)
    p.add_argument("--seed",         default=42, type=int)
    p.add_argument("--no_augment",   action="store_true")
    return p.parse_args()


def extract_frames(seq_dir: Path, hand_key: str, max_distance: float) -> list[dict]:
    """Extract frames with 15-dim features (v2)."""
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

    for ts, hand_poses in ume_traj.items():
        if hand_key not in hand_poses:
            continue
        if ts not in obj_by_ts:
            continue

        pose         = hand_poses[hand_key]
        wrist_pos    = np.array(pose["wrist_xform"]["t_xyz"],   dtype=np.float32)
        wrist_q      = np.array(pose["wrist_xform"]["q_wxyz"],  dtype=np.float32)  # w,x,y,z
        joint_angles = np.array(pose["joint_angles"],           dtype=np.float32)

        # Find nearest object in this frame
        min_dist     = float("inf")
        best_feature = None
        best_target  = None

        for obj in obj_by_ts[ts]:
            bop_id = uid_to_bop.get(obj["object_uid"])
            if bop_id is None:
                continue

            obj_pos  = obj["pos_world"]
            delta    = obj_pos - wrist_pos
            rel_pos  = rotate_vec(quat_conjugate(wrist_q), delta)  # wrist frame
            dist     = float(np.linalg.norm(rel_pos))

            if dist < min_dist:
                min_dist = dist
                grip_oh, bbox = object_features(bop_id)
                direction     = rel_pos / (dist + 1e-8)

                # v2 feature: dir(3) + dist(1) + wrist_quat(4) + grip_oh(4) + bbox(3) = 15
                best_feature = np.concatenate([
                    direction,           # 3
                    [dist],              # 1
                    wrist_q,             # 4  — world-frame wrist orientation
                    grip_oh,             # 4
                    bbox,                # 3
                ]).astype(np.float32)
                best_target = joint_angles

        if best_feature is None or min_dist > max_distance:
            continue

        label = b"grip" if min_dist < GRIP_THRESHOLD else b"pre_shape"
        frames.append({
            "feature":  best_feature,
            "target":   best_target,
            "distance": np.float32(min_dist),
            "label":    label,
        })

    return frames


def compute_norm_stats(frames: list[dict]) -> dict:
    features = np.stack([f["feature"] for f in frames])
    targets  = np.stack([f["target"]  for f in frames])

    feat_mean = features.mean(axis=0)
    feat_std  = features.std(axis=0)
    tgt_mean  = targets.mean(axis=0)
    tgt_std   = targets.std(axis=0)

    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    tgt_std  = np.where(tgt_std  < 1e-8, 1.0, tgt_std)

    return {
        "feature_mean": feat_mean.tolist(),
        "feature_std":  feat_std.tolist(),
        "target_mean":  tgt_mean.tolist(),
        "target_std":   tgt_std.tolist(),
        "architecture": {
            "spatial_input_dim": SPATIAL_DIM,
            "object_input_dim":  OBJECT_DIM,
            "output_dim":        22,
            "hidden_dim":        256,
            "embedding_dim":     128,
            "version":           2,
        },
    }


def write_hdf5(output_path: Path, train_frames: list, val_frames: list, meta: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        hf.attrs["meta"] = json.dumps(meta)
        for split_name, frames in [("train", train_frames), ("val", val_frames)]:
            grp = hf.create_group(split_name)
            grp.create_dataset("features",  data=np.stack([f["feature"]  for f in frames]), compression="gzip")
            grp.create_dataset("targets",   data=np.stack([f["target"]   for f in frames]), compression="gzip")
            grp.create_dataset("distances", data=np.array([f["distance"] for f in frames], dtype=np.float32), compression="gzip")
            dt = h5py.string_dtype()
            grp.create_dataset("labels",    data=np.array([f["label"] for f in frames], dtype=dt))


def main():
    args = parse_args()
    hand_key = HAND_KEY[args.hand]
    print(f"build_dataset_v2: hand={args.hand}  feature_dim={FEATURE_DIM}  max_distance={args.max_distance}m")

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

    if not args.no_augment:
        seeds_tr = [f for f in train_frames if f["distance"] < 0.15]
        seeds_va = [f for f in val_frames   if f["distance"] < 0.15]
        aug_tr   = augment_approach_samples(seeds_tr)
        aug_va   = augment_approach_samples(seeds_va)
        train_frames.extend(aug_tr)
        val_frames.extend(aug_va)
        print(f"Approach aug: +{len(aug_tr)} train, +{len(aug_va)} val")

    meta = compute_norm_stats(train_frames)

    output_path = args.output_dir / "dataset_v2.h5"
    print(f"\nWriting {output_path} …")
    write_hdf5(output_path, train_frames, val_frames, meta)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Done. {size_mb:.1f} MB")
    print(f"\nFeature stats (first 8 dims = spatial):")
    for i, (m, s) in enumerate(zip(meta["feature_mean"][:8], meta["feature_std"][:8])):
        label = ["dir_x","dir_y","dir_z","dist","qw","qx","qy","qz"][i]
        print(f"  [{i}] {label:6s}  mean={m:+.4f}  std={s:.4f}")


if __name__ == "__main__":
    main()
