"""build_dataset.py — Step 1: extract HOT3D frames and write dataset.h5.

Run:
    python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/right/ --hand right
    python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/left/  --hand left

Feature vector (11 dims):
    [dir_x, dir_y, dir_z, grip_oh(4), bbox(3), distance(1)]
    dir = rel_pos / distance — unit approach-direction vector in wrist frame.
    Separating direction from distance helps the model learn pose from approach angle.

Output (output_dir/dataset.h5):
    /train/features   (N_train, 11) float32
    /train/targets    (N_train, 22) float32
    /train/labels     (N_train,)    bytes  (b"pre_shape" or b"grip")
    /train/distances  (N_train,)    float32
    /val/...          same structure
    attrs["meta"]     JSON string with norm stats + architecture config
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

MAX_DISTANCE = 0.40    # meters — real HOT3D frames beyond this are skipped
GRIP_THRESHOLD = 0.10  # meters — below this → "grip" label

# ── Approach augmentation ────────────────────────────────────────────────────
# Synthesise pre-grasp frames from grip-range samples.
# Hand is open (near-zero angles) far away and closes smoothly as it approaches.
_APPROACH_DISTANCES = [0.30, 0.50, 0.70, 1.00, 1.50, 2.50]  # metres
_D_CONTACT = 0.15   # full grip pose at or below this
_D_OPEN    = 0.80   # fully open hand at or above this


def _approach_blend(dist: float) -> float:
    """Smoothstep blend: 1.0 (grip) at dist ≤ D_CONTACT → 0.0 (open) at dist ≥ D_OPEN."""
    t = np.clip((dist - _D_CONTACT) / (_D_OPEN - _D_CONTACT), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)   # smoothstep
    return float(1.0 - t)


def augment_approach_samples(grip_frames: list) -> list:
    """For every grip frame generate synthetic approach samples at larger distances.

    At large distances the target is the neutral open-hand pose (all zeros).
    At close distances it is the real grip pose.  Between them: smoothstep blend.
    Only grip frames are used as seeds — they have the most defined curl pose.
    """
    neutral = np.zeros(22, dtype=np.float32)
    aug = []
    for f in grip_frames:
        direction = f["feature"][:3]    # unit vector in wrist frame
        grip_oh   = f["feature"][4:8]
        bbox      = f["feature"][8:11]
        real_tgt  = f["target"]

        for d in _APPROACH_DISTANCES:
            blend   = _approach_blend(d)
            feature = np.concatenate([direction, [d], grip_oh, bbox]).astype(np.float32)
            target  = (blend * real_tgt + (1.0 - blend) * neutral).astype(np.float32)
            aug.append({
                "feature":  feature,
                "target":   target,
                "distance": np.float32(d),
                "label":    b"approach",
            })
    return aug


def parse_args():
    p = argparse.ArgumentParser(description="Build AuraXR training dataset from HOT3D Quest3 ZIPs.")
    p.add_argument("--data_dir",    required=True, type=Path,
                   help="Path to data/quest3/ containing train/ and test/ subdirectories.")
    p.add_argument("--output_dir",  required=True, type=Path,
                   help="Directory to write dataset.h5.")
    p.add_argument("--hand",        default="right", choices=["right", "left"],
                   help="Which hand to extract.")
    p.add_argument("--max_distance", default=MAX_DISTANCE, type=float,
                   help="Skip frames with hand-object distance above this (meters).")
    p.add_argument("--val_frac",    default=0.15, type=float,
                   help="Fraction of train sequences to hold out for validation.")
    p.add_argument("--seed",        default=42, type=int,
                   help="Random seed for train/val sequence split.")
    p.add_argument("--no_augment",  action="store_true",
                   help="Disable approach-phase augmentation (for ablation).")
    return p.parse_args()


def extract_frames(seq_dir: Path, hand_key: str, max_distance: float) -> list[dict]:
    """Extract all usable frames from one sequence.

    Returns list of dicts:
        feature (11,), target (22,), distance (float), label (str)
    """
    hand_zip, gt_zip = zip_paths(seq_dir)
    if hand_zip is None:
        return []

    try:
        ume_traj = read_umetrack_trajectory(hand_zip)
        obj_by_ts = read_dynamic_objects(gt_zip)
        metadata = read_metadata(gt_zip)
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

        pose = hand_poses[hand_key]
        wrist_pos = np.array(pose["wrist_xform"]["t_xyz"], dtype=np.float32)
        wrist_q   = np.array(pose["wrist_xform"]["q_wxyz"], dtype=np.float32)
        joint_angles = np.array(pose["joint_angles"], dtype=np.float32)

        # Find nearest object in this frame
        min_dist = float("inf")
        best_feature = None
        best_target  = None

        for obj in obj_by_ts[ts]:
            bop_id = uid_to_bop.get(obj["object_uid"])
            if bop_id is None:
                continue

            obj_pos = obj["pos_world"]
            delta = obj_pos - wrist_pos
            rel_pos = rotate_vec(quat_conjugate(wrist_q), delta)  # wrist frame
            dist = float(np.linalg.norm(rel_pos))

            if dist < min_dist:
                min_dist = dist
                grip_oh, bbox = object_features(bop_id)
                direction = rel_pos / (dist + 1e-8)  # unit approach-direction in wrist frame
                best_feature = np.concatenate([direction, [dist], grip_oh, bbox]).astype(np.float32)
                best_target  = joint_angles

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

    # Avoid division by zero for constant features (e.g. one-hot dims with single class)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    tgt_std  = np.where(tgt_std  < 1e-8, 1.0, tgt_std)

    return {
        "feature_mean": feat_mean.tolist(),
        "feature_std":  feat_std.tolist(),
        "target_mean":  tgt_mean.tolist(),
        "target_std":   tgt_std.tolist(),
        "architecture": {
            "spatial_input_dim": 4,
            "object_input_dim":  7,
            "output_dim":        22,
            "hidden_dim":        64,
            "embedding_dim":     32,
        },
    }


def write_hdf5(output_path: Path, train_frames: list[dict], val_frames: list[dict], meta: dict):
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
    print(f"Building dataset: hand={args.hand}  max_distance={args.max_distance}m")

    sequences = find_sequences(args.data_dir, split="train")
    if not sequences:
        print(f"[ERROR] No sequences found in {args.data_dir}/train/")
        return

    # Reproducible train/val split by sequence
    rng = random.Random(args.seed)
    shuffled = sequences.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_seqs  = set(str(s) for s in shuffled[:n_val])
    train_seqs = set(str(s) for s in shuffled[n_val:])

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

    print(f"\nExtracted: {len(train_frames)} train frames, {len(val_frames)} val frames")
    print(f"Skipped sequences: {skipped}")

    if not train_frames:
        print("[ERROR] No training frames extracted. Check data_dir path.")
        return

    # Count grip/pre_shape
    grip_train = sum(1 for f in train_frames if f["label"] == b"grip")
    print(f"Train — grip: {grip_train}  pre_shape: {len(train_frames) - grip_train}")
    grip_val = sum(1 for f in val_frames if f["label"] == b"grip")
    print(f"Val   — grip: {grip_val}  pre_shape: {len(val_frames) - grip_val}")

    # ── Approach augmentation ────────────────────────────────────────────────
    if not args.no_augment:
        grip_seed_tr = [f for f in train_frames if f["distance"] < 0.15]
        grip_seed_va = [f for f in val_frames   if f["distance"] < 0.15]
        aug_tr = augment_approach_samples(grip_seed_tr)
        aug_va = augment_approach_samples(grip_seed_va)
        train_frames.extend(aug_tr)
        val_frames.extend(aug_va)
        print(f"Approach aug  — +{len(aug_tr)} train, +{len(aug_va)} val "
              f"(seeds: {len(grip_seed_tr)} grip frames × {len(_APPROACH_DISTANCES)} distances)")
    else:
        print("Approach augmentation disabled (--no_augment).")

    meta = compute_norm_stats(train_frames)

    output_path = args.output_dir / "dataset.h5"
    print(f"\nWriting {output_path} …")
    write_hdf5(output_path, train_frames, val_frames, meta)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
