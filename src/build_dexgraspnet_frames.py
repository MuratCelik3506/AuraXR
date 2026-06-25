"""build_dexgraspnet_frames.py — Generate synthetic contact frames from DexGraspNet grasps.

DexGraspNet provides 1.32M grasps over 5355 objects.  Each grasp contains a
MANO hand pose at the moment of contact.  We synthesise HOT3D-compatible feature
vectors for each valid grasp and save them in an HDF5 file that can be merged
with dataset_mano.h5 (the HOT3D split) during training.

Key design choices:
  - distance = 0.02 m (2 cm, at-contact)
  - approach_speed = 0.0 (stationary at contact — the model sees this as a
    contact trigger, not an approach frame)
  - approach direction: random unit vector per grasp (we have no ground-truth
    camera/wrist trajectory; randomisation makes the model invariant to approach dir)
  - wrist_rot_6d: identity (first two columns of R=I); same reasoning
  - BOP object ID: mapped from DexGraspNet obj_id via modulo over HOT3D IDs, so
    the correct grip category and bbox are preserved on average
  - SDF feature: zeros (wrist near surface → SDF ≈ 0; gradient direction
    already encoded in approach dir which we randomise anyway)
  - betas: zeros (DexGraspNet does not provide MANO shape parameters)

The only information we copy verbatim from DexGraspNet is the MANO PCA pose
target (15-dim) — this is the augmentation signal that improves contact-frame
coverage (from 7.7% HOT3D → higher ratio after merge).

Input:
  data/dexgraspnet/grasps_mano15.npz  — output of convert_dexgraspnet.py
    "pose_pca" (M, 15), "obj_id" (M,), "valid" (M,), "hand" (M,)

Output:
  data/dexgraspnet/dex_contact_frames.h5
    Same schema as dataset_mano.h5 (train/val splits, same dataset keys)

Run:
    .venv/bin/python3 src/build_dexgraspnet_frames.py \\
        --grasps data/dexgraspnet/grasps_mano15.npz \\
        --out    data/dexgraspnet/dex_contact_frames.h5 \\
        --hand   right \\
        --max_frames 50000 \\
        --val_frac 0.15 \\
        --seed 42

Merge with HOT3D dataset at training time (DataLoader-level concat), or
pre-merge with the --merge flag:
    .venv/bin/python3 src/build_dexgraspnet_frames.py \\
        --grasps data/dexgraspnet/grasps_mano15.npz \\
        --out    data/dexgraspnet/dex_contact_frames.h5 \\
        --merge  data/processed/hot3d_mano/right/dataset_mano.h5 \\
        --merge_out data/processed/hot3d_mano/right/dataset_mano_aug.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from grip_categories import object_features, OBJ_INFO

# All HOT3D BOP IDs in a stable sorted list
_HOT3D_BOP_IDS = sorted(OBJ_INFO.keys())   # [1, 2, ..., 33]

# Synthetic sequence offset — keeps seq_ids distinct from HOT3D sequences
_DEX_SEQ_OFFSET = 9000
_FRAMES_PER_SEQ = 500   # frames grouped into synthetic sequences of this length


def _bop_id_for_dex_obj(dex_obj_id: int) -> int:
    """Map DexGraspNet object index (0-based) → HOT3D BOP ID.

    Uses modulo so all HOT3D objects appear equally often across augmentation.
    """
    return _HOT3D_BOP_IDS[int(dex_obj_id) % len(_HOT3D_BOP_IDS)]


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n uniform random unit vectors on S^2."""
    v = rng.standard_normal((n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    return v


def _identity_wrist_rot_6d() -> np.ndarray:
    """6D rotation for identity matrix: first two columns of R=I."""
    return np.array([1.0, 0.0, 0.0,  0.0, 1.0, 0.0], dtype=np.float32)


def build_contact_frames(
    pose_pca: np.ndarray,   # (M, 15)
    obj_ids_dex: np.ndarray, # (M,) DexGraspNet obj indices
    rng: np.random.Generator,
    start_seq_id: int = _DEX_SEQ_OFFSET,
) -> list[dict]:
    """Convert (M, 15) MANO PCA poses → list of frame dicts."""
    M = len(pose_pca)
    dir_world = _random_unit_vectors(M, rng)       # (M, 3)
    wrist_6d  = _identity_wrist_rot_6d()           # (6,) constant
    sdf_feat  = np.zeros(4, dtype=np.float32)      # (4,) zeros at contact
    betas     = np.zeros(10, dtype=np.float32)     # (10,) unknown

    CONTACT_DIST = np.float32(0.02)

    frames: list[dict] = []
    for i in range(M):
        bop_id      = _bop_id_for_dex_obj(int(obj_ids_dex[i]))
        grip_oh, bbox = object_features(bop_id)
        d           = dir_world[i]

        feature = np.concatenate([
            d,                                           # [0-2]  approach dir world
            d,                                           # [3-5]  dir in obj frame (≈ same, obj at origin)
            [CONTACT_DIST],                              # [6]    distance
            [0.0],                                       # [7]    approach speed
            [0.0, 0.0, 0.0],                             # [8-10] obj velocity
            wrist_6d,                                    # [11-16] wrist 6D
            [1.0],                                       # [17]  hand confidence
            grip_oh,                                     # [18-21] grip one-hot
            bbox,                                        # [22-24] bbox half-extents
        ]).astype(np.float32)

        seq_id     = np.int32(start_seq_id + i // _FRAMES_PER_SEQ)
        frame_idx  = np.int32(i % _FRAMES_PER_SEQ)

        frames.append({
            "feature":      feature,
            "sdf_feature":  sdf_feat.copy(),
            "obj_id":       np.int32(bop_id),
            "target":       pose_pca[i].astype(np.float32),   # (15,)
            "betas":        betas.copy(),
            "wrist_rot_6d": wrist_6d.copy(),
            "distance":     CONTACT_DIST,
            "seq_id":       seq_id,
            "frame_index":  frame_idx,
            "is_mirror":    np.uint8(0),
            "contact":      np.uint8(1),
        })

    return frames


def write_hdf5(out_path: Path, train_frames: list[dict], val_frames: list[dict], meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        hf.attrs["meta"] = json.dumps(meta)
        for split, frames in [("train", train_frames), ("val", val_frames)]:
            g  = hf.create_group(split)
            kw = dict(compression="gzip")
            g.create_dataset("features",     data=np.stack([f["feature"]      for f in frames]), **kw)
            g.create_dataset("sdf_features", data=np.stack([f["sdf_feature"]  for f in frames]), **kw)
            g.create_dataset("obj_id",       data=np.array([f["obj_id"]       for f in frames], np.int32),   **kw)
            g.create_dataset("targets",      data=np.stack([f["target"]       for f in frames]), **kw)
            g.create_dataset("betas",        data=np.stack([f["betas"]        for f in frames]), **kw)
            g.create_dataset("wrist_rot_6d", data=np.stack([f["wrist_rot_6d"] for f in frames]), **kw)
            g.create_dataset("distances",    data=np.array([f["distance"]     for f in frames], np.float32), **kw)
            g.create_dataset("sequence_id",  data=np.array([f["seq_id"]       for f in frames], np.int32),  **kw)
            g.create_dataset("frame_index",  data=np.array([f["frame_index"]  for f in frames], np.int32),  **kw)
            g.create_dataset("is_mirror",    data=np.array([f["is_mirror"]    for f in frames], np.uint8),  **kw)
            g.create_dataset("contact",      data=np.array([f["contact"]      for f in frames], np.uint8),  **kw)
    print(f"Saved {out_path}  train={len(train_frames)}  val={len(val_frames)}")


def merge_hdf5(base_path: Path, aug_path: Path, out_path: Path):
    """Concatenate aug HDF5 (contact-only) into base HDF5 (HOT3D dataset).

    Both files must have identical dataset keys under train/ and val/ groups.
    The merged file reuses the base meta (norm stats), since augmentation frames
    are contact-only and would skew the statistics if included.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(base_path, "r") as base, \
         h5py.File(aug_path,  "r") as aug,  \
         h5py.File(out_path,  "w") as out:

        out.attrs["meta"] = base.attrs["meta"]   # keep HOT3D norm stats

        for split in ("train", "val"):
            bg = base[split]
            ag = aug[split]
            og = out.create_group(split)
            kw = dict(compression="gzip")
            for key in bg.keys():
                b_arr = bg[key][:]
                a_arr = ag[key][:]
                merged = np.concatenate([b_arr, a_arr], axis=0)
                og.create_dataset(key, data=merged, **kw)
                print(f"  {split}/{key}: {len(b_arr)} + {len(a_arr)} = {len(merged)}")

    print(f"Merged → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasps",    required=True, type=Path,
                   help="data/dexgraspnet/grasps_mano15.npz")
    p.add_argument("--out",       required=True, type=Path,
                   help="Output HDF5 for DexGraspNet contact frames")
    p.add_argument("--hand",      default="right", choices=["left", "right"])
    p.add_argument("--max_frames",default=50000, type=int,
                   help="Max contact frames to include (0=all valid)")
    p.add_argument("--val_frac",  default=0.15, type=float)
    p.add_argument("--seed",      default=42, type=int)
    p.add_argument("--merge",     default=None, type=Path,
                   help="Optional: base dataset_mano.h5 to merge into")
    p.add_argument("--merge_out", default=None, type=Path,
                   help="Output path for merged HDF5")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load converted grasps
    if not args.grasps.exists():
        print(f"[ERROR] Grasps file not found: {args.grasps}")
        print("  Run: python src/convert_dexgraspnet.py first")
        return

    data = np.load(args.grasps, allow_pickle=True)
    pose_pca  = data["pose_pca"]    # (M, 15)
    obj_ids   = data["obj_id"]      # (M,)
    valid     = data["valid"]       # (M,)
    hand_arr  = data["hand"]        # (M,) dtype bytes

    # Filter by hand side and validity
    hand_bytes = args.hand.encode()
    hand_mask  = np.array([h == hand_bytes or h == b"both" for h in hand_arr])
    mask       = valid & hand_mask
    pose_pca   = pose_pca[mask]
    obj_ids    = obj_ids[mask]
    print(f"Valid {args.hand}-hand grasps: {mask.sum()} / {len(valid)}")

    # Subsample if needed
    if args.max_frames > 0 and len(pose_pca) > args.max_frames:
        idx      = rng.choice(len(pose_pca), args.max_frames, replace=False)
        pose_pca = pose_pca[idx]
        obj_ids  = obj_ids[idx]
        print(f"Subsampled to {len(pose_pca)} frames (--max_frames {args.max_frames})")

    frames = build_contact_frames(pose_pca, obj_ids, rng)
    print(f"Built {len(frames)} contact frames  (contact=100%)")

    # Train / val split (by synthetic sequence, not by frame, to avoid leakage)
    seq_ids = sorted({int(f["seq_id"]) for f in frames})
    rng.shuffle(seq_ids := np.array(seq_ids))
    n_val   = max(1, int(len(seq_ids) * args.val_frac))
    val_set = set(seq_ids[:n_val].tolist())

    train_frames = [f for f in frames if int(f["seq_id"]) not in val_set]
    val_frames   = [f for f in frames if int(f["seq_id"]) in val_set]
    print(f"Split: train={len(train_frames)}  val={len(val_frames)}")
    print(f"PCA range: [{pose_pca.min():.3f}, {pose_pca.max():.3f}]")

    meta = {
        "source":       "dexgraspnet_contact_frames",
        "hand":         args.hand,
        "total_frames": len(frames),
        "max_frames":   args.max_frames,
        "contact_frac": 1.0,
        "architecture": {
            "input_dim":    25,
            "sdf_input_dim": 4,
            "output_dim":   15,
            "output_type":  "mano_pca",
        },
    }

    write_hdf5(args.out, train_frames, val_frames, meta)

    if args.merge is not None:
        merge_out = args.merge_out or args.merge.parent / (args.merge.stem + "_aug.h5")
        merge_hdf5(args.merge, args.out, merge_out)


if __name__ == "__main__":
    main()
