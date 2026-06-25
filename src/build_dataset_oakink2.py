"""build_dataset_oakink2.py — OakInk2 anno_preview PKLs → HOT3D-compatible HDF5.

OakInk2 annotation format (anno_preview/<seq>.pkl):
  raw_mano: dict[frame_idx → {
    rh__pose_coeffs: (1, 16, 4)  quaternion (w,x,y,z), joint0=wrist, 1-15=fingers
    lh__pose_coeffs: (1, 16, 4)
    rh__tsl:  (1, 3)   wrist world translation
    lh__tsl:  (1, 3)
    rh__betas: (1, 10)  MANO shape params
    lh__betas: (1, 10)
  }]
  frame_id_list: list[int]  — ordered frame indices
  obj_list: list[str]       — object IDs in scene (O02@XXXX@YYYY format)

Split: actor-based. Actors A001-A006 → train; A007-A008 and O-actors → val.

Output: data/processed/oakink2_mano/{left,right}/dataset_mano.h5
  Same schema as hot3d_mano — directly loadable by TemporalWindowDataset.
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from grip_categories import object_features

# Val actors (last ~20% by actor ID)
VAL_ACTORS = {"A007", "A008", "O001", "O003"}

# OakInk2 uses generic household/tool objects; map to nearest HOT3D BOP grip type.
# Without exact object name list, we use power-grip can as default.
# Known partial mappings based on OakInk2 paper object categories:
OAKINK2_TO_BOP: dict[str, int] = {
    "O02@0010": 10,   # cylindrical can → can_soup (POWER)
    "O02@0011": 10,   # cylindrical → can_soup
    "O02@0013": 21,   # fruit-like → food_vegetables
    "O02@0015": 13,   # bottle-like → bottle_mustard
    "O02@0017": 8,    # mug-like → mug_patterned (POWER)
    "O02@0018": 8,    # cup/mug → mug_patterned
    "O02@0019": 7,    # pot-like → coffee_pot
    "O02@0020": 13,   # bottle → bottle_mustard
    "O02@0025": 8,    # cup → mug_patterned
    "O02@0029": 3,    # plate/flat → plate_bamboo
    "O02@0030": 32,   # tool/thin → whiteboard_marker (PRECISION)
    "O02@0031": 6,    # spatula-like → spatula_red
    "O02@0032": 32,   # pen/marker → whiteboard_marker
    "O02@0033": 6,    # tool → spatula_red
    "O02@0035": 17,   # box/carton → carton_milk
    "O02@0037": 32,   # scissors → whiteboard_marker
    "O02@0038": 26,   # box → birdhouse_toy
    "O02@0039": 26,   # box → birdhouse_toy
    "O02@0044": 20,   # flat tool → food_waffles
    "O02@0045": 21,   # food-like → food_vegetables
    "O02@0047": 15,   # bottle → bottle_ranch
    "O02@0048": 15,   # bottle → bottle_ranch
    "O02@0049": 13,   # bottle → bottle_mustard
    "O02@0053": 10,   # can → can_soup
    "O02@0054": 10,   # can → can_soup
    "O02@0055": 10,   # can → can_soup
    "O02@0056": 7,    # appliance → coffee_pot
    "O02@0080": 24,   # phone → cellphone
    "O02@0081": 24,   # phone → cellphone
    "O02@0090": 28,   # flat/keyboard → keyboard
}
DEFAULT_BOP = 10  # can_soup (POWER grip fallback)

MANO_POSE_DIM  = 15
MANO_BETAS_DIM = 10
SDF_FEATURE_DIM = 4
FEATURE_DIM    = 25


def quat_wxyz_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert (w,x,y,z) quaternion to axis-angle (3,)."""
    from scipy.spatial.transform import Rotation
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy: (x,y,z,w)
    return r.as_rotvec().astype(np.float32)


def pose_coeffs_to_finger_aa(pose_coeffs: np.ndarray) -> np.ndarray:
    """Convert (16, 4) quaternion joints to 45-dim finger axis-angle.

    Joint 0 = global wrist (skipped here).
    Joints 1-15 = 15 finger joints → 45 values.
    """
    aa_list = []
    for j in range(1, 16):
        aa_list.append(quat_wxyz_to_rotvec(pose_coeffs[j]))
    return np.concatenate(aa_list).astype(np.float32)  # (45,)


def pose_coeffs_to_wrist_6d(pose_coeffs: np.ndarray) -> np.ndarray:
    """Convert joint 0 quaternion to 6D wrist rotation."""
    from scipy.spatial.transform import Rotation
    q = pose_coeffs[0]
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1., 0., 0., 0., 1., 0.], dtype=np.float32)
    q = q / norm
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return R[:, :2].T.flatten().astype(np.float32)


def axis_angle_to_pca15(pose_45: np.ndarray,
                         components: np.ndarray,
                         mean: np.ndarray) -> np.ndarray:
    return ((pose_45 - mean) @ components.T).astype(np.float32)


def random_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def build_feature_vec(bop_id: int, wrist_6d: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    grip_onehot, bbox = object_features(bop_id)
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat[0:3]   = random_unit_vec(rng)
    feat[3:6]   = random_unit_vec(rng)
    feat[6]     = 0.02
    feat[7]     = 0.0
    feat[8:11]  = 0.0
    feat[11:17] = wrist_6d
    feat[17]    = 1.0
    feat[18:22] = grip_onehot
    feat[22:25] = bbox
    return feat


def obj_list_to_bop(obj_list: list[str]) -> int:
    """Pick BOP ID from OakInk2 object list. Use first recognizable object."""
    for obj in obj_list:
        cat_key = "@".join(obj.split("@")[:2])
        bop = OAKINK2_TO_BOP.get(cat_key)
        if bop is not None:
            return bop
    return DEFAULT_BOP


def compute_norm_stats(features: np.ndarray, sdf_feats: np.ndarray,
                       targets: np.ndarray) -> dict:
    feat_mean = features.mean(0)
    feat_std  = features.std(0).clip(min=1e-6)
    sdf_mean  = sdf_feats.mean(0) if sdf_feats.std() > 1e-8 else np.zeros(SDF_FEATURE_DIM, np.float32)
    sdf_std   = sdf_feats.std(0).clip(min=1e-6)
    tgt_mean  = targets.mean(0)
    tgt_std   = targets.std(0).clip(min=1e-6)
    return {
        "feature_mean": feat_mean.tolist(),
        "feature_std":  feat_std.tolist(),
        "sdf_mean":     sdf_mean.tolist(),
        "sdf_std":      sdf_std.tolist(),
        "target_mean":  tgt_mean.tolist(),
        "target_std":   tgt_std.tolist(),
    }


def collect_sequence(pkl_path: Path, hand: str,
                     components: np.ndarray, mean: np.ndarray,
                     rng: np.random.Generator, seq_id: int) -> list[dict]:
    """Load one PKL and return per-frame dicts."""
    try:
        data = pickle.load(open(pkl_path, "rb"))
    except Exception as e:
        print(f"  [WARN] Failed to load {pkl_path.name}: {e}")
        return []

    raw_mano   = data.get("raw_mano", {})
    frame_ids  = data.get("frame_id_list", [])
    obj_list   = data.get("obj_list", [])
    bop_id     = obj_list_to_bop(obj_list)

    hand_key_pose = f"r{'h' if hand == 'right' else 'h'}__pose_coeffs".replace(
        "rh", "rh" if hand == "right" else "lh")

    rows = []
    for frame_idx, fid in enumerate(frame_ids):
        frame = raw_mano.get(fid) or raw_mano.get(frame_idx)
        if frame is None:
            continue

        pose_key  = "rh__pose_coeffs" if hand == "right" else "lh__pose_coeffs"
        betas_key = "rh__betas"       if hand == "right" else "lh__betas"

        if pose_key not in frame:
            continue

        pc = frame[pose_key]  # tensor or ndarray (1, 16, 4)
        if hasattr(pc, "numpy"):
            pc = pc.numpy()
        pc = pc.squeeze(0)    # (16, 4)

        # Skip frames with degenerate poses (all-zero quaternions)
        if np.all(np.abs(pc) < 1e-6):
            continue

        wrist_6d  = pose_coeffs_to_wrist_6d(pc)
        finger_aa = pose_coeffs_to_finger_aa(pc)
        pca15     = axis_angle_to_pca15(finger_aa, components, mean)
        feat      = build_feature_vec(bop_id, wrist_6d, rng)

        betas_np = frame.get(betas_key)
        if betas_np is not None and hasattr(betas_np, "numpy"):
            betas_np = betas_np.numpy().squeeze(0)
        else:
            betas_np = np.zeros(MANO_BETAS_DIM, dtype=np.float32)

        rows.append(dict(
            feat=feat, pca15=pca15, w6d=wrist_6d, bop_id=bop_id,
            seq_id=seq_id, frame_idx=frame_idx,
            betas=betas_np.astype(np.float32),
        ))
    return rows


def rows_to_arrays(rows: list[dict]) -> dict:
    return dict(
        features     = np.stack([r["feat"]     for r in rows]),
        targets      = np.stack([r["pca15"]    for r in rows]),
        betas        = np.stack([r["betas"]    for r in rows]),
        wrist_rot_6d = np.stack([r["w6d"]      for r in rows]),
        obj_id       = np.array([r["bop_id"]   for r in rows], dtype=np.int32),
        sdf_features = np.zeros((len(rows), SDF_FEATURE_DIM), dtype=np.float32),
        sequence_id  = np.array([r["seq_id"]   for r in rows], dtype=np.int32),
        frame_index  = np.array([r["frame_idx"]for r in rows], dtype=np.int32),
        contact      = np.ones(len(rows), dtype=np.uint8),   # all frames are grasps
        is_mirror    = np.zeros(len(rows), dtype=np.uint8),
    )


def actor_from_path(p: Path) -> str:
    """Extract actor ID (e.g. 'A001') from PKL filename."""
    name = p.stem  # scene_01__A001++seq__...
    parts = name.split("__")
    if len(parts) >= 2:
        return parts[1].split("++")[0]
    return "A001"


def collect_all(anno_dir: Path, hand: str,
                components: np.ndarray, mean: np.ndarray,
                rng: np.random.Generator) -> tuple[dict, dict]:
    pkls = sorted(p for p in anno_dir.iterdir() if p.suffix == ".pkl")
    print(f"  Found {len(pkls)} annotation PKLs")

    train_pkls = [p for p in pkls if actor_from_path(p) not in VAL_ACTORS]
    val_pkls   = [p for p in pkls if actor_from_path(p) in VAL_ACTORS]
    print(f"  Split: {len(train_pkls)} train / {len(val_pkls)} val sequences")

    train_rows, val_rows = [], []

    for seq_id, pkl in enumerate(tqdm(train_pkls, desc="train")):
        train_rows.extend(collect_sequence(pkl, hand, components, mean, rng, seq_id))

    offset = len(train_pkls)
    for seq_id, pkl in enumerate(tqdm(val_pkls, desc="val")):
        val_rows.extend(collect_sequence(pkl, hand, components, mean, rng, offset + seq_id))

    print(f"  Frames — train: {len(train_rows)}, val: {len(val_rows)}")
    return rows_to_arrays(train_rows), rows_to_arrays(val_rows)


def write_h5(path: Path, train: dict, val: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    norm_stats = compute_norm_stats(
        train["features"], train["sdf_features"], train["targets"])
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps(norm_stats)
        for split_name, d in [("train", train), ("val", val)]:
            g = f.create_group(split_name)
            for k, v in d.items():
                g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
    print(f"Wrote {path}  train={len(train['features'])} val={len(val['features'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anno_dir",  default="data/raw/oakink2/anno_preview", type=Path)
    p.add_argument("--mano_dir",  default="data/models/mano", type=Path)
    p.add_argument("--out_dir",   default="data/processed/oakink2_mano",   type=Path)
    p.add_argument("--hand",      choices=["left", "right", "both"], default="right")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    from scipy.spatial.transform import Rotation  # noqa validate

    rng   = np.random.default_rng(args.seed)
    hands = ["left", "right"] if args.hand == "both" else [args.hand]

    for hand in hands:
        pca_path = args.mano_dir / f"pca_{hand}.json"
        pca = json.load(open(pca_path))
        components = np.array([row["values"] for row in pca["pca_matrix"]])
        mean_pose  = np.array(pca["mean_pose"])

        print(f"\n=== {hand.upper()} ===")
        train, val = collect_all(args.anno_dir, hand, components, mean_pose, rng)

        if len(train["features"]) == 0:
            print(f"No {hand} frames found.")
            continue

        write_h5(args.out_dir / hand / "dataset_mano.h5", train, val)

    print("\nDone.")


if __name__ == "__main__":
    main()
