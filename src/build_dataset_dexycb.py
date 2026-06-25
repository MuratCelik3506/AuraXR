"""build_dataset_dexycb.py — DexYCB → HOT3D-compatible contact-frame HDF5.

DexYCB format (per session/pose.npz):
  pose_m  (N, 1, 51)  — MANO hand pose
    [:, 0, 0:3]   global wrist rotation (axis-angle)
    [:, 0, 3:48]  finger joints (axis-angle, 45-dim)
    [:, 0, 48:51] wrist translation
  pose_y  (N, 4, 7)   — YCB object poses (not used here)

Session meta (meta.yml):
  ycb_ids       list of YCB object IDs in scene
  ycb_grasp_ind index of grasped object in ycb_ids
  mano_sides    'right' or 'left'

YCB → HOT3D BOP mapping (by shape/grip similarity):
  002 master_chef_can  → 10 can_soup         POWER
  003 cracker_box      → 26 birdhouse_toy    POWER
  004 sugar_box        → 17 carton_milk      POWER
  005 tomato_soup_can  → 12 can_tomato_sauce POWER
  006 mustard_bottle   → 13 bottle_mustard   POWER
  007 tuna_fish_can    → 10 can_soup         POWER
  008 pudding_box      → 20 food_waffles     PALMAR
  009 gelatin_box      → 20 food_waffles     PALMAR
  010 potted_meat_can  → 10 can_soup         POWER
  011 banana           → 21 food_vegetables  POWER
  013 apple            → 21 food_vegetables  POWER
  016 pear             → 21 food_vegetables  POWER
  019 pitcher_base     →  7 coffee_pot       POWER
  021 bleach_cleanser  → 15 bottle_ranch     POWER
  024 bowl             →  2 bowl             POWER
  025 mug              →  8 mug_patterned    POWER
  029 plate            →  3 plate_bamboo     PALMAR
  033 spatula          →  6 spatula_red      PRECISION
  037 scissors         → 32 whiteboard_marker PRECISION
  (others)             → 10 can_soup (fallback POWER)

Split: session-based. Sessions sorted alphabetically; last 15% → val.
  Frame-level random splits let same-session frames appear in both train/val,
  inflating val scores. Session-based split ensures no session overlap.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from grip_categories import object_features

YCB_TO_BOP = {
    2:  10,   # master_chef_can → can_soup
    3:  26,   # cracker_box → birdhouse_toy
    4:  17,   # sugar_box → carton_milk
    5:  12,   # tomato_soup_can → can_tomato_sauce
    6:  13,   # mustard_bottle → bottle_mustard
    7:  10,   # tuna_fish_can → can_soup
    8:  20,   # pudding_box → food_waffles
    9:  20,   # gelatin_box → food_waffles
    10: 10,   # potted_meat_can → can_soup
    11: 21,   # banana → food_vegetables
    13: 21,   # apple → food_vegetables
    16: 21,   # pear → food_vegetables
    19:  7,   # pitcher_base → coffee_pot
    21: 15,   # bleach_cleanser → bottle_ranch
    24:  2,   # bowl → bowl
    25:  8,   # mug → mug_patterned
    29:  3,   # plate → plate_bamboo
    33:  6,   # spatula → spatula_red
    37: 32,   # scissors → whiteboard_marker
}
DEFAULT_BOP = 10  # can_soup (POWER)

MANO_POSE_DIM  = 15
MANO_BETAS_DIM = 10
SDF_FEATURE_DIM = 4
FEATURE_DIM    = 25


def rot6d_identity() -> np.ndarray:
    return np.array([1., 0., 0., 0., 1., 0.], dtype=np.float32)


def axis_angle_to_6d(rot3: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    angle = np.linalg.norm(rot3)
    if angle < 1e-8:
        return rot6d_identity()
    R = Rotation.from_rotvec(rot3).as_matrix()
    return R[:, :2].T.flatten().astype(np.float32)


def axis_angle_to_pca15(pose_45: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
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


def collect_session(sess: Path, hand: str, components: np.ndarray, mean: np.ndarray,
                    rng: np.random.Generator, seq_id: int) -> list[dict]:
    """Collect all valid frames from one session. Returns list of per-frame dicts."""
    meta_path = sess / "meta.yml"
    pose_path = sess / "pose.npz"
    if not meta_path.exists() or not pose_path.exists():
        return []

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    sides = meta.get("mano_sides", [])
    if not any(s.lower() == hand for s in sides):
        return []
    hand_idx = next(i for i, s in enumerate(sides) if s.lower() == hand)

    ycb_ids   = meta.get("ycb_ids", [])
    grasp_ind = meta.get("ycb_grasp_ind", 0)
    ycb_id    = ycb_ids[grasp_ind] if grasp_ind < len(ycb_ids) else None
    bop_id    = YCB_TO_BOP.get(ycb_id, DEFAULT_BOP)

    npz = np.load(pose_path)
    pose_m = npz["pose_m"]   # (N, n_hands, 51)
    if pose_m.shape[1] <= hand_idx:
        return []

    hand_poses = pose_m[:, hand_idx, :]   # (N, 51)
    rows = []
    for i, frame_pose in enumerate(hand_poses):
        if np.all(frame_pose == 0):
            continue
        global_rot = frame_pose[:3]
        finger_aa  = frame_pose[3:48]
        pca15  = axis_angle_to_pca15(finger_aa, components, mean)
        w6d    = axis_angle_to_6d(global_rot)
        feat   = build_feature_vec(bop_id, w6d, rng)
        rows.append(dict(
            feat=feat, pca15=pca15, w6d=w6d, bop_id=bop_id,
            seq_id=seq_id, frame_idx=i,
        ))
    return rows


def rows_to_arrays(rows: list[dict]) -> dict:
    return dict(
        features    = np.stack([r["feat"]   for r in rows]),
        targets     = np.stack([r["pca15"]  for r in rows]),
        betas       = np.zeros((len(rows), MANO_BETAS_DIM), dtype=np.float32),
        wrist_rot_6d= np.stack([r["w6d"]    for r in rows]),
        obj_id      = np.array([r["bop_id"] for r in rows], dtype=np.int32),
        sdf_features= np.zeros((len(rows), SDF_FEATURE_DIM), dtype=np.float32),
        sequence_id = np.array([r["seq_id"]    for r in rows], dtype=np.int32),
        frame_index = np.array([r["frame_idx"] for r in rows], dtype=np.int32),
        contact     = np.ones(len(rows), dtype=np.uint8),
        is_mirror   = np.zeros(len(rows), dtype=np.uint8),
    )


def collect_sessions(subj_dirs: list[Path], hand: str,
                     components: np.ndarray, mean: np.ndarray,
                     rng: np.random.Generator,
                     val_frac: float = 0.15) -> tuple[dict, dict]:
    """Collect all sessions from all subject dirs; split by session (not frame)."""
    all_sessions: list[Path] = []
    for subj_dir in subj_dirs:
        all_sessions.extend(sorted(s for s in subj_dir.iterdir() if s.is_dir()))
    all_sessions.sort()

    n_val = max(1, int(len(all_sessions) * val_frac))
    train_sessions = all_sessions[:-n_val]
    val_sessions   = all_sessions[-n_val:]
    print(f"  Sessions: {len(train_sessions)} train / {len(val_sessions)} val")

    train_rows, val_rows = [], []
    for seq_id, sess in enumerate(tqdm(train_sessions, desc="train sessions", leave=False)):
        train_rows.extend(collect_session(sess, hand, components, mean, rng, seq_id))

    offset = len(train_sessions)
    for seq_id, sess in enumerate(tqdm(val_sessions, desc="val sessions", leave=False)):
        val_rows.extend(collect_session(sess, hand, components, mean, rng, offset + seq_id))

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
    p.add_argument("--dexycb_dir", default="data/raw/dexycb", type=Path)
    p.add_argument("--mano_dir",   default="data/models/mano", type=Path)
    p.add_argument("--out_dir",    default="data/processed/dexycb_mano", type=Path)
    p.add_argument("--hand",       choices=["left", "right", "both"], default="both")
    p.add_argument("--val_frac",   type=float, default=0.15)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    from scipy.spatial.transform import Rotation  # noqa validate
    import yaml  # noqa validate

    rng = np.random.default_rng(args.seed)
    hands = ["left", "right"] if args.hand == "both" else [args.hand]

    subj_dirs = sorted(d for d in args.dexycb_dir.iterdir()
                       if d.is_dir() and d.name.startswith("2020"))
    print(f"Found {len(subj_dirs)} subject dir(s): {[d.name for d in subj_dirs]}")

    for hand in hands:
        pca = json.load(open(args.mano_dir / f"pca_{hand}.json"))
        components = np.array([row["values"] for row in pca["pca_matrix"]])
        mean_pose  = np.array(pca["mean_pose"])

        print(f"\n=== {hand.upper()} ===")
        train, val = collect_sessions(subj_dirs, hand, components, mean_pose, rng, args.val_frac)

        if len(train["features"]) == 0:
            print(f"No {hand} frames found.")
            continue

        write_h5(args.out_dir / hand / "dataset_mano.h5", train, val)


if __name__ == "__main__":
    main()
