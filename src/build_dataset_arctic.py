"""build_dataset_arctic.py — ARCTIC raw_seqs → HOT3D-compatible contact-frame HDF5.

ARCTIC format (raw_seqs/s*/seq.mano.npy):
  left/right:
    pose  (N, 45)  — MANO axis-angle (15 joints × 3)
    rot   (N, 3)   — global wrist rotation (axis-angle)
    trans (N, 3)   — wrist world position
    shape (10,)    — MANO betas (per-subject)

Output: data/processed/arctic_mano/{left,right}/dataset_mano.h5
  Same schema as hot3d_mano — directly loadable by TemporalWindowDataset.

Object mapping (ARCTIC → closest HOT3D BOP ID by shape/grip):
  scissors        → 32 whiteboard_marker  (precision/thin)
  ketchup         → 13 bottle_mustard     (power/bottle)
  phone           → 24 cellphone          (palmar/flat)
  notebook        → 33 dvd_remote         (palmar/flat)
  laptop          → 28 keyboard           (palmar/large-flat)
  mixer           →  7 coffee_pot         (power/appliance)
  capsulemachine  →  7 coffee_pot         (power/appliance)
  espressomachine →  7 coffee_pot         (power/appliance)
  microwave       → 26 birdhouse_toy      (power/large-box)
  box             → 26 birdhouse_toy      (power/box)
  waffleiron      → 20 food_waffles       (palmar/flat-large)

Split: subject-based (s08, s09 → val; s01–s07 → train).
  Frame-level random splits allow same-sequence frames in both train and val,
  which inflates val scores. Subject-based split ensures no sequence overlap.
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from grip_categories import object_features

ARCTIC_TO_BOP = {
    "scissors":        32,
    "ketchup":         13,
    "phone":           24,
    "notebook":        33,
    "laptop":          28,
    "mixer":            7,
    "capsulemachine":   7,
    "espressomachine":  7,
    "microwave":       26,
    "box":             26,
    "waffleiron":      20,
}

MANO_POSE_DIM  = 15
MANO_BETAS_DIM = 10
SDF_FEATURE_DIM = 4
FEATURE_DIM    = 25
WRIST_ROT_DIM  = 6

SYNTH_DISTANCE     = 0.02   # 2 cm — at contact
SYNTH_APPROACH_SPD = 0.0
CONTACT_LABEL      = 1

# Val subjects: last 2 of 9 (≈22%). Train: s01–s07.
VAL_SUBJECTS = {"s08", "s09"}


def rot6d_identity() -> np.ndarray:
    return np.array([1., 0., 0., 0., 1., 0.], dtype=np.float32)


def random_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def axis_angle_to_pca15(pose_45: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return ((pose_45 - mean) @ components.T).astype(np.float32)


def wrist_rot_to_6d(rot_3: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    angle = np.linalg.norm(rot_3)
    if angle < 1e-8:
        return rot6d_identity()
    R = Rotation.from_rotvec(rot_3).as_matrix()
    return R[:, :2].T.flatten().astype(np.float32)


def build_feature_vec(bop_id: int, rng: np.random.Generator) -> np.ndarray:
    grip_onehot, bbox = object_features(bop_id)
    dir_w  = random_unit_vec(rng)
    dir_lo = random_unit_vec(rng)
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat[0:3]   = dir_w
    feat[3:6]   = dir_lo
    feat[6]     = SYNTH_DISTANCE
    feat[7]     = SYNTH_APPROACH_SPD
    feat[8:11]  = 0.0
    feat[11:17] = rot6d_identity()
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


def collect_subject(subj_dir: Path, hand: str,
                    components: np.ndarray, mean: np.ndarray,
                    rng: np.random.Generator,
                    seq_counter_start: int) -> tuple[dict, int]:
    """Collect all frames from one subject directory. Returns arrays dict + next seq_counter."""
    features, targets, betas = [], [], []
    wrist6d, obj_ids, sdf_feats = [], [], []
    seq_ids, frame_idxs = [], []

    seq_counter = seq_counter_start
    for npy_file in sorted(subj_dir.glob("*.mano.npy")):
        seq_name = npy_file.stem.replace(".mano", "")
        obj_name = seq_name.split("_")[0]
        bop_id = ARCTIC_TO_BOP.get(obj_name)
        if bop_id is None:
            continue

        data = np.load(npy_file, allow_pickle=True).item()
        if hand not in data:
            continue
        hand_data = data[hand]

        pose_45  = hand_data["pose"]   # (N, 45)
        rot_3    = hand_data["rot"]    # (N, 3)
        shape_10 = hand_data["shape"]  # (10,)

        N = pose_45.shape[0]
        for i in range(N):
            pca15 = axis_angle_to_pca15(pose_45[i], components, mean)
            w6d   = wrist_rot_to_6d(rot_3[i])
            feat  = build_feature_vec(bop_id, rng)
            feat[11:17] = w6d

            features.append(feat)
            targets.append(pca15)
            betas.append(shape_10.astype(np.float32))
            wrist6d.append(w6d)
            obj_ids.append(bop_id)
            sdf_feats.append(np.zeros(SDF_FEATURE_DIM, dtype=np.float32))
            seq_ids.append(seq_counter)
            frame_idxs.append(i)

        seq_counter += 1

    if not features:
        return {}, seq_counter

    return dict(
        features    = np.stack(features),
        targets     = np.stack(targets),
        betas       = np.stack(betas),
        wrist_rot_6d= np.stack(wrist6d),
        obj_id      = np.array(obj_ids,    dtype=np.int32),
        sdf_features= np.stack(sdf_feats),
        sequence_id = np.array(seq_ids,    dtype=np.int32),
        frame_index = np.array(frame_idxs, dtype=np.int32),
        contact     = np.ones(len(features), dtype=np.uint8),
        is_mirror   = np.zeros(len(features), dtype=np.uint8),
    ), seq_counter


def collect_all_subjects(raw_seqs_dir: Path, hand: str,
                         components: np.ndarray, mean: np.ndarray,
                         rng: np.random.Generator) -> tuple[dict, dict]:
    """Walk all subjects, split train/val by subject name."""
    subjects = sorted(raw_seqs_dir.iterdir())

    train_parts, val_parts = [], []
    seq_counter = 0

    for subj_dir in tqdm(subjects, desc="subjects"):
        if not subj_dir.is_dir():
            continue
        subj_name = subj_dir.name   # e.g. "s01"
        frames, seq_counter = collect_subject(subj_dir, hand, components, mean, rng, seq_counter)
        if not frames:
            continue
        if subj_name in VAL_SUBJECTS:
            val_parts.append(frames)
        else:
            train_parts.append(frames)

    if not train_parts:
        raise RuntimeError("No train frames collected — check raw_seqs path.")
    if not val_parts:
        raise RuntimeError("No val frames collected — are s08/s09 present in raw_seqs?")

    def concat(parts: list[dict]) -> dict:
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}

    return concat(train_parts), concat(val_parts)


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
    print(f"  Val subjects: {sorted(VAL_SUBJECTS)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_seqs", default="data/raw/arctic/raw_seqs", type=Path)
    p.add_argument("--mano_dir", default="data/models/mano", type=Path)
    p.add_argument("--out_dir",  default="data/processed/arctic_mano", type=Path)
    p.add_argument("--hand",     choices=["left", "right", "both"], default="both")
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    from scipy.spatial.transform import Rotation  # noqa validate

    rng  = np.random.default_rng(args.seed)
    hands = ["left", "right"] if args.hand == "both" else [args.hand]

    for hand in hands:
        pca_hand = json.load(open(args.mano_dir / f"pca_{hand}.json"))
        comps_h  = np.array([row["values"] for row in pca_hand["pca_matrix"]])  # (15, 45)
        mean_h   = np.array(pca_hand["mean_pose"])                               # (45,)

        print(f"\n=== {hand.upper()} ===")
        train, val = collect_all_subjects(args.raw_seqs, hand, comps_h, mean_h, rng)
        write_h5(args.out_dir / hand / "dataset_mano.h5", train, val)


if __name__ == "__main__":
    main()
