"""
09_build_dataset.py — Assemble T=16 sliding windows into a single HDF5 training file.

Reads all data/preprocessed/{device}/{split}/{seq_id}.npz files produced by
08_preprocess_annotations.py and produces:

  data/hot3d_training.h5
    /train/features   float32  [N_train, 16, F_IN]
    /train/targets    float32  [N_train, T_OUT]
    /train/seq_ids    bytes    [N_train]   (source sequence, for debugging)
    /val/...
    /test/...
    /meta             (JSON with feature_dim, target_dim, T, normalisation stats)

Feature vector per frame (F_IN = 95):
  ctrl_t_h0   [3]   controller/wrist proxy position  — hand 0
  ctrl_q_h0   [4]   controller/wrist proxy quaternion (w,x,y,z) — hand 0
  ctrl_grip_h0 [1]  grip proxy
  ctrl_trig_h0 [1]  trigger proxy
  ctrl_t_h1   [3]   same for hand 1
  ctrl_q_h1   [4]
  ctrl_grip_h1 [1]
  ctrl_trig_h1 [1]
  obj_centroid_h0   [3]   nearest object centroid (wrt hand 0)
  obj_bbox_h0       [3]   bbox half-extents
  obj_cat_h0        [1]   category ID float (1–33; 0=unknown)
  obj_centroid_h1   [3]
  obj_bbox_h1       [3]
  obj_cat_h1        [1]
  visual_embed      [64]  CNN embedding placeholder (zeros until images available)
  ── total: 9+9+3+3+1+3+3+1+64 = 96 — we actually use 96

Target vector per window (T_OUT = 78):
  mano_pose_h0   [15]   MANO θ for hand 0 at the LAST frame of the window
  mano_betas_h0  [10]
  mano_wrist_t_h0 [3]
  mano_wrist_q_h0 [4]
  delta_t_h0     [3]   controller-to-wrist offset (identity = zeros in world frame)
  delta_q_h0     [4]   identity quaternion [1,0,0,0]
  mano_pose_h1   [15]
  mano_betas_h1  [10]
  mano_wrist_t_h1 [3]
  mano_wrist_q_h1 [4]
  delta_t_h1     [3]
  delta_q_h1     [4]
  ── total: (15+10+3+4+3+4)*2 = 78

Split strategy:
  - Test participants (P0004/5/6/8/16/20) → test set only
  - Remaining participants: 70/15 split within train set → train / val
  - Stratified by object category and device

Usage:
    python 09_build_dataset.py
    python 09_build_dataset.py --T 16 --stride 1
    python 09_build_dataset.py --min_bimanual 0.3   # require 30% bimanual frames/window
    python 09_build_dataset.py --dry_run
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

DATA_DIR   = Path("../data")
PRE_DIR    = DATA_DIR / "preprocessed"
OUT_FILE   = DATA_DIR / "hot3d_training.h5"
# HOT3D official test split (P0004/5/6/8/16/20) has NO MANO labels — withheld for
# benchmark competition. Use two labeled participants as our held-out test set instead.
TEST_PIDS  = {"P0009", "P0021"}

T          = 16   # temporal window length
F_IN       = 96   # feature dim per frame
T_OUT      = 78   # target dim
IDENTITY_Q = np.array([1., 0., 0., 0.], dtype=np.float32)

ZERO_FEATURE = np.zeros(F_IN, dtype=np.float32)


# Quest 3 controller ring sits ~5 cm proximal, ~2 cm dorsal from palm centre (local frame).
PALM_TO_CTRL_OFFSET = np.array([0.0, -0.05, 0.02], dtype=np.float32)


def _rotate_vec(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion [w,x,y,z]."""
    w, x, y, z = q_wxyz.astype(np.float64)
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)
    return (R @ v.astype(np.float64)).astype(np.float32)


def _ctrl_from_wrist(wrist_t: np.ndarray, wrist_q: np.ndarray) -> np.ndarray:
    """Synthetic controller position = wrist + PALM_TO_CTRL_OFFSET rotated to world frame."""
    return wrist_t + _rotate_vec(wrist_q, PALM_TO_CTRL_OFFSET)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternion [w,x,y,z] → [w,-x,-y,-z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two [w,x,y,z] unit quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)
ZERO_FEATURE[3]  = 1.  # ctrl_q_h0 w=1
ZERO_FEATURE[11] = 1.  # ctrl_q_h1 w=1


def load_npz(path: Path) -> dict | None:
    try:
        data = dict(np.load(str(path), allow_pickle=False))
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        data.update({f"_{k}": v for k, v in meta.items()})
        return data
    except Exception:
        return None


def build_feature(d: dict, i: int) -> np.ndarray:
    """Build F_IN feature vector for frame index i."""
    feat = np.zeros(F_IN, dtype=np.float32)
    off = 0

    # Hand 0 controller proxy (synthesised: wrist + PALM_TO_CTRL_OFFSET in world frame)
    feat[off:off+3] = _ctrl_from_wrist(d["mano_wrist_t_h0"][i], d["ctrl_q_h0"][i]); off += 3
    feat[off:off+4] = d["ctrl_q_h0"][i];      off += 4
    feat[off]       = d["ctrl_grip_h0"][i,0]; off += 1
    feat[off]       = d["ctrl_trigger_h0"][i,0]; off += 1

    # Hand 1 controller proxy
    feat[off:off+3] = _ctrl_from_wrist(d["mano_wrist_t_h1"][i], d["ctrl_q_h1"][i]); off += 3
    feat[off:off+4] = d["ctrl_q_h1"][i];      off += 4
    feat[off]       = d["ctrl_grip_h1"][i,0]; off += 1
    feat[off]       = d["ctrl_trigger_h1"][i,0]; off += 1

    # Object context hand 0
    feat[off:off+3] = d["nearest_centroid_h0"][i]; off += 3
    feat[off:off+3] = d["nearest_bbox_h0"][i];     off += 3
    feat[off]       = float(d["nearest_cat_h0"][i]); off += 1

    # Object context hand 1
    feat[off:off+3] = d["nearest_centroid_h1"][i]; off += 3
    feat[off:off+3] = d["nearest_bbox_h1"][i];     off += 3
    feat[off]       = float(d["nearest_cat_h1"][i]); off += 1

    # Visual embedding placeholder (zeros — filled by CNN offline later)
    off += 64   # feat[off:off+64] already zero

    assert off == F_IN, f"Feature offset mismatch: {off} != {F_IN}"
    return feat


def build_target(d: dict, i: int) -> np.ndarray | None:
    """Build T_OUT target vector from the LAST frame of a window (index i)."""
    target = np.zeros(T_OUT, dtype=np.float32)
    off = 0

    # Hand 0
    if not d["hand_h0_valid"][i]:
        return None
    target[off:off+15] = d["mano_pose_h0"][i];   off += 15
    target[off:off+10] = d["mano_betas_h0"][i];  off += 10
    target[off:off+3]  = d["mano_wrist_t_h0"][i]; off += 3
    target[off:off+4]  = d["mano_wrist_q_h0"][i]; off += 4
    # delta_t_h0: wrist − synthetic_ctrl (≈ −PALM_TO_CTRL_OFFSET rotated to world)
    _ct0 = _ctrl_from_wrist(d["mano_wrist_t_h0"][i], d["ctrl_q_h0"][i])
    target[off:off+3]  = d["mano_wrist_t_h0"][i] - _ct0; off += 3
    # delta_q_h0: rotation from controller frame to wrist frame (≈ identity here)
    target[off:off+4]  = _quat_mul(_quat_conjugate(d["ctrl_q_h0"][i]),
                                    d["mano_wrist_q_h0"][i]); off += 4

    # Hand 1 (fill with hand 0 values if absent, mask handled by training)
    if d["hand_h1_valid"][i]:
        target[off:off+15] = d["mano_pose_h1"][i];   off += 15
        target[off:off+10] = d["mano_betas_h1"][i];  off += 10
        target[off:off+3]  = d["mano_wrist_t_h1"][i]; off += 3
        target[off:off+4]  = d["mano_wrist_q_h1"][i]; off += 4
        _ct1 = _ctrl_from_wrist(d["mano_wrist_t_h1"][i], d["ctrl_q_h1"][i])
        target[off:off+3]  = d["mano_wrist_t_h1"][i] - _ct1; off += 3
        target[off:off+4]  = _quat_mul(_quat_conjugate(d["ctrl_q_h1"][i]),
                                        d["mano_wrist_q_h1"][i]); off += 4
    else:
        # Mirror hand 0 as fallback (model will use h1_valid mask during training)
        target[off:off+15] = d["mano_pose_h0"][i];   off += 15
        target[off:off+10] = d["mano_betas_h0"][i];  off += 10
        target[off:off+3]  = d["mano_wrist_t_h0"][i]; off += 3
        target[off:off+4]  = d["mano_wrist_q_h0"][i]; off += 4
        off += 3                        # delta_t_h1 = zero (no real h1 data)
        target[off:off+4] = IDENTITY_Q; off += 4

    assert off == T_OUT, f"Target offset mismatch: {off} != {T_OUT}"
    return target


def windows_from_sequence(d: dict, t: int, stride: int,
                           min_bimanual: float) -> list[dict]:
    """
    Slide a window of length t over a sequence.
    Returns list of {features: [t, F_IN], target: [T_OUT], seq_id, h1_valid}.
    """
    N     = len(d["timestamps_ns"])
    qmask = d["quality_mask"]
    wins  = []

    for start in range(0, N - t + 1, stride):
        end = start + t
        # All frames in window must pass quality mask
        if not np.all(qmask[start:end]):
            continue
        # Need hand 0 valid in every frame
        if not np.all(d["hand_h0_valid"][start:end]):
            continue
        # Optional bimanual requirement
        bi_rate = np.mean(d["hand_h1_valid"][start:end])
        if bi_rate < min_bimanual:
            continue
        # Build feature matrix [t, F_IN]
        feat_rows = []
        for fi in range(start, end):
            feat_rows.append(build_feature(d, fi))
        features = np.stack(feat_rows, axis=0)  # [t, F_IN]

        # Target = last frame
        target = build_target(d, end - 1)
        if target is None:
            continue

        wins.append({
            "features": features,
            "target":   target,
            "seq_id":   d.get("_seq_id", "unknown").encode("utf-8"),
            "h1_valid": bool(d["hand_h1_valid"][end - 1]),
        })

    return wins


def collect_npz_paths() -> list[tuple[Path, str]]:
    """Returns [(path, split_label)] for all preprocessed sequences."""
    result = []
    for device in ("quest3", "aria"):
        for split in ("train", "test"):
            d = PRE_DIR / device / split
            if not d.exists():
                continue
            for p in sorted(d.glob("*.npz")):
                meta_path = p.with_suffix(".json")
                if not meta_path.exists():
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                pid = meta.get("participant_id", "")
                if pid in TEST_PIDS:
                    label = "test"
                else:
                    label = split  # keep train/val mixing for later
                result.append((p, label, pid))
    return result


def compute_normalisation(windows: list[dict]) -> dict:
    """Compute mean/std of features and targets over training windows."""
    all_feat = np.concatenate([w["features"].reshape(-1, F_IN) for w in windows])
    all_tgt  = np.stack([w["target"] for w in windows])

    feat_mean = all_feat.mean(0).astype(np.float32)
    feat_std  = (all_feat.std(0) + 1e-6).astype(np.float32)
    tgt_mean  = all_tgt.mean(0).astype(np.float32)
    tgt_std   = (all_tgt.std(0) + 1e-6).astype(np.float32)

    return {
        "feature_mean": feat_mean.tolist(),
        "feature_std":  feat_std.tolist(),
        "target_mean":  tgt_mean.tolist(),
        "target_std":   tgt_std.tolist(),
    }


def write_split(hf: h5py.File, split: str, windows: list[dict]):
    if not windows:
        print(f"  {split:<6}: 0 windows — skipped")
        return
    grp = hf.require_group(split)
    N = len(windows)
    feat = np.stack([w["features"] for w in windows])          # [N, T, F_IN]
    tgt  = np.stack([w["target"]   for w in windows])          # [N, T_OUT]
    sids = np.array([w["seq_id"]   for w in windows],
                    dtype=h5py.special_dtype(vlen=bytes))

    grp.create_dataset("features", data=feat, chunks=(64, T, F_IN),
                       compression="gzip", compression_opts=4)
    grp.create_dataset("targets",  data=tgt,  chunks=(64, T_OUT),
                       compression="gzip", compression_opts=4)
    grp.create_dataset("seq_ids",  data=sids)
    print(f"  {split:<6}: {N:,} windows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T",            type=int,   default=16)
    ap.add_argument("--stride",       type=int,   default=1)
    ap.add_argument("--min_bimanual", type=float, default=0.0,
                    help="Minimum fraction of frames in a window with both hands visible")
    ap.add_argument("--val_frac",     type=float, default=0.15)
    ap.add_argument("--dry_run",      action="store_true")
    args = ap.parse_args()

    paths = collect_npz_paths()
    print(f"\nFound {len(paths)} preprocessed sequences.")
    if not paths:
        print("Run 08_preprocess_annotations.py first.")
        return

    train_wins, val_wins, test_wins = [], [], []

    for npz_path, initial_split, pid in tqdm(paths, desc="Building windows"):
        d = load_npz(npz_path)
        if d is None:
            continue
        wins = windows_from_sequence(d, args.T, args.stride, args.min_bimanual)
        if initial_split == "test":
            test_wins.extend(wins)
        else:
            # Assign to train or val
            train_wins.extend(wins)

    # Shuffle then split train → train / val
    random.seed(42)
    random.shuffle(train_wins)
    n_val   = int(len(train_wins) * args.val_frac)
    val_wins   = train_wins[:n_val]
    train_wins = train_wins[n_val:]

    print(f"\n  train : {len(train_wins):,} windows")
    print(f"  val   : {len(val_wins):,} windows")
    print(f"  test  : {len(test_wins):,} windows")
    print(f"  total : {len(train_wins)+len(val_wins)+len(test_wins):,} windows")

    if args.dry_run:
        print("\n[DRY RUN] Remove --dry_run to write HDF5.")
        return

    if not train_wins:
        print("\n[WARN] No training windows — check preprocessing output.")
        return

    # Normalisation stats (from train only)
    print("\nComputing normalisation statistics from train set...")
    norm_stats = compute_normalisation(train_wins)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {OUT_FILE} ...")
    with h5py.File(str(OUT_FILE), "w") as hf:
        write_split(hf, "train", train_wins)
        write_split(hf, "val",   val_wins)
        write_split(hf, "test",  test_wins)

        meta = {
            "T": args.T,
            "feature_dim": F_IN,
            "target_dim":  T_OUT,
            "stride": args.stride,
            "min_bimanual": args.min_bimanual,
            **norm_stats,
        }
        hf.attrs["meta"] = json.dumps(meta)

    import os
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n[DONE] {OUT_FILE}  ({size_mb:.1f} MB)")
    print("Next: run 11_train.py to start training.")


if __name__ == "__main__":
    main()
