"""evaluate_onnx.py — ONNX model evaluation with per-finger breakdown.

Runs the exported ONNX model on the val split and reports:
  - Overall MAE (degrees) across all 20 active joints
  - Per-finger MAE: Thumb / Index / Middle / Ring / Pinky
  - Per-joint MAE with names (all 22 joints)
  - Per-phase MAE: pre_shape (10–40 cm) vs grip (< 10 cm)
  - Per-grip-category MAE: Power / Precision / Palmar / Pinch

UME joint layout (per finger: [abd/flex, MCP, PIP, DIP]):
  Thumb  : joints  0–3  (0=CMC-flex, 1=abduction, 2=MCP, 3=DIP)
  Index  : joints  4–7  (4=abduction, 5=MCP, 6=PIP, 7=DIP)
  Middle : joints  8–11 (8=abduction, 9=MCP, 10=PIP, 11=DIP)
  Ring   : joints 12–15 (12=abduction, 13=MCP, 14=PIP, 15=DIP)
  Pinky  : joints 16–19 (16=abduction, 17=MCP, 18=PIP, 19=DIP)
  Placeholder: joints 20–21 (always 0, excluded from MAE)

Run:
    python evaluate_onnx.py --hand right
    python evaluate_onnx.py --hand left
    python evaluate_onnx.py --hand right --hand left  # both at once
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort

# ── Joint metadata ────────────────────────────────────────────────────────────

JOINT_NAMES = [
    # Thumb
    "Thumb.CMC-flex", "Thumb.abd",    "Thumb.MCP",     "Thumb.DIP",
    # Index
    "Index.abd",      "Index.MCP",    "Index.PIP",     "Index.DIP",
    # Middle
    "Middle.abd",     "Middle.MCP",   "Middle.PIP",    "Middle.DIP",
    # Ring
    "Ring.abd",       "Ring.MCP",     "Ring.PIP",      "Ring.DIP",
    # Pinky
    "Pinky.abd",      "Pinky.MCP",    "Pinky.PIP",     "Pinky.DIP",
    # Placeholder
    "Placeholder-20", "Placeholder-21",
]

FINGER_GROUPS = {
    "Thumb" :  [0,  1,  2,  3],
    "Index" :  [4,  5,  6,  7],
    "Middle":  [8,  9, 10, 11],
    "Ring"  :  [12, 13, 14, 15],
    "Pinky" :  [16, 17, 18, 19],
}

ACTIVE_JOINTS = list(range(20))  # joints 20–21 are placeholders

GRIP_NAMES = ["Power", "Precision", "Palmar", "Pinch"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def denormalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr * std + mean


def rad_to_deg_abs(rad: np.ndarray) -> np.ndarray:
    return np.abs(rad) * (180.0 / np.pi)


def mae_deg(pred: np.ndarray, tgt: np.ndarray, axis=None) -> float:
    err = rad_to_deg_abs(pred - tgt)
    return float(err.mean(axis=axis)) if axis is None else err.mean(axis=axis)


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_hand(hand: str, base_dir: Path) -> dict:
    onnx_path   = base_dir / "onnx" / f"auraxr_{hand}.onnx"
    meta_path   = base_dir / "onnx" / f"model_meta_{hand}.json"
    data_path   = base_dir / "data" / hand / "dataset.h5"

    for p in [onnx_path, meta_path, data_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    with open(meta_path) as f:
        meta = json.load(f)

    feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
    feat_std  = np.array(meta["feature_std"],  dtype=np.float32)
    tgt_mean  = np.array(meta["target_mean"],  dtype=np.float32)
    tgt_std   = np.array(meta["target_std"],   dtype=np.float32)

    # Load val split
    with h5py.File(data_path, "r") as hf:
        raw_feat      = hf["val"]["features"][:]    # (N, 15) raw (un-normalised)
        raw_targets   = hf["val"]["targets"][:]     # (N, 22)
        raw_distances = hf["val"]["distances"][:]   # (N,)

    N = len(raw_feat)
    print(f"\n[{hand.upper()}] Val frames: {N:,}")

    # Normalize features for model input
    norm_feat = (raw_feat - feat_mean) / (feat_std + 1e-8)   # (N, 15)
    spatial   = norm_feat[:, :8]   # dir_world(3) + dir_obj_local(3) + dist(1) + approach_speed(1)
    obj_in    = norm_feat[:, 8:]   # grip_oh(4) + bbox(3)

    # Normalize targets (ground truth in normalized space, then denorm after inference)
    norm_tgt = (raw_targets - tgt_mean) / (tgt_std + 1e-8)   # for reference only

    # ONNX inference — batch all at once (fast for val set size)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    preds_norm = sess.run(
        ["joint_angles"],
        {"spatial_input": spatial.astype(np.float32),
         "object_input":  obj_in.astype(np.float32)},
    )[0]   # (N, 22)

    # Denormalize
    preds_raw = denormalize(preds_norm, tgt_mean, tgt_std)   # (N, 22) radians
    tgts_raw  = raw_targets                                   # (N, 22) radians (already raw)

    # ── Per-joint MAE (degrees) ───────────────────────────────────────────────
    per_joint_mae = rad_to_deg_abs(np.abs(preds_raw - tgts_raw).mean(axis=0))  # (22,)
    overall_mae   = per_joint_mae[ACTIVE_JOINTS].mean()

    # ── Per-finger MAE ────────────────────────────────────────────────────────
    finger_mae = {}
    for fname, jidx in FINGER_GROUPS.items():
        p_f = preds_raw[:, jidx]
        t_f = tgts_raw[:, jidx]
        finger_mae[fname] = float(rad_to_deg_abs(np.abs(p_f - t_f).mean()))

    # ── Per-phase MAE ─────────────────────────────────────────────────────────
    grip_mask     = raw_distances < 0.10
    preshape_mask = raw_distances >= 0.10

    grip_mae = float(rad_to_deg_abs(
        np.abs(preds_raw[grip_mask][:, ACTIVE_JOINTS] -
               tgts_raw[grip_mask][:, ACTIVE_JOINTS]).mean()
    )) if grip_mask.any() else float("nan")

    preshape_mae = float(rad_to_deg_abs(
        np.abs(preds_raw[preshape_mask][:, ACTIVE_JOINTS] -
               tgts_raw[preshape_mask][:, ACTIVE_JOINTS]).mean()
    )) if preshape_mask.any() else float("nan")

    # ── Per-grip-category MAE ─────────────────────────────────────────────────
    grip_oh = raw_feat[:, 8:12]   # one-hot: Power/Precision/Palmar/Pinch
    cat_maes = {}
    for i, name in enumerate(GRIP_NAMES):
        mask = grip_oh[:, i] == 1.0
        if mask.any():
            cat_maes[name] = float(rad_to_deg_abs(
                np.abs(preds_raw[mask][:, ACTIVE_JOINTS] -
                       tgts_raw[mask][:, ACTIVE_JOINTS]).mean()
            ))
        else:
            cat_maes[name] = float("nan")

    return {
        "hand":              hand,
        "n_val_frames":      N,
        "n_grip_frames":     int(grip_mask.sum()),
        "n_preshape_frames": int(preshape_mask.sum()),
        "overall_mae_deg":   float(overall_mae),
        "finger_mae_deg":    finger_mae,
        "phase_mae_deg":     {"grip": grip_mae, "pre_shape": preshape_mae},
        "category_mae_deg":  cat_maes,
        "per_joint_mae_deg": {JOINT_NAMES[i]: float(per_joint_mae[i]) for i in range(22)},
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(r: dict):
    h = r["hand"].upper()
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  AuraXR ONNX Evaluation  ──  {h} HAND")
    print(sep)
    print(f"  Val frames : {r['n_val_frames']:,}  "
          f"(grip={r['n_grip_frames']:,}  pre-shape={r['n_preshape_frames']:,})")
    print(f"\n  Overall MAE (active joints 0–19):  {r['overall_mae_deg']:.2f}°  "
          f"{'✓ below target' if r['overall_mae_deg'] < 5 else '✗ above 5° target'}")

    print(f"\n  ┌─ Per-Finger MAE ({'°':>6}) ─────────────────────────┐")
    for fname, mae in r["finger_mae_deg"].items():
        bar = "█" * int(mae / 0.5)
        print(f"  │  {fname:<8}  {mae:6.2f}°  {bar}")
    print(f"  └────────────────────────────────────────────────────┘")

    print(f"\n  Phase breakdown:")
    print(f"    Pre-shape (10–40 cm) : {r['phase_mae_deg']['pre_shape']:.2f}°")
    print(f"    Grip      (<  10 cm) : {r['phase_mae_deg']['grip']:.2f}°")

    print(f"\n  Grip category breakdown:")
    for name, mae in r["category_mae_deg"].items():
        tag = f"{mae:.2f}°" if not np.isnan(mae) else "  n/a"
        print(f"    {name:<12}: {tag}")

    print(f"\n  Per-joint MAE (all 22 joints):")
    print(f"  {'#':>3}  {'Joint name':<18}  {'MAE (°)':>8}  {'active':>8}")
    print(f"  {'-'*3}  {'-'*18}  {'-'*8}  {'-'*8}")
    for i, (jname, mae) in enumerate(r["per_joint_mae_deg"].items()):
        active = "active" if i < 20 else "skip"
        marker = "  ←" if i >= 20 else ""
        print(f"  {i:>3}  {jname:<18}  {mae:>8.2f}°  {active:>8}{marker}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AuraXR ONNX model with per-finger breakdown.")
    p.add_argument("--hand",       nargs="+", default=["right"],
                   choices=["right", "left"],
                   help="Which hand(s) to evaluate (default: right).")
    p.add_argument("--base_dir",   type=Path,
                   default=Path(__file__).resolve().parent.parent,
                   help="Workspace root (default: parent of this script's directory).")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Directory to save JSON results (default: <base_dir>/results/).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or (args.base_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for hand in args.hand:
        try:
            result = evaluate_hand(hand, args.base_dir)
            print_results(result)
            all_results[hand] = result

            out_path = out_dir / f"onnx_eval_{hand}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved → {out_path}")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    if len(all_results) == 2:
        # Combined summary across both hands
        r_r = all_results["right"]["overall_mae_deg"]
        r_l = all_results["left"]["overall_mae_deg"]
        print(f"\n{'='*62}")
        print(f"  Combined  Right={r_r:.2f}°  Left={r_l:.2f}°  "
              f"Avg={(r_r+r_l)/2:.2f}°")
        print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
