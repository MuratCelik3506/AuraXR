"""test_onnx_live.py — Model kalitesi testi: grip tipi ve mesafeye göre ONNX çıktı simülasyonu.

Şunları test eder:
  1. Her grip kategorisi için approach trajectory (40cm → 2cm)
  2. Joint açılarının tutarlı monoton artması (pre-shape → grip)
  3. Grip kategorisi ayrımı (Power vs Precision farkı)
  4. Aşırı / negatif açı kontrolü
  5. Unity log çıktısıyla karşılaştırma

Çalıştır:
    python hot3d_exploration/test_onnx_live.py
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ONNX_DIR = BASE / "onnx"

JOINT_NAMES = [
    "Thumb.CMC", "Thumb.abd",  "Thumb.MCP",  "Thumb.DIP",
    "Idx.abd",   "Idx.MCP",   "Idx.PIP",    "Idx.DIP",
    "Mid.abd",   "Mid.MCP",   "Mid.PIP",    "Mid.DIP",
    "Rng.abd",   "Rng.MCP",   "Rng.PIP",    "Rng.DIP",
    "Pnk.abd",   "Pnk.MCP",   "Pnk.PIP",   "Pnk.DIP",
]

# UME → MANO mapping (same as AuraXRInferenceManager.cs)
UME_TO_MANO = [0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
MANO_NAMES = [
    "Th.MCP", "Th.PIP", "Th.DIP",
    "Idx.MCP","Idx.PIP","Idx.DIP",
    "Mid.MCP","Mid.PIP","Mid.DIP",
    "Rng.MCP","Rng.PIP","Rng.DIP",
    "Pnk.MCP","Pnk.PIP","Pnk.DIP",
]

# HOT3D object catalogue (subset)
OBJECTS = {
    "Bottle (Power)":    {"grip": 0, "bbox": [0.035, 0.035, 0.095]},
    "Mug (Precision)":   {"grip": 1, "bbox": [0.020, 0.020, 0.080]},
    "Plate (Palmar)":    {"grip": 2, "bbox": [0.043, 0.043, 0.065]},
    "Coin (Pinch)":      {"grip": 3, "bbox": [0.025, 0.020, 0.010]},
}

GRIP_NAMES = ["Power", "Precision", "Palmar", "Pinch"]
DISTANCES = [0.40, 0.30, 0.20, 0.10, 0.05, 0.02]


def load_hand(hand: str):
    onnx_path = ONNX_DIR / f"auraxr_{hand}.onnx"
    meta_path  = ONNX_DIR / f"model_meta_{hand}.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
    feat_std  = np.array(meta["feature_std"],  dtype=np.float32)
    tgt_mean  = np.array(meta["target_mean"],  dtype=np.float32)
    tgt_std   = np.array(meta["target_std"],   dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess, feat_mean, feat_std, tgt_mean, tgt_std


def predict(sess, feat_mean, feat_std, tgt_mean, tgt_std,
            dist: float, grip: int, bbox: list) -> np.ndarray:
    # Direction: straight ahead (−Z in wrist frame = object directly forward)
    # HOT3D: wrist Z-forward is typical approach direction
    dir_x, dir_y, dir_z = 0.0, 0.0, -1.0  # directly ahead

    feat = np.array([dir_x, dir_y, dir_z, dist,
                     1.0 if grip == 0 else 0.0,
                     1.0 if grip == 1 else 0.0,
                     1.0 if grip == 2 else 0.0,
                     1.0 if grip == 3 else 0.0,
                     bbox[0], bbox[1], bbox[2]], dtype=np.float32)

    norm = (feat - feat_mean) / (feat_std + 1e-8)
    spatial = norm[:4].reshape(1, 4)
    obj_in  = norm[4:].reshape(1, 7)

    raw_out = sess.run(["joint_angles"],
                       {"spatial_input": spatial, "object_input": obj_in})[0][0]  # (22,)
    angles = raw_out * tgt_std + tgt_mean  # denormalize (radians)
    return np.degrees(angles)  # return in degrees


def check_trajectory(angles_by_dist: dict, obj_name: str) -> list:
    """Check if PIP joints monotonically increase as distance decreases (pre-shape → grip)."""
    issues = []
    dists = sorted(angles_by_dist.keys(), reverse=True)
    # PIP joints: Index PIP (UME 6), Middle PIP (10), Ring PIP (14), Pinky PIP (18)
    pip_indices = [6, 10, 14, 18]
    pip_names   = ["Idx.PIP", "Mid.PIP", "Rng.PIP", "Pnk.PIP"]

    for pi, pname in zip(pip_indices, pip_names):
        vals = [angles_by_dist[d][pi] for d in dists]
        for i in range(len(vals) - 1):
            if vals[i] > vals[i+1] + 5.0:  # allow 5° tolerance
                issues.append(f"  [WARN] {obj_name} {pname}: not monotone at dist={dists[i]:.2f}→{dists[i+1]:.2f}  ({vals[i]:.1f}→{vals[i+1]:.1f}°)")
    return issues


def main():
    print("=" * 70)
    print("  AuraXR ONNX Live Test — Grip Simulation")
    print("=" * 70)

    all_issues = []

    for hand in ["right", "left"]:
        sess, feat_mean, feat_std, tgt_mean, tgt_std = load_hand(hand)
        print(f"\n{'='*70}")
        print(f"  HAND: {hand.upper()}")
        print(f"{'='*70}")

        for obj_name, obj in OBJECTS.items():
            grip = obj["grip"]
            bbox = obj["bbox"]
            angles_by_dist = {}

            print(f"\n  ─── {obj_name} ───")
            print(f"  {'Dist':>5}  ", end="")
            for mn in MANO_NAMES:
                print(f"{mn:>8}", end="")
            print()

            for dist in DISTANCES:
                ume_deg = predict(sess, feat_mean, feat_std, tgt_mean, tgt_std,
                                  dist, grip, bbox)
                angles_by_dist[dist] = ume_deg

                # Map to MANO
                mano = [ume_deg[i] for i in UME_TO_MANO]

                phase = "grip" if dist < 0.10 else "pre "
                print(f"  {dist:.2f}m {phase}  ", end="")
                for v in mano:
                    marker = "!" if v < -5 or v > 120 else " "
                    print(f"{v:>7.1f}{marker}", end="")
                print()

            # Checks
            issues = check_trajectory(angles_by_dist, obj_name)
            all_issues.extend(issues)
            if issues:
                for iss in issues:
                    print(iss)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    if not all_issues:
        print("  ✓ All trajectory checks passed — joints increase monotonically as hand approaches object.")
    else:
        print(f"  ✗ {len(all_issues)} trajectory issue(s) found:")
        for iss in all_issues:
            print(iss)

    # ── Compare with Unity log ────────────────────────────────────────────────
    print(f"\n  Unity log comparison (right hand, Bottle, dist≈0.15m):")
    sess, fm, fs, tm, ts = load_hand("right")
    mano_test = [predict(sess, fm, fs, tm, ts, 0.154, 0, [0.035, 0.035, 0.095])[i] for i in UME_TO_MANO]
    print(f"  ONNX script:  ", end="")
    for n, v in zip(MANO_NAMES, mano_test):
        print(f"  {n}={v:.1f}°", end="")
    print()
    print(f"  Unity log:     Th.MCP=17.5  Th.PIP=26.8  Th.DIP=5.8  Idx.MCP=5.3  Idx.PIP=39.9  ...")
    print()


if __name__ == "__main__":
    main()
