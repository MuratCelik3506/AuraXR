"""
Frame 720 Unity log değerlerini Python/ONNX ile karşılaştır.

Unity log'dan alınan değerler (AuraXRInferenceManager, frame 720):
  Right: ctrl=(0.717, 0.981, 0.987)  obj=(0.150, 0.925, 1.400)  dist=0.704m  cat=13 Bottle
  Left:  ctrl=(0.198, 0.981, 1.287)  obj=(0.150, 0.925, 1.400)  dist=0.135m  cat=13 Bottle
"""

import json
import numpy as np
import onnxruntime as ort

ONNX_DIR = "onnx"
UME_TO_MANO = [0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
MANO_NAMES  = ["Th.MCP","Th.PIP","Th.DIP","Idx.MCP","Idx.PIP","Idx.DIP",
               "Mid.MCP","Mid.PIP","Mid.DIP","Rng.MCP","Rng.PIP","Rng.DIP",
               "Pnk.MCP","Pnk.PIP","Pnk.DIP"]

# ── Unity log'dan alınan raw feature değerleri (frame 720) ───────────────────

RIGHT_FEAT_RAW = np.array([
    -0.471, -0.382, -0.795,   # dir (x,y,z)
     0.704,                    # dist
     1.0, 0.0, 0.0, 0.0,      # grip one-hot: Power
     0.035, 0.035, 0.095      # bbox half-extents
], dtype=np.float32)

LEFT_FEAT_RAW = np.array([
    -0.326, -0.854, -0.406,
     0.135,
     1.0, 0.0, 0.0, 0.0,
     0.035, 0.035, 0.095
], dtype=np.float32)

# ── Unity'nin logladığı beklenen çıktılar (EMA öncesi, denorm sonrası °) ─────

RIGHT_UNITY_DEG = [16.1, 7.5, 23.4, 3.6, -3.2, 17.5, 38.1, 19.9, -4.3,
                   28.1, 36.2, 24.1, -1.8, 28.1, 39.0, 21.8, -6.8, 19.7, 45.4, 19.7, 0.0, 0.0]
LEFT_UNITY_DEG  = [18.6, -10.5, 19.7, 9.4, -4.2, 4.5, 40.3, 23.0, -5.2,
                   14.8, 35.9, 28.6, -0.4, 17.6, 31.6, 25.7, -3.6, 14.3, 29.2, 18.8, 0.0, 0.0]

# ─────────────────────────────────────────────────────────────────────────────

def load_meta(path):
    with open(path) as f:
        m = json.load(f)
    return (np.array(m["feature_mean"], dtype=np.float32),
            np.array(m["feature_std"],  dtype=np.float32),
            np.array(m["target_mean"],  dtype=np.float32),
            np.array(m["target_std"],   dtype=np.float32))

def run(session, feat_raw, feat_mean, feat_std, tgt_mean, tgt_std):
    feat = (feat_raw - feat_mean) / np.where(feat_std < 1e-6, 1.0, feat_std)

    spatial = feat[:4].reshape(1, 4)
    obj     = feat[4:].reshape(1, 7)

    raw_out = session.run(
        ["joint_angles"],
        {"spatial_input": spatial, "object_input": obj}
    )[0][0]  # shape (22,)

    denorm = raw_out * tgt_std + tgt_mean
    return feat, raw_out, denorm

def compare(tag, feat_raw, unity_deg, feat_mean, feat_std, tgt_mean, tgt_std, session):
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")

    feat_norm, raw_out, denorm = run(session, feat_raw, feat_mean, feat_std, tgt_mean, tgt_std)

    # Normalized feature karşılaştırma
    unity_norm_str = {
        "right": "[ 0.351  0.116 -1.693  6.471  1.132 -0.371 -0.532 -0.532 -0.619 -0.317  0.935]",
        "left":  "[-2.626 -1.135 -0.761 -1.207  1.209 -0.353 -0.652 -0.476 -0.741 -0.251  0.850]",
    }[tag.split()[0].lower()]

    print(f"\nFEAT_NORM (Python): {np.array2string(feat_norm, precision=3, suppress_small=True)}")
    print(f"FEAT_NORM (Unity):  {unity_norm_str}")

    # Raw model output karşılaştırma
    unity_raw = np.array(unity_deg)
    print(f"\nMODEL_RAW_OUT (Python): {np.array2string(raw_out, precision=4, suppress_small=True)}")

    # UME denorm (degrees)
    denorm_deg = np.degrees(denorm)
    unity_ume  = np.array(unity_deg)
    diff       = denorm_deg - unity_ume

    print(f"\n{'Joint':<8} {'Python°':>9} {'Unity°':>9} {'Diff':>8}")
    print("-" * 40)
    for i in range(22):
        mark = " <<<" if abs(diff[i]) > 0.5 else ""
        print(f"UME[{i:02d}]  {denorm_deg[i]:>8.2f}  {unity_ume[i]:>8.2f}  {diff[i]:>7.3f}{mark}")

    # MANO eklem açıları
    mano_py    = [np.degrees(denorm[j]) for j in UME_TO_MANO]
    mano_unity = {
        "right": [16.1,23.4,3.6,17.5,38.1,19.9,28.1,36.2,24.1,28.1,39.0,21.8,19.7,45.4,19.7],
        "left":  [18.6,19.7,9.4, 4.5,40.3,23.0,14.8,35.9,28.6,17.6,31.6,25.7,14.3,29.2,18.8],
    }[tag.split()[0].lower()]

    print(f"\n{'MANO Joint':<12} {'Python°':>9} {'Unity°':>9} {'Diff':>8}")
    print("-" * 44)
    for name, py, un in zip(MANO_NAMES, mano_py, mano_unity):
        d = py - un
        mark = " <<<" if abs(d) > 0.5 else ""
        print(f"{name:<12}  {py:>8.2f}  {un:>8.2f}  {d:>7.3f}{mark}")

    max_diff = max(abs(p - u) for p, u in zip(mano_py, mano_unity))
    print(f"\nMax MANO fark: {max_diff:.3f}°  {'✓ OK (<0.5°)' if max_diff < 0.5 else '✗ Fark var!'}")

def main():
    right_meta = load_meta(f"{ONNX_DIR}/model_meta_right.json")
    left_meta  = load_meta(f"{ONNX_DIR}/model_meta_left.json")

    right_sess = ort.InferenceSession(f"{ONNX_DIR}/auraxr_right.onnx")
    left_sess  = ort.InferenceSession(f"{ONNX_DIR}/auraxr_left.onnx")

    compare("right (dist=0.704m, cat=13 Bottle, Power grip)",
            RIGHT_FEAT_RAW, RIGHT_UNITY_DEG, *right_meta, right_sess)

    compare("left  (dist=0.135m, cat=13 Bottle, Power grip)",
            LEFT_FEAT_RAW, LEFT_UNITY_DEG, *left_meta, left_sess)

if __name__ == "__main__":
    main()
