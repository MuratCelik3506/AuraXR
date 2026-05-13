"""
03_controller_proxy.py — Derive synthetic controller poses from MANO wrist transforms.

This script answers Q-A from questions.md:
  "Where do controller poses come from during training?"

HOT3D has MANO hand poses (wrist position + orientation + finger angles).
The AuraXR model needs Quest 3 controller poses as INPUT.
These do not exist in HOT3D — we must synthesise them.

Strategy:
  The Quest 3 controller tracking origin sits ~5 cm below the palm base
  (toward the handle) in the palm's local coordinate frame.
  Synthetic controller pose = wrist_transform × PALM_TO_CONTROLLER_OFFSET

What this script answers:
  - Is the simulated controller pose plausible?
  - What is the distribution of ΔT (the offset the model must learn to invert)?
  - How sensitive is ΔT to grip variation (noise on the offset)?

Usage:
  python 03_controller_proxy.py
  python 03_controller_proxy.py --n_clips 10 --noise_std 0.01 --plot
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from hot3d_utils import decode_json, load_hot3d, ensure_output_dir, TRANSL_KEYS, ORIENT_KEYS, first_value

# Quest 3 controller tracking ring sits ~5 cm proximal, ~2 cm dorsal from palm centre.
# Coordinate frame: +Y = distal (fingers), +Z = dorsal (back of hand), +X = radial.
# Both hands are symmetric in this approximation.
PALM_PROXIMAL_M = 0.05
PALM_DORSAL_M   = 0.02
PALM_TO_CTRL_OFFSET = np.array([0.0, -PALM_PROXIMAL_M, PALM_DORSAL_M])

MAX_POSES_PER_CLIP = 300   # truncate long clips to avoid memory growth


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_clips", type=int, default=3)
    p.add_argument("--noise_std", type=float, default=0.005,
                   help="Std of Gaussian noise added to offset (metres), simulating grip variation")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def _axis_angle_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to quaternion [w,x,y,z] via scipy."""
    xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return xyzw[[3, 0, 1, 2]]


def extract_wrist_poses(mano_data: dict) -> list:
    poses = []
    for frame_data in mano_data.values():
        for side in ("left", "right"):
            if side not in frame_data or frame_data[side] is None:
                continue
            hand = frame_data[side]

            pos = first_value(hand, TRANSL_KEYS)
            if pos is None:
                continue
            pos = np.array(pos).flatten()[:3]

            raw_orient = first_value(hand, ORIENT_KEYS)
            if raw_orient is None:
                continue
            raw = np.array(raw_orient).flatten()
            quat_wxyz = raw if len(raw) == 4 else _axis_angle_to_wxyz(raw)

            poses.append({"side": side, "pos": pos, "quat": quat_wxyz})

    return poses


def batch_derive_controller_proxy(wrist_poses: list, noise_std: float) -> tuple:
    """
    Vectorised computation of synthetic controller poses for a list of wrist poses.

    Returns:
        delta_pos:   (N, 3) offsets — what the model must predict to recover the wrist
        wrist_pos:   (N, 3) wrist world-space positions
        ctrl_pos:    (N, 3) controller world-space positions
    """
    positions  = np.stack([wp["pos"]  for wp in wrist_poses])   # (N, 3)
    quats_wxyz = np.stack([wp["quat"] for wp in wrist_poses])   # (N, 4)

    offsets = np.tile(PALM_TO_CTRL_OFFSET, (len(wrist_poses), 1))  # (N, 3)
    if noise_std > 0:
        offsets += np.random.normal(0, noise_std, offsets.shape)

    # scipy expects xyzw; our quats are wxyz
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    rot_matrices = Rotation.from_quat(quats_xyzw).as_matrix()   # (N, 3, 3)

    ctrl_positions = positions + np.einsum("nij,nj->ni", rot_matrices, offsets)
    delta_positions = -offsets

    return delta_positions, positions, ctrl_positions


def analyse_clips(n_clips: int, noise_std: float):
    dataset = load_hot3d("train")

    all_delta, all_wrist, all_ctrl = [], [], []
    clips_processed = 0

    for sample in dataset:
        if clips_processed >= n_clips:
            break

        mano_key = next((k for k in sample if "mano" in k.lower()), None)
        if mano_key is None:
            print(f"[SKIP] Clip {clips_processed}: no MANO key")
            continue

        mano_data = decode_json(sample[mano_key])
        wrist_poses = extract_wrist_poses(mano_data)[:MAX_POSES_PER_CLIP]

        if not wrist_poses:
            print(f"[SKIP] Clip {clips_processed}: no wrist poses (key mismatch?)")
            print(f"       MANO keys in first frame: "
                  f"{list(list(mano_data.values())[0].get('left', {}).keys()) if mano_data else []}")
            continue

        print(f"[INFO] Clip {clips_processed}: {len(wrist_poses)} wrist poses")
        delta, wrist, ctrl = batch_derive_controller_proxy(wrist_poses, noise_std)
        all_delta.append(delta)
        all_wrist.append(wrist)
        all_ctrl.append(ctrl)
        clips_processed += 1

    if not all_delta:
        print("\n[RESULT] No wrist poses extracted. Run 01_explore_clips.py first to confirm MANO key names.")
        return None

    delta_arr = np.vstack(all_delta)
    wrist_arr = np.vstack(all_wrist)
    ctrl_arr  = np.vstack(all_ctrl)
    distances = np.linalg.norm(delta_arr, axis=1)

    print(f"\n{'='*60}")
    print(f"  CONTROLLER PROXY ANALYSIS ({len(wrist_arr)} poses)")
    print(f"{'='*60}")
    print(f"\n  ΔT distance stats (m):")
    print(f"    mean {distances.mean():.4f}  std {distances.std():.4f}  "
          f"min {distances.min():.4f}  max {distances.max():.4f}")
    print(f"\n  Tight std → offset predictable from controller state alone.")
    print(f"  Wide std  → model needs object/grip context to predict ΔT accurately.")
    print(f"\n  Wrist XYZ mean {wrist_arr.mean(axis=0).round(3)}  std {wrist_arr.std(axis=0).round(3)}")
    print(f"  Ctrl  XYZ mean {ctrl_arr.mean(axis=0).round(3)}  std {ctrl_arr.std(axis=0).round(3)}")

    return delta_arr, wrist_arr, ctrl_arr


def main():
    args = parse_args()
    result = analyse_clips(args.n_clips, args.noise_std)

    if result is not None and args.plot:
        delta_arr, _, _ = result
        ensure_output_dir()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, label, col in zip(axes, ["ΔX (m)", "ΔY (m)", "ΔZ (m)"], range(3)):
            ax.hist(delta_arr[:, col], bins=30, color="tomato", edgecolor="white")
            ax.set_title(label)
            ax.set_xlabel("offset (m)")
        fig.suptitle(f"Controller-to-Wrist Offset (ΔT) Distribution  noise_std={args.noise_std}m")
        plt.tight_layout()
        plt.savefig("output/controller_proxy_delta.png", dpi=150)
        print(f"\n  [SAVED] output/controller_proxy_delta.png")
        plt.close()


if __name__ == "__main__":
    main()
