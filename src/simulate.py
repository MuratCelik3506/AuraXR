"""simulate.py — Step 5: visualize predicted joint angles along a synthetic approach trajectory.

Run:
    python simulate.py --checkpoint ../checkpoints/right/ --object bottle
    python simulate.py --checkpoint ../checkpoints/right/ --object cup

Simulates an object approaching from 40cm to 2cm directly ahead (along wrist Z axis).
Plots predicted joint angles vs. distance to catch bad transitions before Unity testing.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grip_categories import OBJ_INFO, OBJ_NAMES, object_features
from model import AuraXRModel

# Preset objects: name → BOP ID
OBJECT_PRESETS = {
    "bottle":     13,   # bottle_mustard
    "cup":        9,    # mug_white
    "can":        10,   # can_soup
    "plate":      3,    # plate_bamboo
    "spoon":      4,    # spoon_wooden
    "mouse":      31,   # mouse
    "phone":      24,   # cellphone
    "marker":     32,   # whiteboard_marker
    "vase":       16,   # vase
    "remote":     33,   # dvd_remote
}


def parse_args():
    p = argparse.ArgumentParser(description="Simulate approach trajectory and plot joint angles.")
    p.add_argument("--checkpoint",       required=True, type=Path)
    p.add_argument("--object",           default="bottle",
                   help=f"Object preset name. Options: {list(OBJECT_PRESETS)}")
    p.add_argument("--distance_steps",   default=10, type=int,
                   help="Number of distance steps from 40cm to 2cm.")
    p.add_argument("--output_dir",       default=Path("results"), type=Path)
    p.add_argument("--jump_threshold_deg", default=10.0, type=float,
                   help="Warn if any joint changes more than this (degrees) between steps.")
    return p.parse_args()


def build_feature(rel_pos: np.ndarray, bop_id: int, meta: dict) -> torch.Tensor:
    """Build normalized 15-dim feature vector for a single simulated frame.

    Layout: dir_world(3) + dir_obj_local(3) + dist(1) + approach_speed(1) + grip_oh(4) + bbox(3)
    Object rotation defaults to identity (dir_obj_local == dir_world).
    Approach_speed = 0 (static simulation).
    """
    grip_oh, bbox = object_features(bop_id)
    direction = rel_pos / (np.linalg.norm(rel_pos) + 1e-8)
    dist = float(np.linalg.norm(rel_pos))

    raw = np.concatenate([
        direction,   # 3 world-frame
        direction,   # 3 object-local (identity rotation → same as world-frame)
        [dist],      # 1
        [0.0],       # 1 approach_speed (static simulation)
        grip_oh,     # 4
        bbox,        # 3
    ]).astype(np.float32)

    feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
    feat_std  = np.array(meta["feature_std"],  dtype=np.float32)
    normalized = (raw - feat_mean) / (feat_std + 1e-8)
    return torch.from_numpy(normalized).unsqueeze(0)  # (1, 15)


def denorm_angles(normalized: np.ndarray, meta: dict) -> np.ndarray:
    tgt_mean = np.array(meta["target_mean"], dtype=np.float32)
    tgt_std  = np.array(meta["target_std"],  dtype=np.float32)
    return normalized * tgt_std + tgt_mean


def main():
    args = parse_args()
    device = torch.device("cpu")

    meta_path  = args.checkpoint / "model_meta.json"
    model_path = args.checkpoint / "best_model.pt"
    if not model_path.exists():
        print(f"[ERROR] {model_path} not found.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    arch = meta["architecture"]
    model = AuraXRModel(
        spatial_input_dim=arch["spatial_input_dim"],
        object_input_dim=arch["object_input_dim"],
        hidden_dim=arch["hidden_dim"],
        embedding_dim=arch["embedding_dim"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    bop_id = OBJECT_PRESETS.get(args.object.lower())
    if bop_id is None:
        # Try to parse as integer BOP ID directly
        try:
            bop_id = int(args.object)
        except ValueError:
            print(f"[ERROR] Unknown object '{args.object}'. Options: {list(OBJECT_PRESETS)}")
            return

    obj_name = OBJ_NAMES.get(bop_id, f"bop_{bop_id}")
    print(f"Simulating approach to: {obj_name} (BOP ID {bop_id})")

    distances = np.linspace(0.40, 0.02, args.distance_steps)
    predicted_angles = []

    with torch.no_grad():
        for dist in distances:
            # Object directly ahead along wrist Z axis; identity wrist orientation
            rel_pos = np.array([0.0, 0.0, dist], dtype=np.float32)
            feat    = build_feature(rel_pos, bop_id, meta)
            spatial_in, object_in = AuraXRModel.split_feature(feat)
            pred_joints, _ = model(spatial_in, object_in)
            pred_norm = pred_joints.numpy()[0]  # (22,)
            pred_deg  = denorm_angles(pred_norm, meta)
            predicted_angles.append(pred_deg)

    predicted_angles = np.array(predicted_angles)  # (steps, 22)
    pred_deg_rad = predicted_angles * (180.0 / np.pi)

    # Check for large jumps
    print("\nJump check (consecutive steps):")
    any_jump = False
    for step in range(1, len(distances)):
        delta = np.abs(pred_deg_rad[step] - pred_deg_rad[step - 1])
        if delta.max() > args.jump_threshold_deg:
            joint_idx = np.argmax(delta)
            print(f"  [WARNING] Step {step} ({distances[step-1]*100:.0f}cm→{distances[step]*100:.0f}cm): "
                  f"joint {joint_idx} changed {delta.max():.1f}°")
            any_jump = True
    if not any_jump:
        print(f"  OK — no jumps > {args.jump_threshold_deg}°")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Joint angle trajectories
    n_joints = predicted_angles.shape[1]
    colors = plt.cm.tab20(np.linspace(0, 1, n_joints))
    for j in range(n_joints):
        ax1.plot(distances * 100, pred_deg_rad[:, j], color=colors[j], alpha=0.7, linewidth=1.2, label=f"J{j}")
    ax1.invert_xaxis()
    ax1.axvline(x=10, color="red", linestyle="--", alpha=0.5, label="grip threshold (10cm)")
    ax1.set_xlabel("Distance (cm)")
    ax1.set_ylabel("Joint angle (degrees)")
    ax1.set_title(f"Predicted joint angles during approach: {obj_name}")
    ax1.legend(loc="upper right", ncol=4, fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Mean angle over joints
    mean_angles = pred_deg_rad.mean(axis=1)
    ax2.plot(distances * 100, mean_angles, "b-o", linewidth=2)
    ax2.invert_xaxis()
    ax2.axvline(x=10, color="red", linestyle="--", alpha=0.5, label="grip threshold")
    ax2.set_xlabel("Distance (cm)")
    ax2.set_ylabel("Mean joint angle (degrees)")
    ax2.set_title("Mean joint angle across all 22 joints")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Infer hand side from checkpoint directory name
    hand = args.checkpoint.name  # e.g. "right" or "left"
    out_path = args.output_dir / f"simulation_{hand}_{args.object}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
