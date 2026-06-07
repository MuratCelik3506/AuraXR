"""evaluate.py — Step 4: evaluate trained model on val split and report metrics.

Run:
    python evaluate.py --checkpoint ../checkpoints/right/ --data_dir ../data/right/
    python evaluate.py --checkpoint ../checkpoints/left/  --data_dir ../data/left/

Metrics reported:
    - Joint Angle MAE (degrees) — per joint + overall mean
    - Per-phase MAE: pre_shape (10–40cm) vs grip (<10cm)
    - Per-grip-category MAE: Power / Precision / Palmar / Pinch
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from hot3d_dataset import HOT3DDataset
from model import AuraXRModel


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AuraXR model.")
    p.add_argument("--checkpoint",  required=True, type=Path,
                   help="Directory with best_model.pt and model_meta.json.")
    p.add_argument("--data_dir",    required=True, type=Path)
    p.add_argument("--output_dir",  default=Path("results"), type=Path)
    p.add_argument("--batch_size",  default=256, type=int)
    return p.parse_args()


def denormalize(tensor: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return tensor * std + mean


def rad_to_deg(rad: np.ndarray) -> np.ndarray:
    return np.abs(rad) * (180.0 / np.pi)


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint_dir = args.checkpoint
    meta_path = checkpoint_dir / "model_meta.json"
    model_path = checkpoint_dir / "best_model.pt"

    if not model_path.exists():
        print(f"[ERROR] {model_path} not found. Run train.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    arch = meta["architecture"]
    model = AuraXRModel(
        spatial_input_dim=arch["spatial_input_dim"],
        object_input_dim=arch["object_input_dim"],
        hidden_dim=arch["hidden_dim"],
        embedding_dim=arch["embedding_dim"],
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    hdf5_path = args.data_dir / "dataset.h5"
    val_ds = HOT3DDataset(hdf5_path, split="val", normalise=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    tgt_mean = np.array(meta["target_mean"], dtype=np.float32)
    tgt_std  = np.array(meta["target_std"],  dtype=np.float32)

    # Also load raw distances and labels for per-phase/category breakdown
    with h5py.File(hdf5_path, "r") as hf:
        raw_distances = hf["val"]["distances"][:]
        raw_labels    = hf["val"]["labels"][:]
        raw_features  = hf["val"]["features"][:]  # raw un-normalised

    all_preds  = []
    all_targets = []

    with torch.no_grad():
        for feat, tgt, dist in val_loader:
            feat = feat.to(device)
            spatial_in, object_in = AuraXRModel.split_feature(feat)
            pred_joints, _ = model(spatial_in, object_in)
            pred_norm = pred_joints.cpu().numpy()
            tgt_norm  = tgt.numpy()
            all_preds.append(pred_norm)
            all_targets.append(tgt_norm)

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize
    preds_raw   = denormalize(all_preds,   tgt_mean, tgt_std)
    targets_raw = denormalize(all_targets, tgt_mean, tgt_std)

    # Per-joint MAE (radians → degrees)
    mae_per_joint = rad_to_deg(np.abs(preds_raw - targets_raw).mean(axis=0))
    overall_mae   = mae_per_joint.mean()

    # Per-phase MAE
    grip_mask     = raw_distances < 0.10
    preshape_mask = raw_distances >= 0.10

    grip_mae     = rad_to_deg(np.abs(preds_raw[grip_mask]     - targets_raw[grip_mask]).mean())     if grip_mask.any()     else float("nan")
    preshape_mae = rad_to_deg(np.abs(preds_raw[preshape_mask] - targets_raw[preshape_mask]).mean()) if preshape_mask.any() else float("nan")

    # Per-grip-category MAE (using one-hot in raw features indices 3:7)
    grip_names = ["Power", "Precision", "Palmar", "Pinch"]
    feat_mean_arr = np.array(meta["feature_mean"], dtype=np.float32)
    feat_std_arr  = np.array(meta["feature_std"],  dtype=np.float32)

    # layout: [dir_world(3), dir_obj_local(3), dist(1), approach_speed(1), grip_oh(4), bbox(3)] → grip_oh at [8:12]
    raw_feat_grip = raw_features[:, 8:12]  # already raw from disk
    cat_maes = {}
    for i, name in enumerate(grip_names):
        mask = raw_feat_grip[:, i] == 1.0
        if mask.any():
            cat_maes[name] = float(rad_to_deg(np.abs(preds_raw[mask] - targets_raw[mask]).mean()))
        else:
            cat_maes[name] = float("nan")

    # Print results
    print("\n" + "=" * 60)
    print("  AuraXR Evaluation Results")
    print("=" * 60)
    print(f"\n  Overall MAE: {overall_mae:.2f}°  (target < 5°)")
    print(f"\n  Phase breakdown:")
    print(f"    Pre-shape (10–40cm): {preshape_mae:.2f}°")
    print(f"    Grip       (< 10cm): {grip_mae:.2f}°")
    print(f"\n  Grip category breakdown:")
    for name, mae in cat_maes.items():
        print(f"    {name:<12}: {mae:.2f}°")
    print(f"\n  Per-joint MAE (degrees):")
    for i, mae in enumerate(mae_per_joint):
        print(f"    Joint {i:2d}: {mae:.2f}°")

    # Save to JSON
    results = {
        "overall_mae_deg":  float(overall_mae),
        "preshape_mae_deg": float(preshape_mae),
        "grip_mae_deg":     float(grip_mae),
        "per_category_mae_deg": cat_maes,
        "per_joint_mae_deg": mae_per_joint.tolist(),
        "n_val_frames":     int(len(preds_raw)),
        "n_grip_frames":    int(grip_mask.sum()),
        "n_preshape_frames":int(preshape_mask.sum()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"eval_{checkpoint_dir.name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
