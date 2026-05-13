"""
03_controller_proxy_h5.py — Derive synthetic controller poses from MANO wrist in H5 file.

This version reads directly from hot3d_training.h5 instead of HuggingFace streaming,
avoiding dataset schema casting issues.

Usage:
  python 03_controller_proxy_h5.py
  python 03_controller_proxy_h5.py --noise_std 0.01 --plot
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path

# Quest 3 controller tracking ring sits ~5 cm proximal, ~2 cm dorsal from palm centre.
PALM_PROXIMAL_M = 0.05
PALM_DORSAL_M   = 0.02
PALM_TO_CTRL_OFFSET = np.array([0.0, -PALM_PROXIMAL_M, PALM_DORSAL_M])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", type=str, default="../data/hot3d_training.h5")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="Std of Gaussian noise on offset (simulating grip variation)")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def _axis_angle_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to quaternion [w,x,y,z]."""
    if rotvec.shape[-1] == 3:
        xyzw = Rotation.from_rotvec(rotvec).as_quat()
        return xyzw[[3, 0, 1, 2]]
    return rotvec


def extract_mano_wrist_poses(h5_path: str, max_samples: int = 5000) -> tuple:
    """
    Read MANO wrist poses from H5 file.
    
    Returns:
        (wrist_positions, wrist_quats, n_frames_total)
    """
    print(f"[INFO] Opening {h5_path}")
    
    if not Path(h5_path).exists():
        print(f"[ERROR] File not found: {h5_path}")
        return None, None, 0
    
    wrist_positions = []
    wrist_quats = []
    
    with h5py.File(h5_path, 'r') as f:
        print(f"[INFO] H5 structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: {obj.shape} {obj.dtype}")
        
        f.visititems(print_structure)
        
        # Try common key patterns for MANO data
        possible_keys = [
            'mano_wrist_position', 'wrist_position', 'hand_position',
            'mano_wrist_orientation', 'wrist_orientation', 'hand_orientation',
            'left_hand_mano', 'right_hand_mano'
        ]
        
        found_keys = set(f.keys())
        print(f"\n[INFO] Top-level keys in H5: {list(found_keys)}")
        
        # Look for MANO-like data
        if 'mano' in found_keys or any('mano' in k.lower() for k in found_keys):
            # Try to extract from mano group/dataset
            for key in found_keys:
                if 'mano' in key.lower():
                    print(f"\n[INFO] Found MANO key: {key}")
                    try:
                        data = f[key]
                        print(f"  Shape: {data.shape}, dtype: {data.dtype}")
                        if len(data.shape) >= 2:
                            sample = data[:min(5)]
                            print(f"  Sample: {sample}")
                    except Exception as e:
                        print(f"  [ERROR] {e}")
        
        # Fallback: check for position/orientation arrays
        for key in found_keys:
            if 'position' in key.lower() or 'translation' in key.lower():
                try:
                    pos_data = f[key][:]
                    print(f"\n[INFO] Found position key: {key}")
                    print(f"  Shape: {pos_data.shape}")
                    if pos_data.shape[0] <= max_samples:
                        wrist_positions.append(pos_data)
                except Exception as e:
                    print(f"  [ERROR] {e}")
        
        for key in found_keys:
            if 'orientation' in key.lower() or 'rotation' in key.lower() or 'quaternion' in key.lower():
                try:
                    rot_data = f[key][:]
                    print(f"\n[INFO] Found orientation key: {key}")
                    print(f"  Shape: {rot_data.shape}")
                    if rot_data.shape[0] <= max_samples:
                        wrist_quats.append(rot_data)
                except Exception as e:
                    print(f"  [ERROR] {e}")
    
    if not wrist_positions or not wrist_quats:
        print("\n[ERROR] Could not find MANO wrist position/orientation in H5")
        return None, None, 0
    
    pos_arr = np.vstack(wrist_positions)
    quat_arr = np.vstack(wrist_quats)
    
    print(f"\n[INFO] Extracted {len(pos_arr)} wrist poses")
    return pos_arr, quat_arr, len(pos_arr)


def derive_controller_poses(wrist_pos: np.ndarray, wrist_quat: np.ndarray, noise_std: float) -> tuple:
    """
    Derive synthetic controller poses from wrist data.
    
    Args:
        wrist_pos: (N, 3) wrist positions
        wrist_quat: (N, 4) quaternions in wxyz format
        noise_std: Gaussian noise std on offset
    
    Returns:
        (delta_positions, wrist_pos, ctrl_pos)
    """
    N = len(wrist_pos)
    
    # Create offset with optional noise
    offsets = np.tile(PALM_TO_CTRL_OFFSET, (N, 1))
    if noise_std > 0:
        offsets += np.random.normal(0, noise_std, offsets.shape)
    
    # Rotate offset by hand orientation
    # Assuming wrist_quat is wxyz, convert to xyzw for scipy
    if wrist_quat.shape[1] == 4:
        if np.abs(wrist_quat[0, 0]) > 0.5:  # Likely wxyz (w is first and typically > 0.5 for identity)
            quat_xyzw = wrist_quat[:, [1, 2, 3, 0]]
        else:  # Likely xyzw already
            quat_xyzw = wrist_quat
    else:
        print(f"[ERROR] Unexpected quat shape: {wrist_quat.shape}")
        return None, None, None
    
    rot_matrices = Rotation.from_quat(quat_xyzw).as_matrix()  # (N, 3, 3)
    
    # Controller position = wrist position + rotated offset
    ctrl_positions = wrist_pos + np.einsum("nij,nj->ni", rot_matrices, offsets)
    
    # Delta = what the model must predict
    delta_positions = -offsets
    
    return delta_positions, wrist_pos, ctrl_positions


def main():
    args = parse_args()
    
    wrist_pos, wrist_quat, n_frames = extract_mano_wrist_poses(args.h5_path)
    
    if wrist_pos is None:
        print("\n[RESULT] Failed to extract MANO data from H5 file")
        print("[HINT] H5 file may have unexpected structure. Run with verbose output above.")
        return
    
    print(f"\n[INFO] Deriving synthetic controller poses (noise_std={args.noise_std}m)...")
    delta, wrist, ctrl = derive_controller_poses(wrist_pos, wrist_quat, args.noise_std)
    
    if delta is None:
        print("[ERROR] Failed to derive controller poses")
        return
    
    distances = np.linalg.norm(delta, axis=1)
    
    print(f"\n{'='*60}")
    print(f"  CONTROLLER PROXY ANALYSIS ({len(wrist)} poses)")
    print(f"{'='*60}")
    print(f"\n  ΔT (Controller-to-Wrist Offset) Statistics:")
    print(f"    Distance (magnitude):")
    print(f"      mean: {distances.mean():.6f} m ({distances.mean()*100:.2f} cm)")
    print(f"      std:  {distances.std():.6f} m ({distances.std()*100:.2f} cm)")
    print(f"      min:  {distances.min():.6f} m ({distances.min()*100:.2f} cm)")
    print(f"      max:  {distances.max():.6f} m ({distances.max()*100:.2f} cm)")
    print(f"\n  Interpretation:")
    print(f"    Tight std (< 2cm)  → fixed offset assumption is valid")
    print(f"    Wide std  (> 5cm)  → model needs context to predict ΔT")
    print(f"\n  Component breakdown:")
    print(f"    ΔX (radial):      mean {delta.mean(axis=0)[0]:.6f}  std {delta.std(axis=0)[0]:.6f}")
    print(f"    ΔY (proximal):    mean {delta.mean(axis=0)[1]:.6f}  std {delta.std(axis=0)[1]:.6f}")
    print(f"    ΔZ (dorsal):      mean {delta.mean(axis=0)[2]:.6f}  std {delta.std(axis=0)[2]:.6f}")
    
    print(f"\n  Wrist position (world frame):")
    print(f"    X: mean {wrist.mean(axis=0)[0]:.3f}  std {wrist.std(axis=0)[0]:.3f}")
    print(f"    Y: mean {wrist.mean(axis=0)[1]:.3f}  std {wrist.std(axis=0)[1]:.3f}")
    print(f"    Z: mean {wrist.mean(axis=0)[2]:.3f}  std {wrist.std(axis=0)[2]:.3f}")
    
    print(f"\n  Controller position (world frame):")
    print(f"    X: mean {ctrl.mean(axis=0)[0]:.3f}  std {ctrl.std(axis=0)[0]:.3f}")
    print(f"    Y: mean {ctrl.mean(axis=0)[1]:.3f}  std {ctrl.std(axis=0)[1]:.3f}")
    print(f"    Z: mean {ctrl.mean(axis=0)[2]:.3f}  std {ctrl.std(axis=0)[2]:.3f}")
    
    # Plot
    if args.plot:
        Path("output").mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, label, col in zip(axes, ["ΔX (m)", "ΔY (m)", "ΔZ (m)"], range(3)):
            ax.hist(delta[:, col], bins=40, color="steelblue", edgecolor="white", alpha=0.7)
            ax.axvline(delta.mean(axis=0)[col], color="red", linestyle="--", linewidth=2, label=f"mean={delta.mean(axis=0)[col]:.4f}")
            ax.axvline(delta.mean(axis=0)[col] - delta.std(axis=0)[col], color="orange", linestyle=":", linewidth=1.5, label=f"±1σ")
            ax.axvline(delta.mean(axis=0)[col] + delta.std(axis=0)[col], color="orange", linestyle=":", linewidth=1.5)
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("offset (m)")
            ax.legend()
        fig.suptitle(f"Controller-to-Wrist Offset Distribution (N={len(delta)}, noise_std={args.noise_std}m)", fontsize=14)
        plt.tight_layout()
        plt.savefig("output/controller_proxy_delta_h5.png", dpi=150)
        print(f"\n  [SAVED] output/controller_proxy_delta_h5.png")
        plt.close()


if __name__ == "__main__":
    main()
