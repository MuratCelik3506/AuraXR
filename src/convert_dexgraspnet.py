"""convert_dexgraspnet.py — Convert DexGraspNet grasps to MANO PCA 15-dim format.

DexGraspNet stores grasps as MANO axis-angle (45-dim = 15 joints × 3).
This script projects them to the 15-dim PCA subspace used by HOT3D (and our model).

DexGraspNet download: https://pku-epic.github.io/DexGraspNet/
Expected layout after download:
  data/dexgraspnet/
    grasp_data/
      core-bottle-*/  (object category folders)
        *.npy  or  *.npz   (grasp files, format varies by release)
    meshes/            (optional, for SDF grid computation)

Output:
  data/dexgraspnet/grasps_mano15.npz
    "pose_pca"    (M, 15)  — MANO PCA 15-dim (projected from 45-dim axis-angle)
    "hand"        (M,)     — b'right' (DexGraspNet uses right hand throughout)
    "obj_id"      (M,)     — object index (within DexGraspNet, not HOT3D BOP IDs)
    "valid"       (M,)     — bool: within PCA reconstruction threshold

Run:
    .venv/bin/python3 src/convert_dexgraspnet.py \\
        --dex_dir data/dexgraspnet/ \\
        --mano_dir data/models/mano/ \\
        --out data/dexgraspnet/grasps_mano15.npz \\
        --hand right \\
        --max_per_obj 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ── MANO PCA utilities ────────────────────────────────────────────────────────

def load_mano_pca(mano_dir: Path, hand: str = "right") -> tuple[np.ndarray, np.ndarray]:
    """Load MANO PCA basis and mean pose from model pkl.

    Returns:
        pca_matrix : (15, 45) — PCA components (rows are basis vectors)
        mean_pose  : (45,)    — mean axis-angle pose
    """
    import smplx
    model = smplx.create(
        str(mano_dir),
        model_type="mano",
        is_rhand=(hand == "right"),
        use_pca=True,
        num_pca_comps=15,
        flat_hand_mean=False,
    )
    pca_matrix = model.hand_components.detach().cpu().numpy()   # (15, 45)
    mean_pose  = model.hand_mean.detach().cpu().numpy()         # (45,)
    return pca_matrix, mean_pose


def axis_angle_to_pca(axis_angle_45: np.ndarray, pca_matrix: np.ndarray,
                      mean_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project axis-angle (N, 45) → MANO PCA (N, 15).

    Projection: pca_pose = (aa - mean) @ pca_matrix.T
    Reconstruction error used to filter invalid grasps.
    """
    centered = axis_angle_45 - mean_pose[None, :]              # (N, 45)
    pca_pose = centered @ pca_matrix.T                         # (N, 15)

    # Reconstruction error: how well does the PCA subspace explain this grasp?
    reconstructed = pca_pose @ pca_matrix + mean_pose[None, :]
    recon_err = np.linalg.norm(reconstructed - axis_angle_45, axis=-1)   # (N,)
    return pca_pose, recon_err


# ── DexGraspNet loader ────────────────────────────────────────────────────────

def discover_grasp_files(dex_dir: Path) -> list[Path]:
    """Find all grasp data files in DexGraspNet directory."""
    grasp_dir = dex_dir / "grasp_data"
    if not grasp_dir.exists():
        # Some releases put files at the top level
        grasp_dir = dex_dir
    files = sorted(grasp_dir.rglob("*.npy")) + sorted(grasp_dir.rglob("*.npz"))
    return files


def load_grasp_file(path: Path) -> np.ndarray | None:
    """Load a DexGraspNet grasp file → (N, 45) MANO axis-angle or None.

    DexGraspNet releases have varied formats. This handles the two known formats:
      Format A (grasp_dict npz):  data['qpos'] or data['hand_pose'] (N, 45)
      Format B (raw npy):          shape (N, 45) or (45,) directly
    """
    try:
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            # Try common key names
            for key in ("hand_pose", "qpos", "pose", "mano_pose", "hand_poses"):
                if key in data:
                    arr = np.array(data[key], dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    if arr.shape[-1] == 45:
                        return arr.reshape(-1, 45)
            # If no known key, try the first array with shape (*, 45)
            for key in data.files:
                arr = np.array(data[key], dtype=np.float32)
                if arr.ndim >= 1 and arr.shape[-1] == 45:
                    return arr.reshape(-1, 45)
        elif path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True).astype(np.float32)
            if arr.ndim == 1 and arr.shape[0] == 45:
                return arr[None, :]
            if arr.ndim == 2 and arr.shape[1] == 45:
                return arr
    except Exception as e:
        print(f"  [SKIP] {path.name}: {e}")
    return None


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(args):
    dex_dir  = Path(args.dex_dir)
    mano_dir = Path(args.mano_dir)
    out_path = Path(args.out)

    if not dex_dir.exists():
        print(f"[ERROR] DexGraspNet directory not found: {dex_dir}")
        print("  Download from https://pku-epic.github.io/DexGraspNet/ and place in data/dexgraspnet/")
        return

    print(f"Loading MANO PCA basis ({args.hand} hand)…")
    pca_matrix, mean_pose = load_mano_pca(mano_dir, args.hand)
    print(f"  PCA matrix: {pca_matrix.shape}  mean_pose: {mean_pose.shape}")

    grasp_files = discover_grasp_files(dex_dir)
    print(f"Found {len(grasp_files)} grasp files in {dex_dir}")
    if not grasp_files:
        print("[ERROR] No grasp files (.npy / .npz) found.")
        return

    all_pca: list[np.ndarray] = []
    all_obj_id: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    RECON_THRESHOLD = 0.5   # rad — grasps with higher recon error are excluded

    for obj_idx, path in enumerate(grasp_files):
        aa = load_grasp_file(path)
        if aa is None or len(aa) == 0:
            continue

        # Subsample if too many grasps per object
        if args.max_per_obj > 0 and len(aa) > args.max_per_obj:
            idxs = np.random.choice(len(aa), args.max_per_obj, replace=False)
            aa = aa[idxs]

        pca, recon_err = axis_angle_to_pca(aa, pca_matrix, mean_pose)
        valid = recon_err < RECON_THRESHOLD

        all_pca.append(pca)
        all_obj_id.append(np.full(len(pca), obj_idx, dtype=np.int32))
        all_valid.append(valid)

        if obj_idx % 100 == 0:
            print(f"  [{obj_idx+1}/{len(grasp_files)}] {path.parent.name}/{path.name}  "
                  f"grasps={len(pca)}  valid={valid.sum()}")

    if not all_pca:
        print("[ERROR] No grasps loaded.")
        return

    pose_pca = np.concatenate(all_pca,    axis=0)   # (M, 15)
    obj_ids  = np.concatenate(all_obj_id, axis=0)   # (M,)
    valid    = np.concatenate(all_valid,  axis=0)   # (M,)

    print(f"\nTotal: {len(pose_pca)} grasps  valid={valid.sum()} ({100*valid.mean():.1f}%)")
    print(f"PCA range: [{pose_pca.min():.3f}, {pose_pca.max():.3f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pose_pca=pose_pca, obj_id=obj_ids, valid=valid,
                        hand=np.array([args.hand.encode()] * len(pose_pca)))
    print(f"Saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dex_dir",     default="data/dexgraspnet", type=Path)
    p.add_argument("--mano_dir",    default="data/models/mano", type=Path)
    p.add_argument("--out",         default="data/dexgraspnet/grasps_mano15.npz", type=Path)
    p.add_argument("--hand",        default="right", choices=["left", "right"])
    p.add_argument("--max_per_obj", default=500, type=int,
                   help="Max grasps per object (0=all). Default 500 keeps dataset manageable.")
    p.add_argument("--seed",        default=42, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    convert(args)
