"""Recompute missing normalization stats from processed canonical datasets.

This does not rebuild datasets. It only updates:
  - oakink_canonical/stats.json: input_mean/input_std for object-relative frame_feat
  - hot3d_canonical/stats.json: pts_mean/pts_std for PointNet obj_pts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.model_io import HOT3D_FRAME_DIM  # noqa: E402
from utils.paths import HOT3D_CANON, OAKINK_CANON  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _sample_points(points: np.ndarray, n_points: int, item_index: int) -> np.ndarray:
    replace = len(points) < n_points
    rng = np.random.default_rng(item_index)
    idx = rng.choice(len(points), n_points, replace=replace)
    return points[idx].astype(np.float32)


def _rotmat_to_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def recompute_oakink_input_stats(root: Path = OAKINK_CANON, n_points: int = 1024) -> dict:
    data_path = root / "dataset.npz"
    split_path = root / "split.json"
    stats_path = root / "stats.json"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    data = np.load(data_path, allow_pickle=True)
    split = _load_json(split_path)
    train_indices = [int(i) for i in split["train"]]

    pose = data["pose"].astype(np.float32)
    tsl = data["tsl"].astype(np.float32)
    obj_anno = data["obj_anno"].astype(np.float32)
    obj_name = data["obj_name"].astype(str)

    point_cache: dict[str, np.ndarray] = {}
    feats = np.empty((len(train_indices), HOT3D_FRAME_DIM), dtype=np.float64)

    for item_index, idx in enumerate(train_indices):
        name = str(obj_name[idx])
        if name not in point_cache:
            point_cache[name] = np.load(root / "obj_pts" / f"{name}.npy").astype(np.float32)
        raw_points = _sample_points(point_cache[name], n_points, item_index)

        anno = obj_anno[idx]
        r_obj = anno[:9].reshape(3, 3)
        t_obj = anno[9:12]
        wrist_t = tsl[idx]
        r_wrist = Rotation.from_rotvec(pose[idx, :3]).as_matrix().astype(np.float32)

        pts_world = raw_points @ r_obj.T + t_obj
        nearest_dist = float(np.linalg.norm(pts_world - wrist_t[None, :], axis=-1).min())
        rel_pos = r_obj.T @ (wrist_t - t_obj)
        rel_rot6d = _rotmat_to_6d(r_obj.T @ r_wrist)
        feats[item_index] = np.concatenate(
            [rel_pos, rel_rot6d, np.zeros(3, dtype=np.float32), np.array([nearest_dist], dtype=np.float32)]
        )

    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0

    stats = _load_json(stats_path)
    stats["input_feature_order"] = ["rel_pos(3)", "rel_rot6d(6)", "rel_vel(3)", "dist(1)"]
    stats["input_mean"] = mean.tolist()
    stats["input_std"] = std.tolist()
    stats["train_samples_for_input_stats"] = int(len(feats))
    _write_json(stats_path, stats)
    return {"root": str(root), "n": int(len(feats)), "mean": mean.tolist(), "std": std.tolist()}


def recompute_hot3d_point_stats(root: Path = HOT3D_CANON) -> dict:
    stats_path = root / "stats.json"
    obj_dir = root / "obj_pts"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    if not obj_dir.exists():
        raise FileNotFoundError(obj_dir)

    point_files = sorted(obj_dir.glob("*.npy"))
    if not point_files:
        raise FileNotFoundError(f"No *.npy point clouds in {obj_dir}")
    pts = np.concatenate([np.load(path).astype(np.float32) for path in point_files], axis=0).astype(np.float64)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    std[std < 1e-6] = 1.0

    stats = _load_json(stats_path)
    stats["pts_mean"] = mean.tolist()
    stats["pts_std"] = std.tolist()
    stats["point_files_for_stats"] = int(len(point_files))
    stats["points_for_stats"] = int(len(pts))
    _write_json(stats_path, stats)
    return {"root": str(root), "n_files": int(len(point_files)), "n_points": int(len(pts)), "mean": mean.tolist(), "std": std.tolist()}


def main() -> None:
    result = {
        "oakink_input": recompute_oakink_input_stats(),
        "hot3d_points": recompute_hot3d_point_stats(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
