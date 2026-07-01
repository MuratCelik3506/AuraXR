"""Export AuraXR Unity demo assets.

Creates Unity-readable assets from the HOT3D canonical preprocessing outputs:

- model_stats.json: input/point normalization stats used by aura_phase2_best.pt
- objects/*.bytes: float32 little-endian point clouds, exactly (1024, 3)
- objects_manifest.json: object_id -> point cloud asset path and bbox metadata

Unity runtime must not resample point clouds. If a source .npy has a different
number of points, this script deterministically resamples it to exactly 1024.

Usage:
  python3 src/unity/export_demo_assets.py
  python3 src/unity/export_demo_assets.py --objects mug_white can_parmesan spatula_red
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.model_io import DEFAULT_N_POINTS  # noqa: E402
from utils.paths import HOT3D_CANON  # noqa: E402


DEFAULT_OBJECTS = ("mug_white", "can_parmesan", "spatula_red")


def _resample_points(points: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    """Deterministic random resampling to exactly n_points."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N,3), got {points.shape}")
    if len(points) == n_points:
        return points.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    replace = len(points) < n_points
    idx = rng.choice(len(points), n_points, replace=replace)
    return points[idx].astype(np.float32, copy=False)


def _bbox_meta(points: np.ndarray) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = maxs - mins
    return {
        "bbox_min": mins.astype(float).tolist(),
        "bbox_max": maxs.astype(float).tolist(),
        "bbox_size": size.astype(float).tolist(),
        "bbox_diagonal_m": float(np.linalg.norm(size)),
    }


def _copy_stats(stats_path: Path, out_path: Path) -> None:
    stats = json.loads(stats_path.read_text())
    required = ("input_mean", "input_std", "pts_mean", "pts_std", "input_feature_order")
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(f"{stats_path} missing required keys: {missing}")
    slim = {
        "source": str(stats_path),
        "input_feature_order": stats["input_feature_order"],
        "input_mean": stats["input_mean"],
        "input_std": stats["input_std"],
        "pts_mean": stats["pts_mean"],
        "pts_std": stats["pts_std"],
        "units": stats.get("canonical_convention", {}).get("units", "meters"),
        "rotation_format": stats.get("canonical_convention", {}).get(
            "rotation_format", "6D continuous (first two columns of R_rel)"
        ),
    }
    out_path.write_text(json.dumps(slim, indent=2))


def export_assets(
    root: Path,
    out_dir: Path,
    objects: list[str],
    n_points: int,
) -> None:
    stats_path = root / "stats.json"
    obj_dir = root / "obj_pts"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    if not obj_dir.exists():
        raise FileNotFoundError(obj_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    objects_out = out_dir / "objects"
    objects_out.mkdir(parents=True, exist_ok=True)

    _copy_stats(stats_path, out_dir / "model_stats.json")

    manifest: dict[str, Any] = {
        "n_points": n_points,
        "point_format": "float32 little-endian xyz, normalized in Unity with model_stats pts_mean/pts_std",
        "objects": [],
    }

    for object_id in objects:
        src = obj_dir / f"{object_id}.npy"
        if not src.exists():
            raise FileNotFoundError(src)
        raw = np.load(src).astype(np.float32)
        sampled = _resample_points(raw, n_points=n_points, seed=abs(hash(object_id)) % (2**32))
        out_bytes = objects_out / f"{object_id}_pts.bytes"
        sampled.astype("<f4", copy=False).tofile(out_bytes)
        meta = _bbox_meta(sampled)
        manifest["objects"].append(
            {
                "object_id": object_id,
                "source": str(src),
                "source_count": int(len(raw)),
                "point_count": int(len(sampled)),
                "points_asset": f"objects/{out_bytes.name}",
                **meta,
            }
        )

    (out_dir / "objects_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HOT3D_CANON)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("UnityDemo/Assets/StreamingAssets/AuraXR"),
        help="Unity StreamingAssets/AuraXR output directory.",
    )
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS)
    args = parser.parse_args()

    export_assets(args.root, args.out, args.objects, args.n_points)
    print(f"Wrote Unity demo assets to {args.out}")


if __name__ == "__main__":
    main()
