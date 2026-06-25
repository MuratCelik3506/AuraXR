"""sdf_utils.py — SDF grid computation and query for HOT3D objects.

Offline: compute_sdf_grid() → .npy files in data/models/sdf_grids/
Runtime: SDFDatabase loads pre-computed grids and answers point queries.

Feature per query: [sdf_value (1), sdf_gradient_xyz (3)] = 4-dim
"""

import json
from pathlib import Path

import numpy as np

# SDF_GRID_RES=32: resolution/time tradeoff.
#   32^3 (32,768 voxels) → ~0.1s per object offline, ~0.15MB per grid file.
#   64^3 improved surface accuracy by <1mm but took 8× longer and caused
#   Unity memory issues (33 × 1.2MB = 40MB just for SDF grids at runtime).
#   32^3 gives ≈5mm voxel at typical hand-object scale (15cm bounding box).
SDF_GRID_RES = 32

# SDF_PADDING=0.02: 2cm margin added to all sides of the mesh bounding box.
# Without padding, wrist positions just outside the mesh bounds produced
# undefined SDF values.  2cm covers the closest approach distance where the
# wrist is still outside the object.
SDF_PADDING  = 0.02

SDF_FEATURE_DIM = 4   # sdf_value(1) + sdf_gradient_xyz(3)

# Mapping: BOP object ID → .glb filename stem (numeric instance ID)
# Built from data/raw/hot3d/assets/Hot3DAssets_assets_assets/instance.json
_BOP_TO_INSTANCE: dict[int, str] | None = None

def _load_bop_map(asset_dir: Path) -> dict[int, str]:
    global _BOP_TO_INSTANCE
    if _BOP_TO_INSTANCE is not None:
        return _BOP_TO_INSTANCE
    meta = json.loads((asset_dir / "instance.json").read_text())
    _BOP_TO_INSTANCE = {}
    for iid, info in meta.items():
        if info.get("instance_type") == "object":
            bop = int(info["instance_bop_id"])
            _BOP_TO_INSTANCE[bop] = iid
    return _BOP_TO_INSTANCE


# ---------------------------------------------------------------------------
# Offline: compute and save SDF grids
# ---------------------------------------------------------------------------

def compute_sdf_grid(glb_path: Path, resolution: int = SDF_GRID_RES) -> dict:
    """Compute SDF grid for one mesh using voxelization + EDT (fast: ~0.1s/object).

    Algorithm:
      1. Voxelize mesh at target resolution via trimesh
      2. Compute EDT (Euclidean distance transform) inside and outside separately
      3. Combine: SDF = -EDT_inside (inside voxels) or +EDT_outside (outside voxels)
      4. Convert from voxel units to meters

    Why EDT instead of ray-casting or mesh queries?
      - EDT is O(N) on the voxel grid and trivially parallelisable.
      - Mesh query (trimesh.proximity) is O(F) per query point — too slow for 32^3 queries.
      - Accuracy: EDT has ≈ cell_size/2 error at surfaces, which is 2-3mm at our resolution —
        acceptable because the SDF feature is used as a soft geometry cue, not a hard constraint.

    Returns dict with:
        grid   (res, res, res) float32  — signed distance in meters (negative = inside mesh)
        bounds (2, 3) float32           — [min_xyz, max_xyz] of padded grid in object local frame
    """
    import trimesh
    from scipy.ndimage import distance_transform_edt, zoom

    mesh = trimesh.load(str(glb_path), force="mesh")

    lo = mesh.bounds[0] - SDF_PADDING
    hi = mesh.bounds[1] + SDF_PADDING
    cell = (hi - lo) / (resolution - 1)
    cell_m = float(np.min(cell))  # smallest cell dimension → scale factor

    # Voxelize at fine pitch, then zoom to target resolution
    pitch = cell_m
    voxel = mesh.voxelized(pitch=pitch)
    vox_dense = voxel.fill().matrix  # bool (nx, ny, nz) — True=inside

    # Resize to exactly resolution^3
    scale = resolution / np.array(vox_dense.shape, dtype=float)
    vox_res = zoom(vox_dense.astype(np.float32), scale, order=0) > 0.5

    # EDT: distance from each voxel to nearest surface voxel
    dt_out = distance_transform_edt(~vox_res)  # outside voxels → nearest surface
    dt_in  = distance_transform_edt(vox_res)   # inside  voxels → nearest surface

    sdf_vox = np.where(vox_res, -dt_in, dt_out)  # negative inside
    sdf_m   = (sdf_vox * cell_m).astype(np.float32)  # convert to meters

    bounds = np.array([lo, hi], dtype=np.float32)
    return {"grid": sdf_m, "bounds": bounds}


def save_sdf_grid(grid_data: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), grid=grid_data["grid"], bounds=grid_data["bounds"])


def load_sdf_grid(path: Path) -> dict:
    d = np.load(str(path))
    return {"grid": d["grid"], "bounds": d["bounds"]}


# ---------------------------------------------------------------------------
# Runtime: query SDF at a 3D point (trilinear interpolation)
# ---------------------------------------------------------------------------

def _trilinear(grid: np.ndarray, frac) -> float:
    """Trilinear interpolation. frac = (fx, fy, fz) each in [0, res-1]."""
    res = grid.shape[0]
    frac = np.clip(np.asarray(frac, dtype=np.float64), 0.0, res - 1.0)
    x0 = min(int(frac[0]), res - 2)
    y0 = min(int(frac[1]), res - 2)
    z0 = min(int(frac[2]), res - 2)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    dx = frac[0] - x0
    dy = frac[1] - y0
    dz = frac[2] - z0
    v = (
        grid[x0, y0, z0] * (1-dx)*(1-dy)*(1-dz)
      + grid[x1, y0, z0] *    dx *(1-dy)*(1-dz)
      + grid[x0, y1, z0] * (1-dx)*   dy *(1-dz)
      + grid[x0, y0, z1] * (1-dx)*(1-dy)*   dz
      + grid[x1, y1, z0] *    dx *   dy *(1-dz)
      + grid[x1, y0, z1] *    dx *(1-dy)*   dz
      + grid[x0, y1, z1] * (1-dx)*   dy *   dz
      + grid[x1, y1, z1] *    dx *   dy *   dz
    )
    return float(v)


def query_sdf_and_gradient(grid: np.ndarray, bounds: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Query SDF value + gradient at a 3D point in object local frame.

    Returns: float32 array of shape (4,) = [sdf_value, grad_x, grad_y, grad_z]
    Gradient approximates the surface normal (points away from the nearest surface).
    Both sdf_value and gradient are in meters.

    Why central differences for gradient (not analytical)?
      - No closed-form gradient for EDT-based SDF.
      - Central diff with eps=1 voxel ≈ 3mm accuracy — sufficient for a soft
        geometry cue (not precise contact detection).

    For points outside the grid bounds (wrist far from object):
      SDF  = boundary_SDF + distance_to_boundary  (valid for convex objects)
      Grad = direction from boundary toward query point (outward, ≈ exterior normal)
      This extrapolation degrades for concave objects but acceptable at dist > 5cm.
    """
    res = grid.shape[0]
    lo, hi = bounds[0], bounds[1]
    point = np.asarray(point, dtype=np.float64)

    # Handle out-of-bounds: extrapolate SDF linearly, gradient points outward
    point_clamped = np.clip(point, lo, hi)
    extra_dist = float(np.linalg.norm(point - point_clamped))

    frac = (point_clamped - lo) / (hi - lo) * (res - 1)
    sdf_boundary = _trilinear(grid, frac)
    sdf_val = sdf_boundary + extra_dist  # add distance outside bounds

    cell_size = (hi - lo) / (res - 1)
    eps = 1.0

    if extra_dist > 1e-6:
        # Gradient: unit vector from boundary toward query point (outward SDF normal)
        grad = (point - point_clamped) / extra_dist
    else:
        # Central-difference gradient inside the grid
        gx = (_trilinear(grid, frac + [eps, 0, 0]) - _trilinear(grid, frac - [eps, 0, 0])) / (2 * eps * cell_size[0])
        gy = (_trilinear(grid, frac + [0, eps, 0]) - _trilinear(grid, frac - [0, eps, 0])) / (2 * eps * cell_size[1])
        gz = (_trilinear(grid, frac + [0, 0, eps]) - _trilinear(grid, frac - [0, 0, eps])) / (2 * eps * cell_size[2])
        grad = np.array([gx, gy, gz])

    return np.array([sdf_val, grad[0], grad[1], grad[2]], dtype=np.float32)


# ---------------------------------------------------------------------------
# SDFDatabase: loads all pre-computed grids and answers queries by BOP ID
# ---------------------------------------------------------------------------

class SDFDatabase:
    """In-memory cache of all 33 HOT3D object SDF grids.

    Usage:
        db = SDFDatabase(Path("data/models/sdf_grids"))
        feature = db.query(bop_id, point_in_obj_local_frame)
        # → float32 (4,): [sdf_value, grad_x, grad_y, grad_z]
    """

    def __init__(self, grid_dir: Path):
        self._grids: dict[int, dict] = {}
        self._dir = Path(grid_dir)
        self._load_all()

    def _load_all(self):
        for path in self._dir.glob("bop*.npz"):
            # filename: bop{id:02d}.npz
            bop_id = int(path.stem[3:])
            self._grids[bop_id] = load_sdf_grid(path)

    def available_ids(self) -> list[int]:
        return sorted(self._grids.keys())

    def query(self, bop_id: int, point: np.ndarray) -> np.ndarray:
        """Return 4-dim SDF feature for a point in object local frame."""
        if bop_id not in self._grids:
            return np.zeros(SDF_FEATURE_DIM, dtype=np.float32)
        gd = self._grids[bop_id]
        return query_sdf_and_gradient(gd["grid"], gd["bounds"], point)

    def __len__(self):
        return len(self._grids)


# ---------------------------------------------------------------------------
# Pre-computation script helper (called by compute_sdf_grids.py)
# ---------------------------------------------------------------------------

def compute_all_grids(
    asset_dir: Path,
    out_dir: Path,
    resolution: int = SDF_GRID_RES,
    overwrite: bool = False,
):
    """Compute and save SDF grids for all 33 HOT3D objects."""
    bop_map = _load_bop_map(asset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for bop_id, iid in sorted(bop_map.items()):
        glb = asset_dir / f"{iid}.glb"
        out = out_dir / f"bop{bop_id:02d}.npz"

        if out.exists() and not overwrite:
            print(f"  BOP {bop_id:2d}: {out.name} already exists — skip")
            results[bop_id] = True
            continue

        if not glb.exists():
            print(f"  BOP {bop_id:2d}: GLB not found ({glb.name}) — skip")
            results[bop_id] = False
            continue

        import time
        t0 = time.time()
        data = compute_sdf_grid(glb, resolution)
        save_sdf_grid(data, out)
        t1 = time.time()
        neg = (data["grid"] < 0).sum()
        print(f"  BOP {bop_id:2d}: {out.name}  {t1-t0:.1f}s  inside={neg}/{resolution**3}")
        results[bop_id] = True

    ok = sum(results.values())
    print(f"\nDone: {ok}/{len(bop_map)} grids saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--asset_dir", default="data/raw/hot3d/assets/Hot3DAssets_assets_assets")
    p.add_argument("--out_dir",   default="data/models/sdf_grids")
    p.add_argument("--res",       default=SDF_GRID_RES, type=int)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    compute_all_grids(Path(args.asset_dir), Path(args.out_dir), args.res, args.overwrite)
