"""export_sdf_grids_unity.py — Convert SDF NPZ grids to Unity-readable binary files.

Outputs (all go to Resources/SDFGrids/ in the Unity project):
  sdf_manifest.json    — bounds + metadata per object (33 entries)
  sdf_bop{N:02d}.bytes — raw float32 LE grid data (32³ = 32768 floats = 128 KB each)

Unity loads .bytes files as TextAsset.bytes (raw byte arrays).
SDFGridDatabase.cs does trilinear interpolation at runtime.

Run:
    .venv/bin/python3 src/export_sdf_grids_unity.py
"""

import json
import struct
import zipfile
import os
from pathlib import Path

import numpy as np

# BOP name table (matches AuraXRInferenceManager)
BOP_NAMES = {
     1: "holder_black",    2: "bowl",              3: "plate_bamboo",
     4: "spoon_wooden",    5: "potato_masher",     6: "spatula_red",
     7: "coffee_pot",      8: "mug_patterned",     9: "mug_white",
    10: "can_soup",       11: "can_parmesan",      12: "can_tomato_sauce",
    13: "bottle_mustard", 14: "bottle_bbq",        15: "bottle_ranch",
    16: "vase",           17: "carton_milk",       18: "carton_oj",
    19: "flask",          20: "food_waffles",      21: "food_vegetables",
    22: "dumbbell_5lb",   23: "aria_small",        24: "cellphone",
    25: "holder_gray",    26: "birdhouse_toy",     27: "dino_toy",
    28: "keyboard",       29: "whiteboard_eraser", 30: "puzzle_toy",
    31: "mouse",          32: "whiteboard_marker", 33: "dvd_remote",
}


def export(grid_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    for bop_id in range(1, 34):
        npz_path = grid_dir / f"bop{bop_id:02d}.npz"
        if not npz_path.exists():
            print(f"  [SKIP] bop{bop_id:02d}.npz not found")
            continue

        d = np.load(str(npz_path))
        grid   = d["grid"].astype(np.float32)   # (32, 32, 32)
        bounds = d["bounds"].astype(np.float32)  # (2, 3) → [min_xyz, max_xyz]

        # Write raw binary: float32 LE, row-major (x outer, z inner) = 32768 floats
        out_bytes = out_dir / f"sdf_bop{bop_id:02d}.bytes"
        with open(out_bytes, "wb") as f:
            f.write(grid.tobytes())

        entry = {
            "bop_id":     bop_id,
            "name":       BOP_NAMES.get(bop_id, f"obj_{bop_id}"),
            "grid_size":  grid.shape[0],          # always 32
            "bounds_min": bounds[0].tolist(),      # [x, y, z] in meters, object-local
            "bounds_max": bounds[1].tolist(),
        }
        manifest_entries.append(entry)
        print(f"  bop{bop_id:02d} {entry['name']:<25} "
              f"bounds=[{bounds[0].round(3)}..{bounds[1].round(3)}]  "
              f"→ {out_bytes.name} ({out_bytes.stat().st_size//1024}KB)")

    manifest = {"objects": manifest_entries}
    out_manifest = out_dir / "sdf_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ {len(manifest_entries)} grids exported.")
    print(f"  manifest → {out_manifest}")
    print(f"  binaries → {out_dir}/sdf_bop*.bytes")


if __name__ == "__main__":
    grid_dir = Path("data/models/sdf_grids")
    out_dir  = Path("/Users/muratcelik/Desktop/Thesis/Unity/AURAXR/Assets/AuraXR/Resources/SDFGrids")
    export(grid_dir, out_dir)
