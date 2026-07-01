"""Inspect processed AuraXR datasets without rebuilding them."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.paths import HOT3D_CANON, OAKINK_CANON  # noqa: E402


def inspect_oakink(root: Path) -> dict:
    data = np.load(root / "dataset.npz", allow_pickle=True)
    split = json.load(open(root / "split.json"))
    stats = json.load(open(root / "stats.json"))
    return {
        "path": str(root),
        "fields": {key: list(data[key].shape) for key in data.files},
        "splits": {key: len(value) for key, value in split.items()},
        "n_obj_point_files": len(list((root / "obj_pts").glob("*.npy"))),
        "stats_keys": sorted(stats.keys()),
    }


def inspect_hot3d(root: Path, max_files: int = 5) -> dict:
    files = sorted(glob.glob(str(root / "seq_*.npz")))
    stats = json.load(open(root / "stats.json")) if (root / "stats.json").exists() else {}
    summaries = []
    for path in files[:max_files]:
        data = np.load(path, allow_pickle=True)
        summaries.append({
            "file": Path(path).name,
            "fields": {key: list(data[key].shape) for key in data.files},
        })
    return {
        "path": str(root),
        "n_sequences": len(files),
        "input_feature_order": stats.get("input_feature_order"),
        "examples": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oakink", type=Path, default=OAKINK_CANON)
    parser.add_argument("--hot3d", type=Path, default=HOT3D_CANON)
    args = parser.parse_args()
    print(json.dumps({
        "oakink": inspect_oakink(args.oakink),
        "hot3d": inspect_hot3d(args.hot3d),
    }, indent=2))


if __name__ == "__main__":
    main()
