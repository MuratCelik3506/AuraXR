"""Validate HOT3D canonical data expected by the thesis pipeline (A1, A4 contract).

Raw HOT3D parsing is outside this rebuilt minimal pipeline — this script validates
the processed layout and reports the temporal feature contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.paths import HOT3D_CANON  # noqa: E402

REQUIRED_FIELDS = [
    "rel_pos",
    "rel_rot6d",
    "rel_vel",
    "dist",
    "finger_aa45",
    "contact_flag",
    "segment_id",
    "obj_name",
]


def validate(root: Path, max_files: int = 10) -> dict:
    if not (root / "stats.json").exists():
        raise FileNotFoundError(root / "stats.json")
    files = sorted(root.glob("seq_*.npz"))
    if not files:
        raise FileNotFoundError(f"No seq_*.npz files in {root}")
    checked = []
    for path in files[:max_files]:
        data = np.load(path, allow_pickle=True)
        missing = [f for f in REQUIRED_FIELDS if f not in data.files]
        if missing:
            raise ValueError(f"{path.name} missing fields: {missing}")
        checked.append({"file": path.name, "frames": int(len(data["finger_aa45"]))})
    stats = json.load(open(root / "stats.json"))
    return {
        "root": str(root),
        "n_sequences": len(files),
        "checked": checked,
        "input_feature_order": stats.get("input_feature_order"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HOT3D_CANON)
    parser.add_argument("--max-files", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.max_files), indent=2))


if __name__ == "__main__":
    main()
