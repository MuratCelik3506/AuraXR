"""Audit auxiliary HDF5 hand-object datasets before mixing into training.

The goal is to decide whether a dataset is suitable for:
  - main temporal LSTM training,
  - contact/refiner training only,
  - or exclusion until preprocessing is fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED_LSTM = [
    "features",
    "sdf_features",
    "obj_id",
    "targets",
    "wrist_rot_6d",
    "sequence_id",
    "frame_index",
]

REQUIRED_REFINER = [
    "features",
    "obj_id",
    "targets",
    "wrist_rot_6d",
]


def summarize_split(g: h5py.Group) -> dict:
    keys = set(g.keys())
    n = int(g["features"].shape[0]) if "features" in g else 0
    out = {
        "n_frames": n,
        "keys": sorted(keys),
        "missing_for_lstm": [k for k in REQUIRED_LSTM if k not in keys],
        "missing_for_refiner": [k for k in REQUIRED_REFINER if k not in keys],
    }
    for key in ["features", "sdf_features", "targets", "wrist_rot_6d"]:
        if key in g:
            out[f"{key}_shape"] = list(g[key].shape)
    contact_name = "contact_v2" if "contact_v2" in keys else "contact" if "contact" in keys else None
    if contact_name:
        contact = g[contact_name][:]
        out["contact_field"] = contact_name
        out["contact_ratio"] = float(np.mean(contact > 0)) if len(contact) else 0.0
    if "sequence_id" in keys:
        out["n_sequences"] = int(len(np.unique(g["sequence_id"][:])))
    return out


def audit(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"]) if "meta" in f.attrs else {}
        splits = {name: summarize_split(f[name]) for name in f.keys() if isinstance(f[name], h5py.Group)}
    decision = "exclude"
    train = splits.get("train") or next(iter(splits.values()), {})
    if not train.get("missing_for_lstm"):
        decision = "lstm_or_refiner"
    elif not train.get("missing_for_refiner"):
        decision = "refiner_only"
    return {
        "path": str(path),
        "decision": decision,
        "meta_architecture": meta.get("architecture", {}),
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reports = [audit(path) for path in args.paths]
    text = json.dumps(reports, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
