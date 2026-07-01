"""A7/D2 -- HOT3D 4-level object split (33 objects total).

Reads the unique objects seen across all seq_*.npz files and deterministically
assigns them to: backbone_train(22) / backbone_val(4) / calibration(3) / held_out_test(4).

Outputs split.json and manifest.csv next to the seq_*.npz files.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.model_io import HOT3D_SPLIT_COUNTS  # noqa: E402
from utils.paths import HOT3D_CANON  # noqa: E402

SPLIT_ORDER = ["backbone_train", "backbone_val", "calibration", "held_out_test"]


def collect_objects(root: Path) -> dict[str, list[Path]]:
    """Returns {obj_name: [seq_paths]} mapping for all seq_*.npz files."""
    obj_to_seqs: dict[str, list[Path]] = {}
    for path in sorted(root.glob("seq_*.npz")):
        data = np.load(path, allow_pickle=True)
        obj_name = str(data["obj_name"][0])
        obj_to_seqs.setdefault(obj_name, []).append(path)
    return obj_to_seqs


def build_split(obj_to_seqs: dict[str, list[Path]], seed: int) -> dict[str, list[str]]:
    objects = sorted(obj_to_seqs.keys())
    n_total = len(objects)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(objects))

    counts = {k: HOT3D_SPLIT_COUNTS[k] for k in SPLIT_ORDER}
    surplus = n_total - sum(counts.values())
    if surplus > 0:
        counts["backbone_train"] += surplus

    result: dict[str, list[str]] = {}
    cursor = 0
    for level in SPLIT_ORDER:
        c = min(counts[level], len(shuffled) - cursor)
        result[level] = shuffled[cursor: cursor + c]
        cursor += c
    return result


def write_manifest(root: Path, obj_to_seqs: dict[str, list[Path]], split: dict[str, list[str]]) -> Path:
    obj_to_level: dict[str, str] = {}
    for level, objs in split.items():
        for obj in objs:
            obj_to_level[obj] = level

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_id", "obj_name", "split"])
        writer.writeheader()
        for obj, seqs in sorted(obj_to_seqs.items()):
            level = obj_to_level.get(obj, "unassigned")
            # Map 4-level to train/val/test for dataset split lookup
            split_tag = (
                "train" if level == "backbone_train"
                else "val" if level == "backbone_val"
                else "test" if level in ("calibration", "held_out_test")
                else "all"
            )
            for seq_path in seqs:
                writer.writerow({"seq_id": seq_path.stem.removeprefix("seq_"), "obj_name": obj, "split": split_tag})
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HOT3D_CANON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-split", type=Path, default=None)
    args = parser.parse_args()

    obj_to_seqs = collect_objects(args.root)
    print(f"Found {len(obj_to_seqs)} unique objects across {sum(len(v) for v in obj_to_seqs.values())} sequences.")
    split = build_split(obj_to_seqs, args.seed)
    for level, objs in split.items():
        print(f"  {level}: {len(objs)} objects")

    split_path = args.out_split or args.root / "obj_split.json"
    split_path.write_text(json.dumps(split, indent=2))
    print(f"Wrote {split_path}")

    manifest_path = write_manifest(args.root, obj_to_seqs, split)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
