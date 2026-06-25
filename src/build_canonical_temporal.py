"""Build a single canonical-hand temporal dataset from left/right HOT3D h5 files.

The canonical model is trained as a right-hand model:
  - right real frames are kept as-is
  - left real frames are mirrored into right-hand convention

By default source mirror-augmentation frames are dropped because canonicalizing
them would duplicate the corresponding real stream. Use --include_source_mirror
only for explicit augmentation experiments.

Run:
    .venv/bin/python3 src/build_canonical_temporal.py \
        --left_data data/left_temporal_v2/dataset_temporal.h5 \
        --right_data data/right_temporal_v2/dataset_temporal.h5 \
        --output data/canonical_temporal_v2/dataset_temporal.h5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hot3d_utils import mirror_feature, mirror_joints, mirror_wrist_rot


SPLITS = ("train", "val")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--left_data", required=True, type=Path)
    p.add_argument("--right_data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--canonical_hand", default="right", choices=["right"],
                   help="Currently only right-hand canonicalization is supported.")
    p.add_argument("--include_source_mirror", action="store_true",
                   help="Keep mirror-augmentation frames from the source h5 files.")
    return p.parse_args()


def _load_split(path: Path, split: str, source_hand: str, include_source_mirror: bool, seq_offset: int):
    with h5py.File(path, "r") as f:
        g = f[split]
        is_mirror = g["is_mirror"][:] if "is_mirror" in g else np.zeros(len(g["features"]), dtype=np.uint8)
        keep = np.ones(len(is_mirror), dtype=bool) if include_source_mirror else (is_mirror == 0)

        features = g["features"][:][keep].astype(np.float32)
        sdf = g["sdf_features"][:][keep].astype(np.float32)
        targets = g["targets"][:][keep].astype(np.float32)
        wrist = g["wrist_rot_6d"][:][keep].astype(np.float32)
        obj_id = g["obj_id"][:][keep].astype(np.int32)
        seq_id = g["sequence_id"][:][keep].astype(np.int32) + np.int32(seq_offset)
        frame_idx = g["frame_index"][:][keep].astype(np.int32)
        distances = g["distances"][:][keep].astype(np.float32)
        contact = g["contact"][:][keep].astype(np.uint8)
        contact_v2 = (g["contact_v2"][:] if "contact_v2" in g else g["contact"][:])[keep].astype(np.uint8)
        kept_mirror = is_mirror[keep].astype(np.uint8)

    # Current handedness after source augmentation. Real frames keep source_hand;
    # mirrored frames have the opposite handedness.
    source_is_left = source_hand == "left"
    current_is_left = np.logical_xor(source_is_left, kept_mirror.astype(bool))
    to_canonical = current_is_left  # canonical right-hand convention

    for i in np.where(to_canonical)[0]:
        features[i] = mirror_feature(features[i])
        targets[i] = mirror_joints(targets[i])
        wrist[i] = mirror_wrist_rot(wrist[i])

    return {
        "features": features,
        "sdf_features": sdf,
        "obj_id": obj_id,
        "targets": targets,
        "wrist_rot_6d": wrist,
        "distances": distances,
        "sequence_id": seq_id,
        "frame_index": frame_idx,
        "is_mirror": np.zeros_like(kept_mirror, dtype=np.uint8),
        "contact": contact,
        "contact_v2": contact_v2,
        "source_hand": np.full(len(features), 0 if source_hand == "right" else 1, dtype=np.uint8),
        "source_is_mirror": kept_mirror,
    }


def _concat(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def _norm(x: np.ndarray):
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean.tolist(), std.tolist()


def _seq_lengths(seq_ids: np.ndarray) -> dict[int, int]:
    counts = defaultdict(int)
    for sid in seq_ids:
        counts[int(sid)] += 1
    return {int(k): int(v) for k, v in sorted(counts.items())}


def _write_group(hf, split: str, data: dict):
    g = hf.create_group(split)
    for name in [
        "features", "sdf_features", "obj_id", "targets", "wrist_rot_6d",
        "distances", "sequence_id", "frame_index", "is_mirror", "contact",
        "contact_v2", "source_hand", "source_is_mirror",
    ]:
        g.create_dataset(name, data=data[name], compression="gzip")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    split_data = {}
    for split in SPLITS:
        right = _load_split(args.right_data, split, "right", args.include_source_mirror, seq_offset=0)
        max_right_seq = int(right["sequence_id"].max()) if len(right["sequence_id"]) else 0
        left = _load_split(args.left_data, split, "left", args.include_source_mirror, seq_offset=max_right_seq + 1)
        split_data[split] = _concat([right, left])
        print(
            f"{split}: right={len(right['features']):,} left={len(left['features']):,} "
            f"total={len(split_data[split]['features']):,}"
        )

    train = split_data["train"]
    feat_mean, feat_std = _norm(train["features"])
    sdf_mean, sdf_std = _norm(train["sdf_features"])
    tgt_mean, tgt_std = _norm(train["targets"])
    rot_mean, rot_std = _norm(train["wrist_rot_6d"])

    meta = {
        "canonical_hand": "right",
        "source_left": str(args.left_data),
        "source_right": str(args.right_data),
        "source_mirror_policy": "included" if args.include_source_mirror else "dropped",
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "sdf_mean": sdf_mean,
        "sdf_std": sdf_std,
        "target_mean": tgt_mean,
        "target_std": tgt_std,
        "wrist_rot_mean": rot_mean,
        "wrist_rot_std": rot_std,
        "architecture": {
            "input_dim": 25,
            "sdf_input_dim": 4,
            "total_input_dim": 29,
            "output_dim": 22,
            "version": 4,
            "canonical_single_model": True,
        },
        "norm_stats_source": "canonical_train_real_only",
        "train_sequence_lengths": _seq_lengths(split_data["train"]["sequence_id"]),
        "val_sequence_lengths": _seq_lengths(split_data["val"]["sequence_id"]),
    }

    with h5py.File(args.output, "w") as hf:
        hf.attrs["meta"] = json.dumps(meta)
        for split in SPLITS:
            _write_group(hf, split, split_data[split])

    print(f"Wrote canonical dataset -> {args.output}")


if __name__ == "__main__":
    main()
