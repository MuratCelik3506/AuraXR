"""
Synthetic H2O Dataset Generator
=================================
Creates a minimal but structurally correct fake H2O dataset on disk so that
every pipeline component (loader, model, train, evaluate) can be exercised
without downloading the real dataset.

Generated layout (under <out_dir>/):
  annotations/
    subject1/
      h1/
        0/
          cam4/
            hand_pose/     0000.txt … 0099.txt   (128 floats each)
            obj_pose_rt/   0000.txt … 0099.txt   (17 floats: obj_id + 4×4 RT)
            action_label/  0000.txt … 0099.txt   (single int 1-36)
  models/
    label_split/
      action_train.txt
      action_val.txt
      action_test.txt

Usage:
    python -m src.tests.generate_synthetic_h2o --out_dir data/h2o --num_seqs 8
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np


# ─── constants (must match h2o_dataset.py) ──────────────────────────────────
NUM_CLASSES  = 36
NUM_FRAMES   = 100      # frames per synthetic sequence
TOKEN_PER_HAND = 1 + 21 * 3   # visibility + 63 floats
HAND_POSE_LEN  = 2 * TOKEN_PER_HAND   # 128 floats per frame


def _write_hand_pose(path: Path, rng: np.random.Generator):
    """128 space-separated floats (vis flag + 63 xyz per hand × 2 hands)."""
    vals = rng.random(HAND_POSE_LEN).astype(np.float32)
    # First value per hand = visibility flag (1 or 0)
    vals[0] = 1.0
    vals[TOKEN_PER_HAND] = 1.0
    path.write_text(" ".join(f"{v:.6f}" for v in vals) + "\n")


def _write_obj_pose_rt(path: Path, rng: np.random.Generator):
    """17 space-separated floats: obj_id + 4×4 RT matrix."""
    obj_id = float(rng.integers(1, 5))
    rt = rng.random(16).astype(np.float32)
    vals = [obj_id] + rt.tolist()
    path.write_text(" ".join(f"{v:.6f}" for v in vals) + "\n")


def _write_action_label(path: Path, label: int):
    path.write_text(f"{label}\n")


def generate_sequence(seq_dir: Path, label: int, rng: np.random.Generator):
    """Create one fake annotation sequence under seq_dir/cam4/."""
    cam_dir = seq_dir / "cam4"
    for subdir in ("hand_pose", "obj_pose_rt", "action_label"):
        (cam_dir / subdir).mkdir(parents=True, exist_ok=True)

    for i in range(NUM_FRAMES):
        fname = f"{i:04d}.txt"
        _write_hand_pose(cam_dir / "hand_pose"   / fname, rng)
        _write_obj_pose_rt(cam_dir / "obj_pose_rt" / fname, rng)
        _write_action_label(cam_dir / "action_label" / fname, label)


def write_split_file(path: Path, entries: list[dict]):
    """
    Write one split file.
    Columns: id  path  action_label  start_act  end_act  start_frame  end_frame
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["id path action_label start_act end_act start_frame end_frame"]
    for i, e in enumerate(entries):
        rows.append(
            f"{i+1} {e['path']} {e['label']} "
            f"{e['start_act']} {e['end_act']} "
            f"{e['start_frame']} {e['end_frame']}"
        )
    path.write_text("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic H2O data")
    ap.add_argument("--out_dir",   default="data/h2o",
                    help="Root directory for the synthetic dataset")
    ap.add_argument("--num_seqs",  type=int, default=8,
                    help="Number of synthetic sequences (split 70/15/15)")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    # ── Generate sequences and collect split metadata ──────────────────────
    entries = []
    for i in range(args.num_seqs):
        label    = rng.integers(1, NUM_CLASSES + 1)   # 1-indexed raw label
        scene    = "h1"
        subject  = "subject1"
        take_id  = str(i)
        rel_path = f"{subject}/{scene}/{take_id}"

        seq_dir = root / "annotations" / rel_path
        generate_sequence(seq_dir, int(label), rng)

        # Action occupies frames [20, 80] within the sequence
        start_act, end_act = 20, 80
        entries.append({
            "path":        rel_path,
            "label":       int(label),
            "start_act":   start_act,
            "end_act":     end_act,
            "start_frame": 0,
            "end_frame":   NUM_FRAMES - 1,
        })

    # ── Shuffle and split ─────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(entries)
    n = len(entries)
    n_train = max(1, int(n * 0.70))
    n_val   = max(1, int(n * 0.15))

    train_entries = entries[:n_train]
    val_entries   = entries[n_train : n_train + n_val]
    test_entries  = entries[n_train + n_val :] or entries[-1:]   # at least 1

    split_dir = root / "models" / "label_split"
    write_split_file(split_dir / "action_train.txt", train_entries)
    write_split_file(split_dir / "action_val.txt",   val_entries)
    write_split_file(split_dir / "action_test.txt",  test_entries)

    print(f"[generate_synthetic_h2o] Done!")
    print(f"  Root         : {root.resolve()}")
    print(f"  Sequences    : {n} total  ({len(train_entries)} train / "
          f"{len(val_entries)} val / {len(test_entries)} test)")
    print(f"  Frames/seq   : {NUM_FRAMES}")
    print(f"\nNext step:")
    print(f"  python -m src.tests.smoke_test --data_root {root}")


if __name__ == "__main__":
    main()
