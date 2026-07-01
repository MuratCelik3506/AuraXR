"""Rebuild split.json from existing dataset.npz — no raw OakInk data needed.

Produces seen_test + unseen_test (object-level stratified) alongside
existing train/val keys. Writes to the same split.json in-place.

Usage:
  python src/preprocessing/rebuild_split.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("data/processed/oakink_canonical")
NPZ_PATH = OUT_DIR / "dataset.npz"
SPLIT_PATH = OUT_DIR / "split.json"


def main() -> None:
    data = np.load(NPZ_PATH, allow_pickle=True)
    name_arr: np.ndarray = data["obj_name"]
    cat_arr: np.ndarray  = data["category"]
    n = len(name_arr)

    rng = np.random.default_rng(seed=42)

    unique_objs  = [str(o) for o in np.unique(name_arr)]
    name_arr_s   = np.array([str(x) for x in name_arr])
    cat_arr_s    = np.array([str(x) for x in cat_arr])
    obj_to_cat   = {o: cat_arr_s[name_arr_s == o][0] for o in unique_objs}
    unique_cats  = sorted(set(obj_to_cat.values()))

    # With only 1 object per category, per-category split would leave train empty.
    # Instead: shuffle all objects globally, assign 80/10/10.
    all_objs_shuffled = list(rng.permutation(unique_objs))
    n_objs = len(all_objs_shuffled)
    n_test = max(1, int(n_objs * 0.1))
    n_val  = max(1, int(n_objs * 0.1))

    obj_test  = all_objs_shuffled[:n_test]
    obj_val   = all_objs_shuffled[n_test:n_test + n_val]
    obj_train = all_objs_shuffled[n_test + n_val:]

    obj_test_set  = {str(o) for o in obj_test}
    obj_val_set   = {str(o) for o in obj_val}
    obj_train_set = {str(o) for o in obj_train}

    all_idx = np.arange(n)
    rng2 = np.random.default_rng(seed=42)

    # sample-level split (train/val by object membership)
    train_idx    = [i for i in all_idx if name_arr_s[i] in obj_train_set]
    val_idx      = [i for i in all_idx if name_arr_s[i] in obj_val_set]
    unseen_idx   = [i for i in all_idx if name_arr_s[i] in obj_test_set]

    # seen_test: 10% of train frames (sampled from train objects)
    n_seen = max(1, int(len(train_idx) * 0.1))
    seen_idx = list(rng2.choice(train_idx, size=min(n_seen, len(train_idx)), replace=False))

    split = {
        "train":       [int(i) for i in train_idx],
        "val":         [int(i) for i in val_idx],
        "seen_test":   sorted(int(i) for i in seen_idx),
        "unseen_test": sorted(int(i) for i in unseen_idx),
        "obj_train":   obj_train,
        "obj_val":     obj_val,
    }

    SPLIT_PATH.write_text(json.dumps(split, indent=2))

    print(f"split.json güncellendi → {SPLIT_PATH}")
    for k, v in split.items():
        print(f"  {k:14s}: {len(v)}")


if __name__ == "__main__":
    main()
