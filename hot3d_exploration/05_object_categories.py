"""
05_object_categories.py — Catalogue HOT3D's 33 objects and select POC subset.

What this script answers:
  - What are the 33 objects in HOT3D?
  - How many clips / frames per object category?
  - Which categories have strong bimanual interaction?
  - Which 5 are best for the AuraXR POC subset?
  - What category IDs will the Unity object embedding need?

Usage:
  python 05_object_categories.py
  python 05_object_categories.py --n_clips 100 --plot
"""

import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hot3d_utils import decode_json, load_hot3d, ensure_output_dir, MAX_PLOT_OBJECTS

# Ground-truth object list from models_info.json (all 33 objects confirmed).
# poc_priority: 1=highest priority for POC subset.
HOT3D_OBJECTS = {
    1:  {"name": "holder_black",       "grasp_type": "precision",           "poc_priority": 3},
    2:  {"name": "bowl",               "grasp_type": "cup_grasp",           "poc_priority": 2},
    3:  {"name": "plate_bamboo",       "grasp_type": "flat_palm",           "poc_priority": 3},
    4:  {"name": "spoon_wooden",       "grasp_type": "tool_grasp",          "poc_priority": 3},
    5:  {"name": "potato_masher",      "grasp_type": "tool_grasp",          "poc_priority": 3},
    6:  {"name": "spatula_red",        "grasp_type": "tool_grasp",          "poc_priority": 3},
    7:  {"name": "coffee_pot",         "grasp_type": "handle_grasp",        "poc_priority": 2},
    8:  {"name": "mug_patterned",      "grasp_type": "handle_grasp",        "poc_priority": 1},
    9:  {"name": "mug_white",          "grasp_type": "handle_grasp",        "poc_priority": 1},
    10: {"name": "can_soup",           "grasp_type": "cylindrical_power",   "poc_priority": 1},
    11: {"name": "can_parmesan",       "grasp_type": "cylindrical_power",   "poc_priority": 2},
    12: {"name": "can_tomato_sauce",   "grasp_type": "cylindrical_power",   "poc_priority": 2},
    13: {"name": "bottle_mustard",     "grasp_type": "bottle_grasp",        "poc_priority": 1},
    14: {"name": "bottle_bbq",         "grasp_type": "bottle_grasp",        "poc_priority": 2},
    15: {"name": "bottle_ranch",       "grasp_type": "bottle_grasp",        "poc_priority": 2},
    16: {"name": "vase",               "grasp_type": "cylindrical_power",   "poc_priority": 3},
    17: {"name": "carton_milk",        "grasp_type": "carton_grasp",        "poc_priority": 2},
    18: {"name": "carton_oj",          "grasp_type": "carton_grasp",        "poc_priority": 3},
    19: {"name": "flask",              "grasp_type": "cylindrical_power",   "poc_priority": 3},
    20: {"name": "food_waffles",       "grasp_type": "flat_palm",           "poc_priority": 3},
    21: {"name": "food_vegetables",    "grasp_type": "cup_grasp",           "poc_priority": 3},
    22: {"name": "dumbbell_5lb",       "grasp_type": "power_grasp",         "poc_priority": 3},
    23: {"name": "aria_small",         "grasp_type": "precision",           "poc_priority": 3},
    24: {"name": "cellphone",          "grasp_type": "flat_power",          "poc_priority": 2},
    25: {"name": "holder_gray",        "grasp_type": "precision",           "poc_priority": 3},
    26: {"name": "birdhouse_toy",      "grasp_type": "power_grasp",         "poc_priority": 3},
    27: {"name": "dino_toy",           "grasp_type": "power_grasp",         "poc_priority": 3},
    28: {"name": "keyboard",           "grasp_type": "flat_palm",           "poc_priority": 3},
    29: {"name": "whiteboard_eraser",  "grasp_type": "flat_palm",           "poc_priority": 3},
    30: {"name": "puzzle_toy",         "grasp_type": "precision",           "poc_priority": 3},
    31: {"name": "mouse",              "grasp_type": "precision",           "poc_priority": 2},
    32: {"name": "whiteboard_marker",  "grasp_type": "tripod_grasp",        "poc_priority": 2},
    33: {"name": "dvd_remote",         "grasp_type": "precision",           "poc_priority": 1},
}

_OBJ_INT_CACHE: dict[str, int] = {}


def _obj_int(obj_id: str) -> int:
    if obj_id not in _OBJ_INT_CACHE:
        _OBJ_INT_CACHE[obj_id] = int(obj_id) if obj_id.isdigit() else -1
    return _OBJ_INT_CACHE[obj_id]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_clips", type=int, default=50)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def extract_objects_from_clip(sample: dict) -> dict:
    result = {
        "object_ids": set(),
        "frames_per_object": defaultdict(int),
        "bimanual_frames_per_object": defaultdict(int),
    }

    obj_key = next(
        (k for k in sample if "object" in k.lower() and "pose" in k.lower()),
        next((k for k in sample if "object" in k.lower()), None),
    )
    if obj_key is None:
        return result

    obj_data = decode_json(sample[obj_key])
    if not isinstance(obj_data, dict):
        return result

    mano_key = next((k for k in sample if "mano" in k.lower()), None)
    mano_data = decode_json(sample[mano_key]) if mano_key else {}

    for frame_ts, frame_objs in obj_data.items():
        if not isinstance(frame_objs, dict):
            continue
        for obj_id in frame_objs:
            result["object_ids"].add(obj_id)
            result["frames_per_object"][obj_id] += 1

        if frame_ts in mano_data:
            fd = mano_data[frame_ts]
            is_bimanual = (
                fd.get("left") is not None and fd.get("right") is not None
            )
            if is_bimanual:
                for obj_id in frame_objs:
                    result["bimanual_frames_per_object"][obj_id] += 1

    return result


def print_object_summary(all_results: list) -> dict:
    total_frames   = defaultdict(int)
    bimanual_frames = defaultdict(int)
    clips_per_obj  = defaultdict(int)

    for r in all_results:
        for obj_id in r["object_ids"]:
            clips_per_obj[obj_id]   += 1
            total_frames[obj_id]    += r["frames_per_object"].get(obj_id, 0)
            bimanual_frames[obj_id] += r["bimanual_frames_per_object"].get(obj_id, 0)

    print(f"\n{'='*70}")
    print(f"  OBJECT CATEGORY ANALYSIS ({len(all_results)} clips)")
    print(f"{'='*70}")
    print(f"\n  {'ObjID':<8} {'Name':<22} {'Clips':>6} {'Frames':>8} {'Bi%':>6} {'Grasp':<22} POC")
    print(f"  {'-'*8} {'-'*22} {'-'*6} {'-'*8} {'-'*6} {'-'*22} {'-'*3}")

    for obj_id in sorted(total_frames, key=lambda x: total_frames[x], reverse=True):
        info = HOT3D_OBJECTS.get(_obj_int(obj_id), {})
        name  = info.get("name", f"obj_{obj_id}")
        grasp = info.get("grasp_type", "unknown")
        poc   = "★" * (4 - info.get("poc_priority", 4)) if info.get("poc_priority") else ""
        total = total_frames[obj_id]
        bi_pct = 100 * bimanual_frames[obj_id] / total if total else 0
        print(f"  {obj_id:<8} {name:<22} {clips_per_obj[obj_id]:>6} {total:>8,} {bi_pct:>5.1f}% {grasp:<22} {poc}")

    poc_ids = [oid for oid, info in HOT3D_OBJECTS.items() if info.get("poc_priority") == 1]
    print(f"\n  POC objects (★★★): {[HOT3D_OBJECTS[i]['name'] for i in poc_ids]}")
    print(f"  Object IDs for POC: {poc_ids}")
    print(f"  Covers: handle, cylindrical, bottle, precision grasp types.")
    print(f"  Tag these IDs in Unity for the category embedding lookup.")

    return total_frames


def main():
    args = parse_args()
    dataset = load_hot3d("train")

    all_results = []
    for i, sample in enumerate(dataset):
        if i >= args.n_clips:
            break
        all_results.append(extract_objects_from_clip(sample))
        if i % 10 == 0:
            print(f"  Processed clip {i+1}/{args.n_clips}...")

    total_frames = print_object_summary(all_results)

    if args.plot and total_frames:
        ensure_output_dir()
        ids    = list(total_frames.keys())[:MAX_PLOT_OBJECTS]
        counts = [total_frames[i] for i in ids]
        names  = [HOT3D_OBJECTS.get(_obj_int(i), {}).get("name", str(i)) for i in ids]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(ids)), counts, color="steelblue")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Total frames")
        ax.set_title(f"HOT3D Object Category Frame Counts (top {MAX_PLOT_OBJECTS})")
        plt.tight_layout()
        plt.savefig("output/object_categories.png", dpi=150)
        print(f"\n  [SAVED] output/object_categories.png")
        plt.close()


if __name__ == "__main__":
    main()
