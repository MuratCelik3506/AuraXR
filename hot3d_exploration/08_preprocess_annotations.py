"""
08_preprocess_annotations.py — Extract per-frame training features from downloaded annotation ZIPs.

Input (per sequence, from 06_download_annotations.py output):
  data/{device}/{split}/{seq_id}/
      *_hand_data.zip     → mano_hand_pose_trajectory.jsonl, umetrack_hand_user_profile.json
      *_ground_truth.zip  → dynamic_objects.csv, headset_trajectory.csv,
                            camera_models.json, metadata.json, masks/*.csv

Output per sequence:
  data/preprocessed/{device}/{split}/{seq_id}.npz
  Fields:
    timestamps_ns        [N]        nanosecond timestamps of valid frames
    mano_pose_h0         [N, 15]    MANO θ for hand-0 (HOT3D 15-DoF per-joint)
    mano_betas_h0        [N, 10]    MANO β shape coefficients
    mano_wrist_t_h0      [N, 3]     wrist world position (xyz)
    mano_wrist_q_h0      [N, 4]     wrist world orientation (w,x,y,z)
    mano_pose_h1         [N, 15]    same for hand-1 (set to NaN if absent)
    mano_betas_h1        [N, 10]
    mano_wrist_t_h1      [N, 3]
    mano_wrist_q_h1      [N, 4]
    hand_h0_valid        [N]        bool — hand-0 annotation present
    hand_h1_valid        [N]        bool — hand-1 annotation present
    ctrl_t_h0            [N, 3]     controller proxy position  = wrist position
    ctrl_q_h0            [N, 4]     controller proxy quaternion = wrist orientation
    ctrl_t_h1            [N, 3]
    ctrl_q_h1            [N, 4]
    ctrl_grip_h0         [N, 1]     grip proxy (mean finger flexion, 0–1)
    ctrl_grip_h1         [N, 1]
    ctrl_trigger_h0      [N, 1]     trigger proxy (index flexion, 0–1)
    ctrl_trigger_h1      [N, 1]
    nearest_obj_cat_h0   [N]        BOP category ID (1–33) of nearest object to hand-0
    nearest_obj_centroid_h0  [N, 3] nearest object centroid in world space
    nearest_obj_bbox_h0  [N, 3]     bounding box half-extents (approx from category)
    nearest_obj_cat_h1   [N]
    nearest_obj_centroid_h1  [N, 3]
    nearest_obj_bbox_h1  [N, 3]
    headset_t            [N, 3]     headset position (world space)
    headset_q            [N, 4]     headset orientation (w,x,y,z)
    quality_mask         [N]        bool — True = all key quality checks passed

Metadata stored as .npz attributes (via separate metadata .json file):
    device, seq_id, participant_id, split

Usage:
    python 08_preprocess_annotations.py
    python 08_preprocess_annotations.py --device quest3
    python 08_preprocess_annotations.py --workers 4
    python 08_preprocess_annotations.py --dry_run     # count sequences, no output
"""

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

DATA_DIR   = Path("../data")
OUT_DIR    = DATA_DIR / "preprocessed"
TEST_PIDS  = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}

# Approximate bounding-box half-extents per BOP object ID (metres).
# Derived from object mesh inspection — good enough for affordance features.
OBJ_BBOX = {
    1:  [0.04, 0.04, 0.10],   # holder_black
    2:  [0.08, 0.08, 0.05],   # bowl
    3:  [0.14, 0.14, 0.02],   # plate_bamboo
    4:  [0.02, 0.02, 0.16],   # spoon_wooden
    5:  [0.03, 0.03, 0.22],   # potato_masher
    6:  [0.04, 0.02, 0.28],   # spatula_red
    7:  [0.09, 0.12, 0.18],   # coffee_pot
    8:  [0.08, 0.06, 0.10],   # mug_patterned
    9:  [0.08, 0.06, 0.10],   # mug_white
    10: [0.04, 0.04, 0.08],   # can_soup
    11: [0.06, 0.06, 0.10],   # can_parmesan
    12: [0.04, 0.04, 0.06],   # can_tomato_sauce
    13: [0.04, 0.04, 0.15],   # bottle_mustard
    14: [0.04, 0.04, 0.22],   # bottle_bbq
    15: [0.04, 0.04, 0.22],   # bottle_ranch
    16: [0.06, 0.06, 0.18],   # vase
    17: [0.06, 0.04, 0.18],   # carton_milk
    18: [0.06, 0.04, 0.18],   # carton_oj
    19: [0.04, 0.04, 0.18],   # flask
    20: [0.14, 0.10, 0.04],   # food_waffles
    21: [0.08, 0.08, 0.08],   # food_vegetables
    22: [0.05, 0.05, 0.30],   # dumbbell_5lb
    23: [0.06, 0.04, 0.02],   # aria_small
    24: [0.04, 0.08, 0.16],   # cellphone
    25: [0.04, 0.04, 0.10],   # holder_gray
    26: [0.08, 0.08, 0.10],   # birdhouse_toy
    27: [0.06, 0.06, 0.10],   # dino_toy
    28: [0.18, 0.08, 0.03],   # keyboard
    29: [0.06, 0.04, 0.02],   # whiteboard_eraser
    30: [0.10, 0.10, 0.03],   # puzzle_toy
    31: [0.06, 0.12, 0.04],   # mouse
    32: [0.01, 0.01, 0.14],   # whiteboard_marker
    33: [0.03, 0.12, 0.02],   # dvd_remote
}
DEFAULT_BBOX = [0.05, 0.05, 0.10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
        return list(reader)


def _parse_masks(zf: zipfile.ZipFile) -> dict[str, dict[int, bool]]:
    """Load all 8 quality mask CSVs → {mask_name: {timestamp_ns: bool}}."""
    masks = {}
    for info in zf.infolist():
        if info.filename.startswith("masks/") and info.filename.endswith(".csv"):
            mname = Path(info.filename).stem
            mapping: dict[int, bool] = {}
            with zf.open(info.filename) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
                for row in reader:
                    ts = int(row["timestamp[ns]"])
                    val = row["mask"].strip().lower() == "true"
                    # If any stream for this timestamp fails, mark False
                    if ts not in mapping:
                        mapping[ts] = val
                    else:
                        mapping[ts] = mapping[ts] and val
            masks[mname] = mapping
    return masks


def _parse_dynamic_objects(zf: zipfile.ZipFile, uid_to_bop: dict[str, int]
                            ) -> dict[int, dict[str, np.ndarray]]:
    """
    Returns {timestamp_ns: {obj_uid_str: {"t": [3], "q": [4], "bop_id": int}}}.
    """
    rows = _read_csv_from_zip(zf, "dynamic_objects.csv")
    result: dict[int, dict] = defaultdict(dict)
    for row in rows:
        ts  = int(row["timestamp[ns]"])
        uid = row["object_uid"]
        t   = np.array([float(row["t_wo_x[m]"]),
                         float(row["t_wo_y[m]"]),
                         float(row["t_wo_z[m]"])], dtype=np.float32)
        q   = np.array([float(row["q_wo_w"]),
                         float(row["q_wo_x"]),
                         float(row["q_wo_y"]),
                         float(row["q_wo_z"])], dtype=np.float32)
        bop_id = uid_to_bop.get(uid, -1)
        result[ts][uid] = {"t": t, "q": q, "bop_id": bop_id}
    return result


def _parse_headset(zf: zipfile.ZipFile) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Returns {timestamp_ns: (t[3], q[4])}."""
    rows = _read_csv_from_zip(zf, "headset_trajectory.csv")
    result = {}
    for row in rows:
        ts = int(row["timestamp[ns]"])
        t  = np.array([float(row["t_wo_x[m]"]),
                        float(row["t_wo_y[m]"]),
                        float(row["t_wo_z[m]"])], dtype=np.float32)
        q  = np.array([float(row["q_wo_w"]),
                        float(row["q_wo_x"]),
                        float(row["q_wo_y"]),
                        float(row["q_wo_z"])], dtype=np.float32)
        result[ts] = (t, q)
    return result


def _parse_mano_trajectory(zf: zipfile.ZipFile
                            ) -> dict[int, dict[str, dict]]:
    """
    Returns {timestamp_ns: {hand_id_str: {pose, betas, wrist_xform}}}.
    pose  — np.float32 [15]
    betas — np.float32 [10]
    t     — np.float32 [3]
    q     — np.float32 [4]  (w,x,y,z)
    """
    result: dict[int, dict] = {}
    with zf.open("mano_hand_pose_trajectory.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            ts    = entry["timestamp_ns"]
            hands = {}
            for hid, hdata in entry["hand_poses"].items():
                pose  = np.array(hdata["pose"],   dtype=np.float32)   # [15]
                betas = np.array(hdata["betas"],  dtype=np.float32)   # [10]
                t     = np.array(hdata["wrist_xform"]["t_xyz"],  dtype=np.float32)
                q     = np.array(hdata["wrist_xform"]["q_wxyz"], dtype=np.float32)
                hands[hid] = {"pose": pose, "betas": betas, "t": t, "q": q}
            result[ts] = hands
    return result


def _grip_proxy(pose: np.ndarray) -> float:
    """Rough grip proxy: mean of finger joint flexion magnitudes (0-1 clipped)."""
    if pose is None or len(pose) < 15:
        return 0.0
    return float(np.clip(np.mean(np.abs(pose)) / 1.5, 0.0, 1.0))


def _trigger_proxy(pose: np.ndarray) -> float:
    """Index finger flexion proxy: use joints 3-5 (index finger in standard MANO)."""
    if pose is None or len(pose) < 6:
        return 0.0
    index_joints = pose[3:6]  # approximate index finger joints
    return float(np.clip(np.mean(np.abs(index_joints)) / 1.5, 0.0, 1.0))


def _nearest_object(wrist_t: np.ndarray,
                    obj_frame: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (bop_id, centroid[3], bbox_half[3]) of nearest object."""
    if not obj_frame:
        return 0, np.zeros(3, np.float32), np.array(DEFAULT_BBOX, np.float32)

    best_uid   = None
    best_dist  = np.inf
    for uid, info in obj_frame.items():
        d = np.linalg.norm(wrist_t - info["t"])
        if d < best_dist:
            best_dist = d
            best_uid  = uid

    info   = obj_frame[best_uid]
    bop_id = info["bop_id"]
    bbox   = np.array(OBJ_BBOX.get(bop_id, DEFAULT_BBOX), dtype=np.float32)
    return bop_id, info["t"].copy(), bbox


NaN3 = np.full(3, np.nan, np.float32)
NaN4 = np.full(4, np.nan, np.float32)
NaN10 = np.full(10, np.nan, np.float32)
NaN15 = np.full(15, np.nan, np.float32)


# ---------------------------------------------------------------------------
# Main per-sequence processor
# ---------------------------------------------------------------------------

def process_sequence(seq_dir: Path, device: str, split: str) -> dict | None:
    """
    Process one sequence directory that contains *_hand_data.zip + *_ground_truth.zip.
    Returns a dict of numpy arrays, or None on failure.
    """
    seq_id = seq_dir.name

    hand_zip_paths = sorted(seq_dir.glob("*hand_data*.zip"))
    gt_zip_paths   = sorted(seq_dir.glob("*ground_truth*.zip"))

    if not hand_zip_paths or not gt_zip_paths:
        return None

    hand_zip_path = hand_zip_paths[0]
    gt_zip_path   = gt_zip_paths[0]

    try:
        with zipfile.ZipFile(gt_zip_path) as gt_zip:
            # Metadata: object_uid → bop_id
            with gt_zip.open("metadata.json") as f:
                meta = json.load(f)
            uid_to_bop = dict(zip(meta["object_uids"],
                                  [int(b) for b in meta["object_bop_uids"]]))
            participant_id = meta["participant_id"]

            # Quality masks
            masks = _parse_masks(gt_zip)

            # Object 6DoF per frame
            obj_by_ts = _parse_dynamic_objects(gt_zip, uid_to_bop)

            # Headset pose per frame
            headset_by_ts = _parse_headset(gt_zip)

        with zipfile.ZipFile(hand_zip_path) as hand_zip:
            mano_by_ts = _parse_mano_trajectory(hand_zip)

    except Exception as e:
        print(f"  [SKIP] {seq_id}: {e}")
        return None

    if not mano_by_ts:
        return None

    # Critical masks: hand pose available + QA pass
    qa_mask    = masks.get("mask_qa_pass",             {})
    hand_mask  = masks.get("mask_hand_pose_available", {})

    all_ts = sorted(mano_by_ts.keys())
    N = len(all_ts)
    if N == 0:
        return None

    # Pre-allocate output arrays
    timestamps         = np.array(all_ts, dtype=np.int64)
    mano_pose_h0       = np.full((N, 15), np.nan, np.float32)
    mano_betas_h0      = np.full((N, 10), np.nan, np.float32)
    mano_wrist_t_h0    = np.full((N, 3),  np.nan, np.float32)
    mano_wrist_q_h0    = np.full((N, 4),  np.nan, np.float32)
    mano_pose_h1       = np.full((N, 15), np.nan, np.float32)
    mano_betas_h1      = np.full((N, 10), np.nan, np.float32)
    mano_wrist_t_h1    = np.full((N, 3),  np.nan, np.float32)
    mano_wrist_q_h1    = np.full((N, 4),  np.nan, np.float32)
    hand_h0_valid      = np.zeros(N, dtype=bool)
    hand_h1_valid      = np.zeros(N, dtype=bool)
    ctrl_t_h0          = np.zeros((N, 3), np.float32)
    ctrl_q_h0          = np.tile([1,0,0,0], (N, 1)).astype(np.float32)
    ctrl_grip_h0       = np.zeros((N, 1), np.float32)
    ctrl_trigger_h0    = np.zeros((N, 1), np.float32)
    ctrl_t_h1          = np.zeros((N, 3), np.float32)
    ctrl_q_h1          = np.tile([1,0,0,0], (N, 1)).astype(np.float32)
    ctrl_grip_h1       = np.zeros((N, 1), np.float32)
    ctrl_trigger_h1    = np.zeros((N, 1), np.float32)
    nearest_cat_h0     = np.zeros(N, dtype=np.int32)
    nearest_centroid_h0 = np.zeros((N, 3), np.float32)
    nearest_bbox_h0    = np.zeros((N, 3), np.float32)
    nearest_cat_h1     = np.zeros(N, dtype=np.int32)
    nearest_centroid_h1 = np.zeros((N, 3), np.float32)
    nearest_bbox_h1    = np.zeros((N, 3), np.float32)
    headset_t_arr      = np.zeros((N, 3), np.float32)
    headset_q_arr      = np.tile([1,0,0,0], (N, 1)).astype(np.float32)
    quality_mask       = np.zeros(N, dtype=bool)

    for i, ts in enumerate(all_ts):
        # Quality: both QA-pass and hand-pose-available
        qa_ok   = qa_mask.get(ts,   True)
        hand_ok = hand_mask.get(ts, True)
        quality_mask[i] = qa_ok and hand_ok

        # Headset
        if ts in headset_by_ts:
            ht, hq = headset_by_ts[ts]
            headset_t_arr[i] = ht
            headset_q_arr[i] = hq

        # MANO hands
        hands     = mano_by_ts[ts]
        hand_ids  = sorted(hands.keys())
        obj_frame = obj_by_ts.get(ts, {})

        for slot, hid in enumerate(hand_ids[:2]):
            hd = hands[hid]
            if slot == 0:
                mano_pose_h0[i]  = hd["pose"]
                mano_betas_h0[i] = hd["betas"]
                mano_wrist_t_h0[i] = hd["t"]
                mano_wrist_q_h0[i] = hd["q"]
                hand_h0_valid[i]   = True
                ctrl_t_h0[i]       = hd["t"]
                ctrl_q_h0[i]       = hd["q"]
                ctrl_grip_h0[i]    = _grip_proxy(hd["pose"])
                ctrl_trigger_h0[i] = _trigger_proxy(hd["pose"])
                cat, cen, bbox     = _nearest_object(hd["t"], obj_frame)
                nearest_cat_h0[i]      = cat
                nearest_centroid_h0[i] = cen
                nearest_bbox_h0[i]     = bbox
            else:
                mano_pose_h1[i]  = hd["pose"]
                mano_betas_h1[i] = hd["betas"]
                mano_wrist_t_h1[i] = hd["t"]
                mano_wrist_q_h1[i] = hd["q"]
                hand_h1_valid[i]   = True
                ctrl_t_h1[i]       = hd["t"]
                ctrl_q_h1[i]       = hd["q"]
                ctrl_grip_h1[i]    = _grip_proxy(hd["pose"])
                ctrl_trigger_h1[i] = _trigger_proxy(hd["pose"])
                cat, cen, bbox     = _nearest_object(hd["t"], obj_frame)
                nearest_cat_h1[i]      = cat
                nearest_centroid_h1[i] = cen
                nearest_bbox_h1[i]     = bbox

    return {
        # arrays
        "timestamps_ns":          timestamps,
        "mano_pose_h0":           mano_pose_h0,
        "mano_betas_h0":          mano_betas_h0,
        "mano_wrist_t_h0":        mano_wrist_t_h0,
        "mano_wrist_q_h0":        mano_wrist_q_h0,
        "mano_pose_h1":           mano_pose_h1,
        "mano_betas_h1":          mano_betas_h1,
        "mano_wrist_t_h1":        mano_wrist_t_h1,
        "mano_wrist_q_h1":        mano_wrist_q_h1,
        "hand_h0_valid":          hand_h0_valid,
        "hand_h1_valid":          hand_h1_valid,
        "ctrl_t_h0":              ctrl_t_h0,
        "ctrl_q_h0":              ctrl_q_h0,
        "ctrl_grip_h0":           ctrl_grip_h0,
        "ctrl_trigger_h0":        ctrl_trigger_h0,
        "ctrl_t_h1":              ctrl_t_h1,
        "ctrl_q_h1":              ctrl_q_h1,
        "ctrl_grip_h1":           ctrl_grip_h1,
        "ctrl_trigger_h1":        ctrl_trigger_h1,
        "nearest_cat_h0":         nearest_cat_h0,
        "nearest_centroid_h0":    nearest_centroid_h0,
        "nearest_bbox_h0":        nearest_bbox_h0,
        "nearest_cat_h1":         nearest_cat_h1,
        "nearest_centroid_h1":    nearest_centroid_h1,
        "nearest_bbox_h1":        nearest_bbox_h1,
        "headset_t":              headset_t_arr,
        "headset_q":              headset_q_arr,
        "quality_mask":           quality_mask,
        # metadata strings saved as separate keys
        "_device":                device,
        "_seq_id":                seq_id,
        "_participant_id":        participant_id,
        "_split":                 split,
    }


def save_sequence(arrays: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Separate numeric arrays from string metadata
    numeric = {k: v for k, v in arrays.items() if not k.startswith("_")}
    # Save npz
    np.savez_compressed(str(out_path), **numeric)
    # Save companion metadata JSON
    meta = {k[1:]: v for k, v in arrays.items() if k.startswith("_")}
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_seq_dirs(device: str) -> list[tuple[Path, str, str]]:
    """Returns list of (seq_dir, device_str, split_str)."""
    result = []
    for split in ("train", "test"):
        base = DATA_DIR / device / split
        if not base.exists():
            continue
        for seq_dir in sorted(base.iterdir()):
            if seq_dir.is_dir():
                result.append((seq_dir, device, split))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  choices=["quest3", "aria", "both"], default="both")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    devices = ["quest3", "aria"] if args.device == "both" else [args.device]

    all_tasks: list[tuple[Path, str, str]] = []
    for dev in devices:
        all_tasks.extend(collect_seq_dirs(dev))

    print(f"\nFound {len(all_tasks)} sequence directories to process.")

    if args.dry_run:
        for seq_dir, dev, split in all_tasks[:10]:
            out = OUT_DIR / dev / split / (seq_dir.name + ".npz")
            done = "✓" if out.exists() else " "
            print(f"  [{done}] {dev}/{split}/{seq_dir.name}")
        if len(all_tasks) > 10:
            print(f"  ... and {len(all_tasks) - 10} more")
        print("\n[DRY RUN] Remove --dry_run to start preprocessing.")
        return

    ok = skip = fail = 0

    def _run(task):
        seq_dir, dev, split = task
        out_path = OUT_DIR / dev / split / (seq_dir.name + ".npz")
        if out_path.exists() and not args.overwrite:
            return "skip", seq_dir.name
        arrays = process_sequence(seq_dir, dev, split)
        if arrays is None:
            return "fail", seq_dir.name
        save_sequence(arrays, out_path)
        N      = len(arrays["timestamps_ns"])
        n_good = int(arrays["quality_mask"].sum())
        return "ok", f"{seq_dir.name}  ({N} frames, {n_good} valid)"

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run, t): t for t in all_tasks}
            for fut in tqdm(as_completed(futs), total=len(all_tasks), desc="Preprocessing"):
                status, info = fut.result()
                if status == "ok":   ok   += 1
                elif status == "skip": skip += 1
                else:                fail += 1
    else:
        for task in tqdm(all_tasks, desc="Preprocessing"):
            status, info = _run(task)
            if status == "ok":
                ok += 1
                tqdm.write(f"  ✓ {info}")
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                tqdm.write(f"  ✗ {info}")

    print(f"\n[DONE] ok={ok}  skip={skip}  fail={fail}")
    print(f"Output: {OUT_DIR}")
    print("Next: run 09_build_dataset.py to assemble training windows.")


if __name__ == "__main__":
    main()
