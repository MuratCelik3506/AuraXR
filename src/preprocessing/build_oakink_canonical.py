"""OakInk raw anno -> oakink_canonical/dataset.npz + split.json + stats.json + obj_pts/*.npy

OakInk anno format (per-sample PKL):
  general_info/{sample}.pkl -> hand_anno: {hand_pose:(16,4)quat, hand_tsl:(3,), hand_shape:(10,)}
                                obj_anno: (4,4) object transform tensor
  hand_j/{sample}.pkl       -> (21,3) MANO joint positions (world frame)

Filename: {actor}_{seq}_{task}__{datetime}__{cam}__{intent_id}__{obj_id}.pkl
  intent_id -> category via yodaobject_cat.json
  obj_id    -> specific object instance

Output: data/processed/oakink_canonical/
  dataset.npz   pose(N,48) shape(N,10) tsl(N,3) category(N) obj_name(N)
  split.json    train/val/test
  stats.json    wrist_mean/std, pts_mean/std
  obj_pts/      per-category point clouds sampled from OakBase meshes
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

_REPO = Path(__file__).resolve().parents[2]
OAKINK_RAW  = _REPO / "data/raw/oakink"
OAKBASE_DIR = OAKINK_RAW / "OakBase/OakBase"
ANNO_DIR    = OAKINK_RAW / "anno"
META_DIR    = OAKINK_RAW / "shape/meta/metaV2"
OUT_DIR     = _REPO / "data/processed/oakink_canonical"
N_POINTS    = 1024
MAX_SAMPLES = 80000   # üst limit — tüm 314k'yı almaya gerek yok
SEED        = 42

# yodaobject_cat.json ismi != OakBase dizin ismi — normalize et
_CAT_NAME_FIX = {
    # isim normalizasyonu (yodaobject_cat -> OakBase dir)
    "fryingpan": "frying_pan",
    "gamecontroller": "game_controller",
    "cameras": "camera",
    "scissors": "scissor",
    "squeezable": "squeeze_tube",
    "lotion_pump": "lotion_bottle",
    # OakBase'de yok → None
    "pen": None,
    "stapler": None,
    "phone": None,
    "apple": None,
    "banana": None,
    "donut": None,
}


def _to_f32(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.array(t, dtype=np.float32)


def quat_wxyz_to_aa(q_wxyz: np.ndarray) -> np.ndarray:
    """(4,) wxyz quaternion -> (3,) axis-angle."""
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arccos(abs(w))
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z], dtype=np.float64) / sin_half
    if w < 0:
        angle = -angle
    return (axis * angle).astype(np.float32)


def pose16q_to_pose48(pose_16_4: np.ndarray) -> np.ndarray:
    """(16,4) wxyz-quaternion (joint0=wrist, 1-15=fingers) -> (48,) axis-angle MANO order."""
    aa = np.stack([quat_wxyz_to_aa(pose_16_4[i]) for i in range(16)])  # (16,3)
    global_orient = aa[0]    # wrist global rotation
    finger_aa45   = aa[1:].reshape(45)
    return np.concatenate([global_orient, finger_aa45]).astype(np.float32)


def _oakbase_dir(category: str) -> Path | None:
    """Map yodaobject_cat category name -> OakBase directory."""
    fixed = _CAT_NAME_FIX.get(category, category)
    if fixed is None:
        return None
    d = OAKBASE_DIR / fixed
    return d if d.exists() else None


def load_obj_pts(category: str) -> np.ndarray | None:
    """Sample N_POINTS from OakBase PLY point clouds for this category."""
    cat_dir = _oakbase_dir(category)
    if cat_dir is None:
        return None
    rng = np.random.default_rng(0)
    for instance in sorted(cat_dir.iterdir()):
        if not instance.is_dir():
            continue
        ply_parts = sorted(instance.glob("part_*.ply"))
        if not ply_parts:
            ply_parts = sorted(instance.glob("*.ply"))
        if not ply_parts:
            continue
        try:
            all_pts = []
            for p in ply_parts:
                obj = trimesh.load(str(p))
                if hasattr(obj, "vertices") and len(obj.vertices) > 0:
                    all_pts.append(np.array(obj.vertices, dtype=np.float32))
            if not all_pts:
                continue
            pts = np.concatenate(all_pts, axis=0)
            if len(pts) >= N_POINTS:
                idx = rng.choice(len(pts), N_POINTS, replace=False)
            else:
                idx = rng.choice(len(pts), N_POINTS, replace=True)
            return pts[idx].astype(np.float32)
        except Exception:
            continue
    return None


def intent_to_category(intent_id: str, cat_map: dict) -> str:
    return cat_map.get(intent_id, f"intent_{intent_id}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "obj_pts").mkdir(exist_ok=True)

    cat_map: dict[str, str] = json.loads((META_DIR / "yodaobject_cat.json").read_text())

    # collect all sample PKLs
    all_files = sorted((ANNO_DIR / "general_info").glob("*.pkl"))
    print(f"Found {len(all_files)} PKL files — sampling up to {MAX_SAMPLES}")

    rng = np.random.default_rng(SEED)
    if len(all_files) > MAX_SAMPLES:
        chosen = rng.choice(len(all_files), MAX_SAMPLES, replace=False)
        all_files = [all_files[i] for i in sorted(chosen)]

    poses, shapes, tsls, obj_annos, categories, obj_names = [], [], [], [], [], []
    fingertips_world_list: list[np.ndarray] = []  # (5,3) GT fingertip world positions per sample
    pts_cache: dict[str, np.ndarray | None] = {}

    # MANO 21-joint fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
    _MANO_TIP_IDX = [4, 8, 12, 16, 20]
    _HAND_J_DIR = ANNO_DIR / "hand_j"

    skipped_no_pts = 0
    for i, fpath in enumerate(all_files):
        if i % 5000 == 0:
            print(f"  {i}/{len(all_files)} (loaded={len(poses)} skipped_no_pts={skipped_no_pts})", flush=True)
        try:
            gi = pickle.load(open(fpath, "rb"), encoding="latin1")
            ha = gi["hand_anno"]
            pose_q = _to_f32(ha["hand_pose"])   # (16,4) wxyz quaternions
            tsl    = _to_f32(ha["hand_tsl"])     # (3,) wrist world translation
            shape  = _to_f32(ha["hand_shape"])   # (10,)
            # obj_anno: (4,4) object-to-world transform. Store as flat (12,) = R(9) + t(3),
            # i.e. NOT a reshape of the (3,4) [R|t] block — that would interleave rotation
            # and translation columns since reshape is row-major. Reader expects
            # obj_anno_12[:9].reshape(3,3) == R, obj_anno_12[9:12] == t.
            obj_anno_raw = gi.get("obj_anno")
            if obj_anno_raw is None:
                R_obj = np.eye(3, dtype=np.float32)
                t_obj = np.zeros(3, dtype=np.float32)
            else:
                obj_anno_44 = _to_f32(obj_anno_raw)
                R_obj = obj_anno_44[:3, :3]
                t_obj = obj_anno_44[:3, 3]
            if pose_q.shape != (16, 4):
                continue
            # GT fingertip world positions from hand_j (camera frame) via cam_extr inverse.
            # hand_j[0] (wrist, camera frame) == cam_extr^{-1} @ wrist_world = tsl (verified).
            cam_extr_44 = _to_f32(gi["cam_extr"])   # (4,4) world -> camera
            cam_inv = np.linalg.inv(cam_extr_44)
            hj_path = _HAND_J_DIR / fpath.name
            hj = pickle.load(open(hj_path, "rb"), encoding="latin1")
            joints_cam = np.array(hj, dtype=np.float32)  # (21,3) camera frame
            tips_cam = joints_cam[_MANO_TIP_IDX]           # (5,3)
            tips_h = np.concatenate([tips_cam, np.ones((5, 1), dtype=np.float32)], axis=1)  # (5,4)
            tips_world = (cam_inv @ tips_h.T).T[:, :3].astype(np.float32)  # (5,3) world
        except Exception:
            continue

        # intent_id from filename: A01001_0001_0000__date__cam__intent_id__obj_id.pkl
        stem = fpath.stem
        parts = stem.split("__")
        intent_id = parts[3] if len(parts) >= 5 else "0"
        category  = intent_to_category(intent_id, cat_map)
        obj_name  = f"{category}_{intent_id}"

        # load/cache obj_pts keyed by obj_name; skip samples with no geometry
        if obj_name not in pts_cache:
            pts_cache[obj_name] = load_obj_pts(category)
        if pts_cache[obj_name] is None:
            skipped_no_pts += 1
            continue

        pose48 = pose16q_to_pose48(pose_q)
        poses.append(pose48)
        shapes.append(shape)
        tsls.append(tsl)
        obj_annos.append(np.concatenate([R_obj.reshape(9), t_obj]))  # (12,) = R(9) + t(3)
        fingertips_world_list.append(tips_world)  # (5,3) GT world fingertips
        categories.append(category)
        obj_names.append(obj_name)

    print(f"Loaded {len(poses)} samples across {len(set(obj_names))} obj_names "
          f"(skipped {skipped_no_pts} no-geometry)")

    # save obj_pts keyed by obj_name (e.g. bottle_16.npy)
    saved_pts = 0
    for oname, pts in pts_cache.items():
        if pts is not None:
            np.save(OUT_DIR / "obj_pts" / f"{oname}.npy", pts)
            saved_pts += 1
    print(f"Saved {saved_pts} obj_pts files")

    # build arrays
    pose_arr = np.stack(poses).astype(np.float32)
    shape_arr = np.stack(shapes).astype(np.float32)
    tsl_arr = np.stack(tsls).astype(np.float32)
    obj_anno_arr = np.stack(obj_annos).astype(np.float32)        # (N, 12) = R(9) flat + t(3)
    fingertips_arr = np.stack(fingertips_world_list).astype(np.float32)  # (N, 5, 3) GT world
    cat_arr = np.array(categories)
    name_arr = np.array(obj_names)

    np.savez(
        OUT_DIR / "dataset.npz",
        pose=pose_arr, shape=shape_arr, tsl=tsl_arr,
        obj_anno=obj_anno_arr,
        fingertips_world=fingertips_arr,
        category=cat_arr, obj_name=name_arr,
    )
    print(f"Saved dataset.npz — pose: {pose_arr.shape}, obj_anno: {obj_anno_arr.shape}")

    # Object-level, category-stratified split (B2 fix).
    # All samples of the same obj_name stay in the same partition → unseen-object test enabled.
    # Also produce legacy sample-level split as "seen_test" for backwards-compatible comparison.
    name_arr_np = np.array(obj_names)
    cat_arr_np  = np.array(categories)

    # --- object-level split (global shuffle, not per-category) ---
    # Per-category stratification breaks when categories have <3 objects (all go to test).
    # Simple global 70/15/15 split on unique objects is more robust for small datasets.
    unique_objs = list(rng.permutation(np.unique(name_arr_np)))
    n_objs = len(unique_objs)
    n_test = max(1, round(n_objs * 0.15))
    n_val  = max(1, round(n_objs * 0.15))
    # Guarantee at least 1 train object
    if n_test + n_val >= n_objs:
        n_test = max(1, n_objs // 5)
        n_val  = max(1, n_objs // 5)
    obj_test  = unique_objs[:n_test]
    obj_val   = unique_objs[n_test:n_test + n_val]
    obj_train = unique_objs[n_test + n_val:]

    obj_train_set = set(obj_train)
    obj_val_set   = set(obj_val)
    obj_test_set  = set(obj_test)

    all_idx = np.arange(len(poses))
    obj_split = {
        "train":       all_idx[np.isin(name_arr_np, list(obj_train_set))].tolist(),
        "val":         all_idx[np.isin(name_arr_np, list(obj_val_set))].tolist(),
        "unseen_test": all_idx[np.isin(name_arr_np, list(obj_test_set))].tolist(),
    }

    # --- legacy sample-level split (seen_test) ---
    idx_legacy = rng.permutation(len(poses))
    n_val_leg  = max(1, int(len(poses) * 0.1))
    n_test_leg = max(1, int(len(poses) * 0.1))
    split = {
        "train":      idx_legacy[n_test_leg + n_val_leg:].tolist(),
        "val":        idx_legacy[n_test_leg:n_test_leg + n_val_leg].tolist(),
        "seen_test":  idx_legacy[:n_test_leg].tolist(),
        # object-level partitions
        "unseen_test":   obj_split["unseen_test"],
        "obj_val":       obj_split["val"],
        "obj_train":     obj_split["train"],
    }
    (OUT_DIR / "split.json").write_text(json.dumps(split, indent=2))
    print(f"Split (sample-level): train={len(split['train'])} val={len(split['val'])} "
          f"seen_test={len(split['seen_test'])}")
    print(f"Split (object-level): obj_train={len(split['obj_train'])} obj_val={len(split['obj_val'])} "
          f"unseen_test={len(split['unseen_test'])} "
          f"(across {len(obj_train)} / {len(obj_val)} / {len(obj_test)} unique objects)")

    # stats
    train_idx = split["train"]
    wrist = np.concatenate([tsl_arr[train_idx], pose_arr[train_idx, :3]], axis=-1)  # (N,6)
    wrist_mean = wrist.mean(0).tolist()
    wrist_std  = wrist.std(0).tolist()
    pts_all = [v for v in pts_cache.values() if v is not None]
    if pts_all:
        pts_stacked = np.concatenate(pts_all, axis=0)
        pts_mean = pts_stacked.mean(0).tolist()
        pts_std  = pts_stacked.std(0).tolist()
    else:
        pts_mean, pts_std = [0, 0, 0], [1, 1, 1]
    stats = {"wrist_mean": wrist_mean, "wrist_std": wrist_std,
             "pts_mean": pts_mean, "pts_std": pts_std}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print("Saved stats.json")
    print("Done.")


if __name__ == "__main__":
    main()
