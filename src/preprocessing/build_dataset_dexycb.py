"""DexYCB -> kanonik sag-el dataset (Faz 2 veri kaldiraci).

HOT3D builder (build_hot3d_canonical.py) ile AYNI npz semasini uretir; ayni dizine
(seq_DX*.npz) duser, birlesik egitilir. ESKI build_dataset_dexycb.py (sahte random
feature ureten) KULLANILMAZ -- bu, gercek nesne-goreli geometriyi cikarir.

Konvansiyon (probe ile dogrulandi):
  pose_m (N,1,51): [0:3] global rot (axis-angle), [3:48] MANO 45-PCA katsayisi
                   (use_pca=True), [48:51] wrist translation.
    -> finger_aa45 = hands_mean + coeffs @ components   (HOT3D ile ayni; hands_mean dahil)
    -> finger_pca15 = coeffs[:15]                        (HOT3D pca15 ile ayni baz)
  pose_y (N,nobj,7): nesne [quat(xyzw), trans(3)] -- AYNI dunya cercevesi (el ile).
  betas: calibration/mano_<calib>/mano.yml (10).
Sadece sag el (mano_sides 'right'). dt sabit (DexYCB per-frame ts yok).
"""
import os
import csv
import glob
import json
import time
import collections

import numpy as np
import yaml
import trimesh

from utils.mano_right import ManoRight, _quat_to_R
from utils import grasp_taxonomy as tax
from preprocessing.build_hot3d_canonical_full import fingertip_aabb_dist, _rotmat_to_6d, CFG

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEXYCB = os.path.join(_REPO, "data", "raw", "dexycb")
OUT_DIR = os.path.join(_REPO, "data", "processed", "hot3d_canonical")
RESULTS = os.path.join(_REPO, "results")

DT = 1.0 / 30.0   # DexYCB ~30fps (sabit; hiz stats ile normalize)

# DexYCB resmi _YCB_CLASSES: ycb_id (1-21) -> model klasor adi (tax.DEXYCB anahtarlari)
YCB_CLASSES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box',
    4: '005_tomato_soup_can', 5: '006_mustard_bottle', 6: '007_tuna_fish_can',
    7: '008_pudding_box', 8: '009_gelatin_box', 9: '010_potted_meat_can',
    10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block',
    17: '037_scissors', 18: '040_large_marker', 19: '051_large_clamp',
    20: '052_extra_large_clamp', 21: '061_foam_brick'}


def aa_to_quat_wxyz(aa):
    th = np.linalg.norm(aa)
    if th < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([np.cos(th / 2), *(np.sin(th / 2) * (aa / th))])


_bbox_cache = {}
def object_bbox(folder):
    if folder not in _bbox_cache:
        p = os.path.join(DEXYCB, "models", folder, "textured_simple.obj")
        if not os.path.exists(p):
            _bbox_cache[folder] = None
        else:
            mesh = trimesh.load(p, force="mesh")
            _bbox_cache[folder] = (mesh.bounds[0].copy(), mesh.bounds[1].copy())
    return _bbox_cache[folder]


def main():
    m = ManoRight()
    os.makedirs(OUT_DIR, exist_ok=True)
    PRE = CFG.setdefault("pre_contact_window", 30)
    POST = CFG.setdefault("post_contact_window", 5)
    THR = CFG["contact_thresh_m"]
    MINSEG = CFG["min_segment_frames"]

    subj_dirs = sorted(d for d in glob.glob(os.path.join(DEXYCB, "2020*subject*")))
    summ = dict(total_sessions=0, usable_sessions=0, usable_frames=0,
                detected_grasp_segments=0, category_counts=collections.Counter(),
                object_counts=collections.Counter(), skipped_no_cat=0,
                skipped_no_right=0, nan_outlier=0)
    manifest = []
    extents = {}

    print(f"{len(subj_dirs)} subject | thr={THR} minseg={MINSEG} pre={PRE} post={POST}")
    t0 = time.time()
    for sidx, subj in enumerate(subj_dirs, 1):
        sid_prefix = f"DX{sidx:02d}"
        sessions = sorted(s for s in glob.glob(os.path.join(subj, "*")) if os.path.isdir(s))
        for sess in sessions:
            mp = os.path.join(sess, "meta.yml")
            pp = os.path.join(sess, "pose.npz")
            if not (os.path.isfile(mp) and os.path.isfile(pp)):
                continue
            summ["total_sessions"] += 1
            meta = yaml.safe_load(open(mp))
            sides = [x.lower() for x in meta.get("mano_sides", [])]
            if "right" not in sides:
                summ["skipped_no_right"] += 1
                continue
            hand_idx = sides.index("right")

            gi = meta["ycb_grasp_ind"]
            ycb_id = meta["ycb_ids"][gi]
            folder = YCB_CLASSES.get(ycb_id)
            cat = tax.get_category(folder) if folder else None
            if cat is None:
                summ["skipped_no_cat"] += 1
                continue
            bb = object_bbox(folder)
            if bb is None:
                continue
            extents[folder] = (np.asarray(bb[1]) - np.asarray(bb[0])).tolist()

            # betas
            calib = meta["mano_calib"][0]
            mano_yml = os.path.join(DEXYCB, "calibration", f"mano_{calib}", "mano.yml")
            betas = np.array(yaml.safe_load(open(mano_yml))["betas"], np.float64)[:10]
            sh = m.shaped_cached(betas)

            d = np.load(pp)
            pose_m = d["pose_m"]; pose_y = d["pose_y"]
            nF = len(pose_m)

            # per-frame FK + temas
            frames = []   # (fi, aa45, pca15, wt, wq, joints, t_obj, R_obj, q_obj, contact)
            for fi in range(nF):
                pm = pose_m[fi, hand_idx]
                if np.all(pm == 0):
                    continue
                oy = pose_y[fi, gi]
                if np.all(oy == 0) or np.all(oy == -1):
                    continue
                coeffs = pm[3:48]
                aa45 = m.hands_mean + coeffs @ m.hands_components   # full decode (hands_mean dahil)
                wt = np.asarray(pm[48:51], np.float64)
                wq = aa_to_quat_wxyz(pm[0:3])
                res = m.fk(aa45, betas, wq, wt, want_fingertips=True, shaped=sh)
                q_xyzw = oy[:4]
                q_obj = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # -> wxyz
                t_obj = np.asarray(oy[4:7], np.float64)
                R_obj = _quat_to_R(q_obj)
                dist = fingertip_aabb_dist(res["fingertips"], t_obj, q_obj, bb[0], bb[1]).min()
                frames.append(dict(fi=fi, aa45=aa45, pca15=coeffs[:15], wt=wt, wq=wq,
                                   joints=res["joints"], t_obj=t_obj, R_obj=R_obj,
                                   q_obj=q_obj, contact=dist < THR))
            if len(frames) < MINSEG:
                continue

            # contact runs -> segmentler (pre/post pencere)
            contact = np.array([f["contact"] for f in frames])
            nf = len(frames)
            seg_ranges = []
            i = 0
            while i < nf:
                if contact[i]:
                    j = i
                    while j < nf and contact[j]:
                        j += 1
                    if j - i >= MINSEG:
                        s = max(0, i - PRE); e = min(nf, j + POST)
                        seg_ranges.append((s, e))
                    i = j
                else:
                    i += 1
            if not seg_ranges:
                continue

            cols = collections.defaultdict(list)
            seg_id = 0
            cat_id = tax.CATEGORY_ID[cat]
            for (s, e) in seg_ranges:
                prev_rel = None
                for k in range(s, e):
                    f = frames[k]
                    rel_pos = f["R_obj"].T @ (f["wt"] - f["t_obj"])
                    R_rel = f["R_obj"].T @ _quat_to_R(f["wq"])
                    rel6d = _rotmat_to_6d(R_rel)
                    rel_vel = np.zeros(3) if prev_rel is None else (rel_pos - prev_rel) / DT
                    prev_rel = rel_pos
                    dist = np.linalg.norm(rel_pos)
                    if not np.isfinite(rel_pos).all() or dist > 2.0:
                        summ["nan_outlier"] += 1
                        continue
                    cols["rel_pos"].append(rel_pos)
                    cols["rel_rot6d"].append(rel6d)
                    cols["rel_vel"].append(rel_vel)
                    cols["dist"].append([dist])
                    cols["category_id"].append(cat_id)
                    cols["finger_pca15"].append(f["pca15"])
                    cols["finger_aa45"].append(f["aa45"])
                    cols["fk_joints"].append(f["joints"])
                    cols["betas"].append(betas)
                    cols["wrist_world_t"].append(f["wt"])
                    cols["wrist_world_q"].append(f["wq"])
                    cols["obj_world_t"].append(f["t_obj"])
                    cols["obj_world_q"].append(f["q_obj"])
                    cols["frame_t"].append(f["fi"])
                    cols["contact_flag"].append(bool(f["contact"]))
                    cols["segment_id"].append(seg_id)
                    cols["obj_uid"].append(folder)
                    cols["obj_name"].append(folder)
                seg_id += 1

            if not cols["rel_pos"]:
                continue
            arr = {k: np.asarray(v) for k, v in cols.items()}
            n = len(arr["rel_pos"])
            sname = f"{sid_prefix}_{os.path.basename(sess)}"
            np.savez_compressed(os.path.join(OUT_DIR, f"seq_{sname}.npz"), **arr)
            manifest.append(dict(seq_id=sname, split="", n_frames=n, n_segments=seg_id,
                                 objects=folder, categories=cat))
            summ["usable_sessions"] += 1
            summ["usable_frames"] += n
            summ["detected_grasp_segments"] += seg_id
            summ["category_counts"][cat] += n
            summ["object_counts"][folder] += 1
        print(f"  subj {sidx}/{len(subj_dirs)} ({time.time()-t0:.0f}s) "
              f"sessions={summ['usable_sessions']} frames={summ['usable_frames']}")

    # manifest (DexYCB satirlarini ekle; HOT3D manifest'ine dokunma)
    mpath = os.path.join(OUT_DIR, "manifest_dexycb.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_id", "split", "n_frames", "n_segments",
                                          "objects", "categories"])
        w.writeheader(); w.writerows(manifest)
    # extents (YCB) ayri dosyaya
    json.dump(extents, open(os.path.join(OUT_DIR, "object_extents_dexycb.json"), "w"), indent=2)
    summ["category_counts"] = dict(summ["category_counts"])
    summ["object_counts"] = dict(summ["object_counts"])
    json.dump(summ, open(os.path.join(RESULTS, "dexycb_canonical_summary.json"), "w"), indent=2)
    print("\n--- DexYCB ozet ---")
    print(json.dumps({k: summ[k] for k in ["usable_sessions", "usable_frames",
          "detected_grasp_segments", "category_counts", "skipped_no_cat",
          "skipped_no_right"]}, indent=2))
    print(f"yazildi -> {OUT_DIR}/seq_DX*.npz | manifest_dexycb.csv | object_extents_dexycb.json")


if __name__ == "__main__":
    main()
