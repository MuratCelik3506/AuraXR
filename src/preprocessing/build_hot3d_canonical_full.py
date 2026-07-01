"""HOT3D Quest3 -> kanonik sag-el dataset (Faz 1A).

Iki mod:
  --report : grasp-frekans raporu (filtre yok) -> results/hot3d_grasp_freq.csv
  (default tam-hat modu sonraki adimda; once 3 obje raporla secilecek)

Streaming zip okuma (tam extract YOK). Sag el = id "1" (HOT3D convention, dual-hand
preview ile dogrulandi). Grasp tespiti: parmak-ucu -> obje AABB (obje-lokal) mesafesi.
"""
import os
import io
import json
import csv
import glob
import zipfile
import argparse
import collections

import numpy as np
import trimesh

from utils.mano_right import ManoRight, _quat_to_R
from utils import grasp_taxonomy as tax

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOT3D = os.path.join(_REPO, "data/raw/hot3d/quest3")
ASSET_DIR = os.path.join(_REPO, "data/raw/hot3d/assets/Hot3DAssets_assets_assets")
RESULTS = os.path.join(_REPO, "results")

RIGHT_HAND_ID = "1"   # HOT3D convention (0=left, 1=right)

# --- config (stats.json'a yazilir) ---
CFG = dict(
    contact_thresh_m=0.03,    # parmak-ucu AABB mesafesi < bu -> temas
    min_segment_frames=5,     # >= bu kadar ardisik temas -> grasp segmenti
    report_stride=2,          # rapor modunda her N. frame
)


# --- obje bbox cache (GLB local bounds) --------------------------------------
_bbox_cache = {}
def object_bbox(uid):
    if uid not in _bbox_cache:
        p = os.path.join(ASSET_DIR, f"{uid}.glb")
        if not os.path.exists(p):
            _bbox_cache[uid] = None
        else:
            mesh = trimesh.load(p, force="mesh")
            _bbox_cache[uid] = (mesh.bounds[0].copy(), mesh.bounds[1].copy())
    return _bbox_cache[uid]


def fingertip_aabb_dist(tips_world, t_obj, q_obj, lo, hi):
    """her parmak-ucu icin obje-lokal AABB mesafesi (ic=0). -> (5,)"""
    R = _quat_to_R(q_obj)
    pl = (tips_world - t_obj) @ R          # world->local
    clamped = np.clip(pl, lo, hi)
    return np.linalg.norm(pl - clamped, axis=1)


def point_aabb_dist(point_world, t_obj, q_obj, lo, hi):
    """Tek nokta icin obje-lokal AABB en yakin yuzey mesafesi (ic=0)."""
    R = _quat_to_R(q_obj)
    pl = (point_world - t_obj) @ R
    clamped = np.clip(pl, lo, hi)
    return float(np.linalg.norm(pl - clamped))


# --- sequence parsing (streaming) --------------------------------------------
def parse_sequence(seq_dir):
    """yield (ts, right_hand_pose_dict, objects_dict, uid2name).
    right_hand_pose_dict: {'pose','wrist_xform','betas'} veya None."""
    hz = glob.glob(os.path.join(seq_dir, "*hand_data.zip"))
    gz = glob.glob(os.path.join(seq_dir, "*ground_truth.zip"))
    if not hz or not gz:
        return
    # objeler + metadata
    with zipfile.ZipFile(gz[0]) as z:
        meta = json.load(z.open("metadata.json"))
        dyn = collections.defaultdict(dict)
        for r in csv.DictReader(io.TextIOWrapper(z.open("dynamic_objects.csv"))):
            ts = int(r["timestamp[ns]"])
            t = np.array([float(r["t_wo_x[m]"]), float(r["t_wo_y[m]"]), float(r["t_wo_z[m]"])])
            q = np.array([float(r["q_wo_w"]), float(r["q_wo_x"]), float(r["q_wo_y"]), float(r["q_wo_z"])])
            dyn[ts][r["object_uid"]] = (t, q)
    uid2name = dict(zip(meta["object_uids"], meta["object_names"]))
    # el trajektorisi
    with zipfile.ZipFile(hz[0]) as z, z.open("mano_hand_pose_trajectory.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            ts = rec["timestamp_ns"]
            hp = rec.get("hand_poses", {})
            rh = hp.get(RIGHT_HAND_ID)
            objs = dyn.get(ts)
            if rh is None or objs is None:
                continue
            yield ts, rh, objs, uid2name


def split_of(seq_dir):
    return "train" if os.sep + "train" + os.sep in seq_dir + os.sep else "test"


# --- report mode -------------------------------------------------------------
def run_report():
    m = ManoRight()
    os.makedirs(RESULTS, exist_ok=True)
    # agg[(name,split)] = dict(present, contact, segments, seqs(set))
    agg = collections.defaultdict(lambda: dict(present=0, contact=0, segments=0, seqs=set()))
    seq_dirs = sorted(glob.glob(os.path.join(HOT3D, "train", "*"))) + \
               sorted(glob.glob(os.path.join(HOT3D, "test", "*")))
    print(f"{len(seq_dirs)} sekans taraniyor (stride={CFG['report_stride']})...")
    for si, seq in enumerate(seq_dirs):
        split = split_of(seq)
        name = os.path.basename(seq)
        # her obje icin ardisik temas sayaci (segment tespiti)
        run_len = collections.defaultdict(int)
        seen_obj = set()
        fi = 0
        for ts, rh, objs, uid2name in parse_sequence(seq):
            fi += 1
            if fi % CFG["report_stride"]:
                continue
            aa = m.decode_pca(rh["pose"])
            wx = rh["wrist_xform"]
            tips = m.fk(aa, rh["betas"], wx["q_wxyz"], wx["t_xyz"], want_mesh=True)["fingertips"]
            # her objeye min parmak-ucu mesafesi
            best_uid, best_d = None, 1e9
            present_uids = []
            for uid, (t, q) in objs.items():
                bb = object_bbox(uid)
                if bb is None:
                    continue
                present_uids.append(uid)
                d = fingertip_aabb_dist(tips, t, q, bb[0], bb[1]).min()
                if d < best_d:
                    best_uid, best_d = uid, d
            for uid in present_uids:
                key = (uid2name[uid], split)
                agg[key]["present"] += 1
                agg[key]["seqs"].add(name)
            # temas: en yakin obje esik altindaysa
            contact_uid = best_uid if best_d < CFG["contact_thresh_m"] else None
            for uid in present_uids:
                if uid == contact_uid:
                    key = (uid2name[uid], split)
                    agg[key]["contact"] += 1
                    run_len[uid] += 1
                    if run_len[uid] == CFG["min_segment_frames"]:
                        agg[key]["segments"] += 1
                else:
                    run_len[uid] = 0
        if (si + 1) % 25 == 0:
            print(f"  {si+1}/{len(seq_dirs)}")

    # CSV
    out = os.path.join(RESULTS, "hot3d_grasp_freq.csv")
    rows = []
    for (nm, split), d in agg.items():
        rows.append(dict(object=nm, category=tax.get_category(nm) or "", split=split,
                         present_frames=d["present"], contact_frames=d["contact"],
                         grasp_segments=d["segments"], n_sequences=len(d["seqs"])))
    rows.sort(key=lambda r: -r["contact_frames"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["object", "category", "split", "present_frames",
                                          "contact_frames", "grasp_segments", "n_sequences"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nyazildi -> {out}")
    # ozet: kategori bazli en cok temas eden objeler
    print("\n--- en cok temas (contact_frames) ---")
    by_obj = collections.defaultdict(lambda: [0, 0, ""])
    for r in rows:
        by_obj[r["object"]][0] += r["contact_frames"]
        by_obj[r["object"]][1] += r["grasp_segments"]
        by_obj[r["object"]][2] = r["category"]
    for nm, (cf, sg, cat) in sorted(by_obj.items(), key=lambda x: -x[1][0])[:18]:
        print(f"  {nm:18s} {cat:6s} contact={cf:6d} segments={sg:4d}")


def _rotmat_to_6d(R):
    """3x3 -> 6D surekli temsil (ilk iki kolon)."""
    return np.concatenate([R[:, 0], R[:, 1]])


def _quat_conj_rot(q):
    return _quat_to_R(q)


def run_build(categories):
    """Tam hat: secilen kategorilerdeki tum objeler icin reach-to-grasp segmentleri ->
    seq-bazli kanonik npz + manifest + stats + summary."""
    import time
    m = ManoRight()
    out_dir = os.path.join(_REPO, "data", "processed", "hot3d_canonical")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    PRE = CFG.setdefault("pre_contact_window", 30)    # ~1 sn reach
    POST = CFG.setdefault("post_contact_window", 5)
    THR = CFG["contact_thresh_m"]
    MINSEG = CFG["min_segment_frames"]
    catset = set(categories)

    seq_dirs = sorted(glob.glob(os.path.join(HOT3D, "train", "*"))) + \
               sorted(glob.glob(os.path.join(HOT3D, "test", "*")))

    manifest = []
    field_shapes = {}
    # running moments for 13-dim input [rel_pos3, rel_rot6d6, rel_vel3, dist1] (train only)
    msum = np.zeros(13); msq = np.zeros(13); mcnt = 0
    summ = dict(total_sequences=0, total_right_hand_frames=0, usable_frames=0,
                detected_right_hand_id=RIGHT_HAND_ID, detected_grasp_segments=0,
                segment_frames=0, category_counts=collections.Counter(),
                object_counts=collections.Counter(),
                dropped_frames_by_reason=collections.Counter(),
                nan_outlier_counts=0)

    print(f"{len(seq_dirs)} sekans | kategoriler={sorted(catset)} | "
          f"thr={THR} minseg={MINSEG} pre={PRE} post={POST}")
    t0 = time.time()
    for si, seq in enumerate(seq_dirs):
        split = split_of(seq)
        sname = os.path.basename(seq)
        # 1) sekansi buffer'la + per-frame FK
        F = []
        for ts, rh, objs, uid2name in parse_sequence(seq):
            summ["total_right_hand_frames"] += 1
            aa = m.decode_pca(rh["pose"])
            wx = rh["wrist_xform"]
            wt = np.asarray(wx["t_xyz"]); wq = np.asarray(wx["q_wxyz"])
            sh = m.shaped_cached(rh["betas"])
            res = m.fk(aa, rh["betas"], wq, wt, want_fingertips=True, shaped=sh)
            F.append(dict(ts=ts, pca=np.asarray(rh["pose"]), aa=aa, wt=wt, wq=wq,
                          betas=np.asarray(rh["betas"])[:10], tips=res["fingertips"],
                          joints=res["joints"], objs=objs, u2n=uid2name))
        if not F:
            continue
        summ["total_sequences"] += 1
        nF = len(F)

        # 2) hedef objeler (kategori filtresi)
        cand_uids = {}
        for fr in F:
            for uid in fr["objs"]:
                nm = fr["u2n"].get(uid)
                cat = tax.get_category(nm)
                if cat in catset:
                    cand_uids[uid] = (nm, cat)

        # 3) her obje: per-frame temas -> runs -> segmentler
        seg_records = []  # her segment: list of frame-feature dict
        seg_meta = []     # (object_name, category, split)
        for uid, (nm, cat) in cand_uids.items():
            bb = object_bbox(uid)
            if bb is None:
                continue
            contact = np.zeros(nF, dtype=bool)
            present = np.zeros(nF, dtype=bool)
            for i, fr in enumerate(F):
                op = fr["objs"].get(uid)
                if op is None:
                    continue
                present[i] = True
                d = fingertip_aabb_dist(fr["tips"], op[0], op[1], bb[0], bb[1]).min()
                contact[i] = d < THR
            # runs
            i = 0
            while i < nF:
                if contact[i]:
                    j = i
                    while j < nF and contact[j]:
                        j += 1
                    if j - i >= MINSEG:
                        s = max(0, i - PRE); e = min(nF, j + POST)
                        frames_idx = [k for k in range(s, e) if present[k]]
                        if len(frames_idx) >= MINSEG:
                            seg_records.append((uid, frames_idx, contact))
                            seg_meta.append((nm, cat, split))
                    i = j
                else:
                    i += 1

        if not seg_records:
            continue

        # 4) segmentleri kanonik feature'lara cevir
        cols = collections.defaultdict(list)
        seg_id = 0
        for (uid, fidx, contact), (nm, cat, sp) in zip(seg_records, seg_meta):
            cat_id = tax.CATEGORY_ID[cat]
            prev_rel = None; prev_ts = None
            for k in fidx:
                fr = F[k]
                op = fr["objs"][uid]
                R_obj = _quat_to_R(op[1]); t_obj = op[0]
                rel_pos = R_obj.T @ (fr["wt"] - t_obj)
                R_rel = R_obj.T @ _quat_to_R(fr["wq"])
                rel6d = _rotmat_to_6d(R_rel)
                if prev_rel is None:
                    rel_vel = np.zeros(3)
                else:
                    dt = max(1e-6, (fr["ts"] - prev_ts) / 1e9)
                    rel_vel = (rel_pos - prev_rel) / dt
                prev_rel = rel_pos; prev_ts = fr["ts"]
                dist = point_aabb_dist(fr["wt"], t_obj, op[1], bb[0], bb[1])
                rel_norm = np.linalg.norm(rel_pos)
                if not np.isfinite(rel_pos).all() or rel_norm > 2.0:
                    summ["nan_outlier_counts"] += 1
                    summ["dropped_frames_by_reason"]["nan_or_outlier"] += 1
                    continue
                cols["rel_pos"].append(rel_pos)
                cols["rel_rot6d"].append(rel6d)
                cols["rel_vel"].append(rel_vel)
                cols["dist"].append([dist])
                cols["category_id"].append(cat_id)
                cols["finger_pca15"].append(fr["pca"])
                cols["finger_aa45"].append(fr["aa"])
                cols["fk_joints"].append(fr["joints"])
                cols["betas"].append(fr["betas"])
                cols["wrist_world_t"].append(fr["wt"])
                cols["wrist_world_q"].append(fr["wq"])
                cols["obj_world_t"].append(t_obj)
                cols["obj_world_q"].append(op[1])
                cols["frame_t"].append(fr["ts"])
                cols["contact_flag"].append(bool(contact[k]))
                cols["segment_id"].append(seg_id)
                cols["obj_uid"].append(uid)
                cols["obj_name"].append(nm)
                # stats (train only)
                if sp == "train":
                    v = np.concatenate([rel_pos, rel6d, rel_vel, [dist]])
                    msum += v; msq += v * v; mcnt += 1
            summ["object_counts"][nm] += 1
            summ["detected_grasp_segments"] += 1
            seg_id += 1

        if not cols["rel_pos"]:
            continue
        arr = {k: np.asarray(v) for k, v in cols.items()}
        for k, v in arr.items():
            field_shapes[k] = list(v.shape[1:]) or [1]
        n = len(arr["rel_pos"])
        summ["usable_frames"] += n
        summ["segment_frames"] += n
        for c in arr["category_id"]:
            summ["category_counts"][tax.CATEGORIES[c]] += 1
        np.savez_compressed(os.path.join(out_dir, f"seq_{sname}.npz"), **arr)
        manifest.append(dict(seq_id=sname, split=split, n_frames=n, n_segments=seg_id,
                             objects=";".join(sorted({mn for mn, _, _ in seg_meta})),
                             categories=";".join(sorted({c for _, c, _ in seg_meta}))))
        if (si + 1) % 25 == 0:
            print(f"  {si+1}/{len(seq_dirs)}  ({time.time()-t0:.0f}s) "
                  f"frames={summ['segment_frames']} segs={summ['detected_grasp_segments']}")

    # manifest
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_id", "split", "n_frames", "n_segments",
                                          "objects", "categories"])
        w.writeheader(); w.writerows(manifest)

    # stats.json (convention + normalizasyon)
    mean = (msum / mcnt).tolist() if mcnt else []
    std = (np.sqrt(np.maximum(msq / mcnt - (msum / mcnt) ** 2, 1e-12))).tolist() if mcnt else []
    stats = dict(
        canonical_convention=dict(
            frame="object-centric relative", handedness="right-handed (HOT3D source)",
            up_axis="HOT3D world (y-up)", units="meters",
            rotation_format="6D continuous (first two columns of R_rel)",
            note="Unity left-handed flip Faz 3'te uygulanir; veri standardi burada sabit."),
        right_hand_id=RIGHT_HAND_ID, categories=tax.CATEGORIES,
        selected_categories=sorted(catset), config=CFG,
        input_feature_order=["rel_pos(3)", "rel_rot6d(6)", "rel_vel(3)", "dist(1)"],
        input_mean=mean, input_std=std, train_frames_for_stats=mcnt,
        field_shapes=field_shapes)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # summary.json (veri sagligi)
    summ["category_counts"] = dict(summ["category_counts"])
    summ["object_counts"] = dict(summ["object_counts"])
    summ["dropped_frames_by_reason"] = dict(summ["dropped_frames_by_reason"])
    with open(os.path.join(RESULTS, "hot3d_canonical_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)

    print(f"\nyazildi -> {out_dir}/ (manifest, seq_*.npz, stats.json)")
    print(f"summary -> {RESULTS}/hot3d_canonical_summary.json")
    print(json.dumps({k: summ[k] for k in ["total_sequences", "usable_frames",
          "detected_grasp_segments", "category_counts", "nan_outlier_counts"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--categories", nargs="+", default=["hook", "power", "wide"])
    args = ap.parse_args()
    if args.report:
        run_report()
    elif args.build:
        run_build(args.categories)
    else:
        print("Kullanim: --report | --build [--categories hook power wide]")
