"""Tek doğruluk kaynağı: tüm değerlendirme metrikleri (model değişince otomatik üretilir).

full_eval() -> all-joint + parmak-ucu MPJPE, per-source (HOT3D/DexYCB), per-category,
per-object, statik cat_mean baseline + kazanç, jerk. train.py egitim sonunda cagirir;
compare_variants.py ve standalone analiz de buradan kullanir.

Metrik: free-running MPJPE (mm), joint-space, FK ile (temsil-bagimsiz; aa45 cikti).
"""
import os
import csv
import json
import collections

import numpy as np
import torch

from dataset import GraspSegments
import grasp_taxonomy as tax

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO, "results")
DISTAL = [3, 6, 9, 12, 15]   # uc (distal) eklemler — parmak ucu vekili (MANO 16-joint)


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _cat_mean_train(mean, std, obj_size):
    """TRAIN kategori-ortalama hedef poz (aa45) — statik baseline."""
    tr = GraspSegments("train", mean, std, obj_size=obj_size)
    tp = np.concatenate([s["target"] for s in tr.segments])
    tc = np.concatenate([s["cat"] for s in tr.segments])
    cm = {c: tp[tc == c].mean(0) for c in range(len(tax.CATEGORIES)) if (tc == c).any()}
    return cm, tp.mean(0)


def full_eval(model, mano, mean, std, obj_size, dev, split="test", tag="model", write=True, cap=None):
    """Kapsamli degerlendirme. -> dict (ayrica results/eval_<tag>.json + by_object_<tag>.csv).

    cap: ~bu kadar frame'e ulasana dek segment alt-kumesi (hizli karsilastirma icin;
         None = tum test). Segment butunlugu korunur (AR/recurrence icin gerekli).
    """
    model.eval()
    ds = GraspSegments(split, mean, std, obj_size=obj_size)
    cat_mean, glob_mean = _cat_mean_train(mean, std, obj_size)

    segments = ds.segments
    if cap is not None:
        order = np.random.RandomState(0).permutation(len(segments))
        segments, acc = [], 0
        for i in order:
            segments.append(ds.segments[i]); acc += len(ds.segments[i]["feat"])
            if acc >= cap:
                break

    err_all, err_tip, err_cm = [], [], []
    cats, srcs, objs = [], [], []
    jfree, jgt = [], []
    with torch.no_grad():
        for s in segments:
            feat = torch.as_tensor(s["feat"])[None].to(dev)
            cat = torch.as_tensor(s["cat"])[None].to(dev)
            pred = model.forward_free(feat, cat).cpu().numpy()[0]   # aa45
            src = "DexYCB" if s["seq_id"].startswith("DX") else "HOT3D"
            jfree.append(_jerk(pred)); jgt.append(_jerk(s["target"]))
            for t in range(len(pred)):
                gj = s["fk_joints"][t]
                sh = mano.shaped_cached(s["betas"][t])
                j = mano.fk(pred[t], s["betas"][t], s["wrist_q"][t], s["wrist_t"][t], shaped=sh)["joints"]
                c = int(s["cat"][t])
                cmpose = cat_mean.get(c, glob_mean)
                cmj = mano.fk(cmpose, s["betas"][t], s["wrist_q"][t], s["wrist_t"][t], shaped=sh)["joints"]
                err_all.append(np.linalg.norm(j - gj, axis=1).mean() * 1000)
                err_tip.append(np.linalg.norm(j[DISTAL] - gj[DISTAL], axis=1).mean() * 1000)
                err_cm.append(np.linalg.norm(cmj - gj, axis=1).mean() * 1000)
                cats.append(c); srcs.append(src); objs.append(s["obj_name"])

    err_all = np.array(err_all); err_tip = np.array(err_tip); err_cm = np.array(err_cm)
    cats = np.array(cats); srcs = np.array(srcs); objs = np.array(objs)

    rep = dict(tag=tag, split=split, n_frames=int(len(err_all)),
               mpjpe_mm=round(float(err_all.mean()), 2),
               fingertip_mm=round(float(err_tip.mean()), 2),
               cat_mean_baseline_mm=round(float(err_cm.mean()), 2),
               gain_vs_static_mm=round(float(err_cm.mean() - err_all.mean()), 2),
               jerk_free=round(float(np.mean(jfree)), 4), jerk_gt=round(float(np.mean(jgt)), 4))
    # per-source
    rep["by_source"] = {}
    for sname in ["HOT3D", "DexYCB"]:
        m = srcs == sname
        if m.any():
            rep["by_source"][sname] = dict(mpjpe=round(float(err_all[m].mean()), 2), n=int(m.sum()))
    # per-category (model vs static)
    rep["by_category"] = {}
    for c in range(4):
        m = cats == c
        if m.any():
            rep["by_category"][tax.CATEGORIES[c]] = dict(
                mpjpe=round(float(err_all[m].mean()), 2),
                static=round(float(err_cm[m].mean()), 2),
                gain=round(float(err_cm[m].mean() - err_all[m].mean()), 2), n=int(m.sum()))
    # per-object
    rep["by_object"] = {}
    for o in sorted(set(objs.tolist())):
        m = objs == o
        rep["by_object"][o] = dict(mpjpe=round(float(err_all[m].mean()), 2),
                                   source=str(srcs[m][0]), category=tax.get_category(o),
                                   n=int(m.sum()))

    if write:
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(rep, open(os.path.join(RESULTS, f"eval_{tag}.json"), "w"), indent=2)
        with open(os.path.join(RESULTS, f"by_object_{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["object", "source", "category", "mpjpe_mm", "n"])
            for o, d in sorted(rep["by_object"].items(), key=lambda kv: kv[1]["mpjpe"]):
                w.writerow([o, d["source"], d["category"], d["mpjpe"], d["n"]])
    return rep


def _jerk(seq):
    if len(seq) < 3:
        return 0.0
    d2 = seq[2:] - 2 * seq[1:-1] + seq[:-2]
    return float(np.linalg.norm(d2, axis=1).mean())


def print_report(rep):
    print(f"\n=== EVAL [{rep['tag']}] {rep['split']} ({rep['n_frames']} frame) ===")
    print(f"  MPJPE={rep['mpjpe_mm']}mm  fingertip={rep['fingertip_mm']}mm  "
          f"static(cat_mean)={rep['cat_mean_baseline_mm']}mm  kazanc={rep['gain_vs_static_mm']}mm")
    print(f"  jerk free={rep['jerk_free']} gt={rep['jerk_gt']}")
    print("  kaynak:", {k: f"{v['mpjpe']}mm(n={v['n']})" for k, v in rep["by_source"].items()})
    print("  kategori:", {k: f"{v['mpjpe']}/{v['static']}(+{v['gain']})" for k, v in rep["by_category"].items()})
    bo = sorted(rep["by_object"].items(), key=lambda kv: kv[1]["mpjpe"])
    print(f"  obje en iyi: {bo[0][0]}={bo[0][1]['mpjpe']}  en kotu: {bo[-1][0]}={bo[-1][1]['mpjpe']}  ({len(bo)} obje)")


if __name__ == "__main__":
    import argparse
    import torch as _t
    from model import ControllerToHand
    from mano_right import ManoRight
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="arch_lstm")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    dev = _t.device("cpu")
    ck = _t.load(os.path.join(_REPO, "checkpoints", f"c2h_{args.tag}.pt"),
                 map_location=dev, weights_only=False)
    obj_size = ck["args"].get("obj_size", False)
    feat_dim = 16 if obj_size else 13
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=not ck["args"].get("no_prev", False),
                             feat_dim=feat_dim, arch=ck["args"].get("arch", "lstm")).to(dev)
    model.load_state_dict(ck["model"])
    rep = full_eval(model, ManoRight(), ck["mean"], ck["std"], obj_size, dev, args.split, args.tag)
    print_report(rep)
