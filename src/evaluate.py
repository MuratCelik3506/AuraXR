"""Faz 2A eval harness: ogrenilebilirlik kaniti.

Metrikler (test split, MPJPE mm via FK):
  - model teacher-forced  (iyimser; referans)
  - model free-running    (gercek deployment kosulu)
  - baseline copy-prev    (GT onceki poz; TF'in trivial oldugunu gosterir)
  - baseline cat-mean     (kategori ortalama poz; statik non-temporal)
  - baseline global-mean
Kategori kirilimi + smoothness(jerk) + sanity viz (GT vs free-running).
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GraspSegments, collate, load_stats
from model import ControllerToHand
from mano_right import ManoRight
import grasp_taxonomy as tax

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_REPO, "checkpoints", "c2h_best.pt")
RESULTS = os.path.join(_REPO, "results")


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def gather(model, loader, dev):
    model.eval()
    K = dict(tf=[], free=[], target=[], betas=[], wq=[], wt=[], joints=[], cat=[])
    with torch.no_grad():
        for b in loader:
            feat = b["feat"].to(dev); cat = b["cat"].to(dev); tgt = b["target"].to(dev)
            tf = model.forward_teacher(feat, cat, tgt).cpu().numpy()
            fr = model.forward_free(feat, cat).cpu().numpy()
            m = b["mask"].numpy().astype(bool)
            K["tf"].append(tf[m]); K["free"].append(fr[m]); K["target"].append(b["target"].numpy()[m])
            K["betas"].append(b["betas"].numpy()[m]); K["wq"].append(b["wrist_q"].numpy()[m])
            K["wt"].append(b["wrist_t"].numpy()[m]); K["joints"].append(b["fk_joints"].numpy()[m])
            K["cat"].append(b["cat"].numpy()[m])
    return {k: np.concatenate(v) for k, v in K.items()}


def mpjpe_frames(pca, betas, wq, wt, gtj, idx, mano):
    """idx frame'leri icin per-frame MPJPE (mm)."""
    e = np.empty(len(idx))
    for n, i in enumerate(idx):
        aa = pca[i]   # cikti zaten aa45 (decode yok)
        j = mano.fk(aa, betas[i], wq[i], wt[i], shaped=mano.shaped_cached(betas[i]))["joints"]
        e[n] = np.linalg.norm(j - gtj[i], axis=1).mean()
    return e * 1000.0


def jerk(pca_seq):
    """basit smoothness: 2. fark normu ortalama (pca uzayinda)."""
    if len(pca_seq) < 3:
        return 0.0
    d2 = pca_seq[2:] - 2 * pca_seq[1:-1] + pca_seq[:-2]
    return float(np.linalg.norm(d2, axis=1).mean())


def sanity_viz(model, dev, mano, split, mean, std, object_name="mug_white"):
    """GT (yesil) vs free-running tahmin (kirmizi) el mesh, bir test segmentinde."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = GraspSegments(split, mean, std)
    seg = next((s for s in ds.segments if s["obj_name"] == object_name), None)
    if seg is None:
        seg = ds.segments[0]
    feat = torch.as_tensor(seg["feat"])[None].to(dev)
    cat = torch.as_tensor(seg["cat"])[None].to(dev)
    with torch.no_grad():
        pred = model.forward_free(feat, cat).cpu().numpy()[0]
    L = len(seg["feat"])
    picks = np.linspace(0, L - 1, 4).astype(int)
    fig = plt.figure(figsize=(16, 4.4))
    for c, p in enumerate(picks):
        ax = fig.add_subplot(1, 4, c + 1, projection="3d")
        for pca, col in [(seg["target"][p], "tab:green"), (pred[p], "tab:red")]:
            aa = pca   # aa45 (decode yok)
            V = mano.fk(aa, seg["betas"][p], seg["wrist_q"][p], seg["wrist_t"][p],
                        want_mesh=True)["verts"]
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=mano.faces,
                            color=col, alpha=0.5, linewidth=0)
        ax.set_title(f"f{p}/{L-1}"); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.suptitle(f"{seg['obj_name']}  green=GT  red=pred(free-running)")
    out = os.path.join(RESULTS, "viz", "eval_pred_vs_gt.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print("viz ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()
    dev = device()
    mano = ManoRight()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    mean, std = ck["mean"], ck["std"]
    obj_size = ck["args"].get("obj_size", False)
    feat_dim = 16 if obj_size else 13
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=not ck["args"].get("no_prev", False),
                             feat_dim=feat_dim, arch=ck["args"].get("arch", "lstm")).to(dev)
    model.load_state_dict(ck["model"])

    # train kategori-ortalama poz (baseline)
    tr = GraspSegments("train", mean, std)
    tr_pose, tr_cat = [], []
    for s in tr.segments:
        tr_pose.append(s["target"]); tr_cat.append(s["cat"])
    tr_pose = np.concatenate(tr_pose); tr_cat = np.concatenate(tr_cat)
    cat_mean = {c: tr_pose[tr_cat == c].mean(0) for c in range(4) if (tr_cat == c).any()}
    glob_mean = tr_pose.mean(0)

    loader = DataLoader(GraspSegments(args.split, mean, std, obj_size=obj_size), batch_size=32,
                        shuffle=False, collate_fn=collate)
    G = gather(model, loader, dev)
    N = len(G["tf"])
    np.random.seed(0)
    idx = np.random.choice(N, min(args.cap, N), replace=False)

    # baseline pca dizileri
    copy_prev = G["target"].copy()
    copy_prev[1:] = G["target"][:-1]            # kaba (segment siniri yok; referans)
    catmean_pred = np.stack([cat_mean.get(c, glob_mean) for c in G["cat"]])
    globmean_pred = np.tile(glob_mean, (N, 1))

    methods = {
        "model_teacher_forced": G["tf"],
        "model_free_running": G["free"],
        "baseline_copy_prev": copy_prev,
        "baseline_cat_mean": catmean_pred,
        "baseline_global_mean": globmean_pred,
    }
    report = {}
    per_frame = {}
    for name, pca in methods.items():
        e = mpjpe_frames(pca, G["betas"], G["wq"], G["wt"], G["joints"], idx, mano)
        per_frame[name] = e
        report[name] = dict(mpjpe_mm=round(float(e.mean()), 2),
                            median_mm=round(float(np.median(e)), 2))

    # kategori kirilimi (model_free vs cat_mean)
    cats = G["cat"][idx]
    bycat = {}
    for c in range(4):
        cm = cats == c
        if cm.sum() == 0:
            continue
        bycat[tax.CATEGORIES[c]] = dict(
            n=int(cm.sum()),
            model_free=round(float(per_frame["model_free_running"][cm].mean()), 2),
            cat_mean=round(float(per_frame["baseline_cat_mean"][cm].mean()), 2))

    # smoothness: free-running vs GT (segment bazli)
    js_free, js_gt = [], []
    te = GraspSegments(args.split, mean, std, obj_size=obj_size)
    with torch.no_grad():
        for s in te.segments[:200]:
            feat = torch.as_tensor(s["feat"])[None].to(dev)
            cat = torch.as_tensor(s["cat"])[None].to(dev)
            fr = model.forward_free(feat, cat).cpu().numpy()[0]
            js_free.append(jerk(fr)); js_gt.append(jerk(s["target"]))

    report["per_category"] = bycat
    report["smoothness_jerk"] = dict(model_free=round(float(np.mean(js_free)), 4),
                                     ground_truth=round(float(np.mean(js_gt)), 4))
    report["n_test_frames"] = int(N)
    report["eval_frames"] = int(len(idx))

    json.dump(report, open(os.path.join(RESULTS, "eval_report.json"), "w"), indent=2)
    print("=== MPJPE (mm) [test, lower=better] ===")
    for name in methods:
        print(f"  {name:24s} {report[name]['mpjpe_mm']:6.2f}  (median {report[name]['median_mm']:.2f})")
    print("\n=== kategori kirilimi (model_free vs cat_mean) ===")
    for c, v in bycat.items():
        print(f"  {c:6s} n={v['n']:5d}  model_free={v['model_free']:.2f}  cat_mean={v['cat_mean']:.2f}")
    print(f"\n smoothness jerk: model_free={report['smoothness_jerk']['model_free']:.4f} "
          f"gt={report['smoothness_jerk']['ground_truth']:.4f}")
    print("rapor -> results/eval_report.json")
    if args.viz:
        sanity_viz(model, dev, mano, args.split, mean, std)


if __name__ == "__main__":
    main()
