"""Birkaç sahne icin GT (yesil) vs model-tahmini (kirmizi) el mesh videosu uret.

Her sahne: en uzun test segmenti -> free-running tahmin. Obje mesh gri.
Baslikta gercek el-yuzeyi <-> obje-yuzeyi min mesafe (surf, cm) + temas bayragi.
Bilek konum/yonu GT'den; model sadece parmak pozunu tahmin eder -> el govdesi
GT ile ayni yere gider, fark sadece kavrama sikiliginda.

Kullanim:
    .venv/bin/python src/viz_pred_video.py
    .venv/bin/python src/viz_pred_video.py --objects mug_white bowl can_soup --fps 15
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import GraspSegments
from model import ControllerToHand
from mano_right import ManoRight, _quat_to_R

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_REPO, "checkpoints", "c2h_best.pt")
ASSET = os.path.join(_REPO, "data/raw/hot3d/assets/Hot3DAssets_assets_assets")
CANON = os.path.join(_REPO, "data/processed/hot3d_canonical")
OUT = os.path.join(_REPO, "results", "viz")
DEFAULT_OBJECTS = ["mug_white", "bowl", "can_parmesan", "potato_masher", "plate_bamboo"]


def find_uid(seq_id, seg_id):
    f = [p for p in glob.glob(os.path.join(CANON, "seq_*.npz")) if seq_id in p]
    if not f:
        return None
    d = np.load(f[0], allow_pickle=True)
    idx = np.where(d["segment_id"] == seg_id)[0]
    return str(d["obj_uid"][idx[0]]) if len(idx) else None


def render_object(obj_name, seg, pred, mano, fps, max_frames, stride_cap):
    uid = find_uid(seg["seq_id"], seg["segment_id"])
    if uid is None:
        print(f"  [{obj_name}] uid bulunamadi, atla"); return None
    om = trimesh.load(os.path.join(ASSET, f"{uid}.glb"), force="mesh")
    ov, of = np.asarray(om.vertices), np.asarray(om.faces)

    L = len(seg["feat"])
    frames = np.arange(L)
    if L > max_frames:
        frames = np.linspace(0, L - 1, max_frames).astype(int)

    # tum karelerde sabit eksen + sabit kamera (titremesin)
    allp = []
    geom = []  # (VG, VP, ow, surfG, surfP, contact, p)
    for p in frames:
        ot, oq = seg["obj_t"][p], seg["obj_q"][p]
        ow = ov @ _quat_to_R(oq).T + ot
        aaG = mano.decode_pca(seg["target"][p])
        VG = mano.fk(aaG, seg["betas"][p], seg["wrist_q"][p], seg["wrist_t"][p], want_mesh=True)["verts"]
        aaP = mano.decode_pca(pred[p])
        VP = mano.fk(aaP, seg["betas"][p], seg["wrist_q"][p], seg["wrist_t"][p], want_mesh=True)["verts"]
        tree = cKDTree(ow)
        surfG = tree.query(VG)[0].min()
        surfP = tree.query(VP)[0].min()
        geom.append((VG, VP, ow, surfG, surfP, bool(seg["contact"][p]), int(p)))
        allp.append(VG); allp.append(VP); allp.append(ow)
    allp = np.vstack(allp)
    ctr = allp.mean(0); r = (allp.max(0) - allp.min(0)).max() / 2 * 1.05

    tmp = os.path.join(OUT, f"_frames_{obj_name}")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for n, (VG, VP, ow, sG, sP, ct, p) in enumerate(geom):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(ow[:, 0], ow[:, 1], ow[:, 2], triangles=of, color="0.6", alpha=0.30, linewidth=0)
        ax.plot_trisurf(VG[:, 0], VG[:, 1], VG[:, 2], triangles=mano.faces, color="tab:green", alpha=0.45, linewidth=0)
        ax.plot_trisurf(VP[:, 0], VP[:, 1], VP[:, 2], triangles=mano.faces, color="tab:red", alpha=0.55, linewidth=0)
        st = "CONTACT" if ct else "reach"
        ax.set_title(f"{obj_name}  f{p}/{len(seg['feat'])-1}  {st}\n"
                     f"green=GT (surf {sG*100:4.1f}cm)   red=PRED (surf {sP*100:4.1f}cm)")
        for setter, ci in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setter(ctr[ci] - r, ctr[ci] + r)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.view_init(elev=18, azim=-60)
        fig.savefig(os.path.join(tmp, f"f{n:04d}.png"), dpi=100)
        plt.close(fig)

    out = os.path.join(OUT, f"pred_vs_gt_{obj_name}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(tmp, "f%04d.png"),
           "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp)
    surfP = np.array([g[4] for g in geom]); surfG = np.array([g[3] for g in geom])
    print(f"  [{obj_name}] {len(geom)} kare -> {os.path.basename(out)} | "
          f"GTmin={surfG.min()*100:.1f}cm PREDmin={surfP.min()*100:.1f}cm")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=140)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    dev = torch.device("cpu")
    mano = ManoRight()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    mean, std = ck["mean"], ck["std"]
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=not ck["args"].get("no_prev", False)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    ds = GraspSegments(args.split, mean, std)
    # obje basina en uzun segment
    best = {}
    for s in ds.segments:
        nm = s["obj_name"]
        if nm not in best or len(s["feat"]) > len(best[nm]["feat"]):
            best[nm] = s

    for obj in args.objects:
        seg = best.get(obj)
        if seg is None:
            print(f"  [{obj}] {args.split} split'te yok, atla"); continue
        feat = torch.as_tensor(seg["feat"])[None].to(dev)
        cat = torch.as_tensor(seg["cat"])[None].to(dev)
        with torch.no_grad():
            pred = model.forward_free(feat, cat).cpu().numpy()[0]
        render_object(obj, seg, pred, mano, args.fps, args.max_frames, None)


if __name__ == "__main__":
    main()
