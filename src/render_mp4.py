"""En iyi model (A+C) ile train/val/test'ten birer segmentin GT-vs-pred animasyonu -> MP4.

yesil = GT (gercek), kirmizi = model tahmini (free-running). Obje gri mesh.
"""
import os
import glob
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import trimesh

from dataset import GraspSegments, load_stats
from model import ControllerToHand
from mano_right import ManoRight

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(_REPO, "data/raw/hot3d/assets/Hot3DAssets_assets_assets")
OUT = os.path.join(_REPO, "results", "mp4")
DEMO = ["mug_white", "can_parmesan", "bowl"]


def pick_segment(ds):
    cand = [s for s in ds.segments if s["obj_name"] in DEMO]
    cand = cand or ds.segments
    return max(cand, key=lambda s: len(s["feat"]))


def render(seg, model, mano, dev, out_path, fps=15):
    feat = torch.as_tensor(seg["feat"])[None].to(dev)
    cat = torch.as_tensor(seg["cat"])[None].to(dev)
    with torch.no_grad():
        pred = model.forward_free(feat, cat).cpu().numpy()[0]
    L = len(seg["feat"])
    om = trimesh.load(os.path.join(ASSET, f"{seg['obj_uid']}.glb"), force="mesh")
    ov, of = np.asarray(om.vertices), np.asarray(om.faces)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(t):
        ax.clear()
        ot = seg["obj_t"][t]; oq = seg["obj_q"][t]
        from mano_right import _quat_to_R
        ow = ov @ _quat_to_R(oq).T + ot
        ax.plot_trisurf(ow[:, 0], ow[:, 1], ow[:, 2], triangles=of, color="0.6", alpha=0.3, linewidth=0)
        for pca, col, lab in [(seg["target"][t], "tab:green", "GT"), (pred[t], "tab:red", "pred")]:
            aa = mano.decode_pca(pca)
            V = mano.fk(aa, seg["betas"][t], seg["wrist_q"][t], seg["wrist_t"][t], want_mesh=True)["verts"]
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=mano.faces, color=col, alpha=0.55, linewidth=0)
        ax.set_title(f"{seg['obj_name']}  f{t}/{L-1}   green=GT  red=pred")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        allp = np.vstack([V, ow]); ctr = allp.mean(0); r = (allp.max(0) - allp.min(0)).max() / 2
        ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r); ax.set_zlim(ctr[2]-r, ctr[2]+r)

    anim = FuncAnimation(fig, draw, frames=L, interval=1000/fps)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close()
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "checkpoints", "c2h_fk_objsize.pt"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    obj_size = ck["args"].get("obj_size", False)
    mean, std = ck["mean"], ck["std"]
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=not ck["args"].get("no_prev", False),
                             feat_dim=16 if obj_size else 13).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    mano = ManoRight()

    for split in ["train", "val", "test"]:
        ds = GraspSegments(split, mean, std, obj_size=obj_size)
        seg = pick_segment(ds)
        out = os.path.join(OUT, f"{split}_{seg['obj_name']}.mp4")
        n = render(seg, model, mano, dev, out)
        print(f"{split}: {seg['seq_id']} seg{seg['segment_id']} {seg['obj_name']} ({n} frame) -> {out}")


if __name__ == "__main__":
    main()
