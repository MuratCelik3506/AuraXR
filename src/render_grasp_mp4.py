"""Final demo MP4: objeyi AVUC ICINDE gosteren (enclosure) segmentleri secip render.

Sorun: 'en uzun segment' cogunlukla uzanma+hafif dokunus -> obje parmak ucunda.
Cozum: enclosure skoru = wrap(parmak-yuzey) + enclose(obje merkezi <-> el merkezi).
En dusuk skorlu (gercek kavrama) segment + pencere secilir, iyi kamerayla render.
yesil=GT, kirmizi=pred(A+C model, free-running).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import trimesh
from scipy.spatial import cKDTree

from dataset import GraspSegments, load_stats
from model import ControllerToHand
from mano_right import ManoRight, _quat_to_R

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(_REPO, "data/raw/hot3d/assets/Hot3DAssets_assets_assets")
OUT = os.path.join(_REPO, "results", "mp4")
DEMO = ["mug_white", "can_parmesan", "bowl"]

_oc = {}
def obj_mesh(uid):
    if uid not in _oc:
        m = trimesh.load(os.path.join(ASSET, f"{uid}.glb"), force="mesh")
        _oc[uid] = (np.asarray(m.vertices), np.asarray(m.faces))
    return _oc[uid]


def grasp_score(s, t, mano):
    """dusuk = obje avucta. wrap(parmak-yuzey ort) + enclose(obje merkezi<->el merkezi)."""
    r = mano.fk(mano.decode_pca(s["target"][t]), s["betas"][t], s["wrist_q"][t], s["wrist_t"][t],
                want_fingertips=True)
    ov, _ = obj_mesh(s["obj_uid"])
    ow = ov @ _quat_to_R(s["obj_q"][t]).T + s["obj_t"][t]
    wrap = cKDTree(ow).query(r["fingertips"])[0].mean()
    enclose = np.linalg.norm(ow.mean(0) - r["joints"].mean(0))
    return wrap + enclose


def best_window(segs, mano):
    """en dusuk enclosure skorlu segment + grasp penceresi."""
    best = (1e9, None, None)
    for s in segs:
        sc = np.array([grasp_score(s, t, mano) for t in range(len(s["feat"]))])
        ti = int(sc.argmin())
        if sc[ti] < best[0]:
            best = (float(sc[ti]), s, sc)
    sc_min, s, sc = best
    L = len(s["feat"])
    thr = sc_min + 0.04  # 4cm tolerans -> sureklI kavrama bolgesi
    lo = ti = int(sc.argmin())
    while lo > 0 and sc[lo - 1] < thr:
        lo -= 1
    hi = ti
    while hi < L - 1 and sc[hi + 1] < thr:
        hi += 1
    a = max(0, lo - 30)  # reach girisi
    b = min(L, hi + 8)
    return s, a, b, sc_min


CATS = ["hook", "power", "wide"]


def render(s, a, b, model, mano, dev, out_path, fps=15):
    feat = torch.as_tensor(s["feat"])[None].to(dev)
    cat = torch.as_tensor(s["cat"])[None].to(dev)
    with torch.no_grad():
        pred = model.forward_free(feat, cat).cpu().numpy()[0]
    ov, of = obj_mesh(s["obj_uid"])
    frames = list(range(a, b))
    catname = CATS[int(s["cat"][0])]

    fig = plt.figure(figsize=(6.6, 6.6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(i):
        t = frames[i]
        ax.clear()
        ow = ov @ _quat_to_R(s["obj_q"][t]).T + s["obj_t"][t]
        ax.plot_trisurf(ow[:, 0], ow[:, 1], ow[:, 2], triangles=of, color="0.55", alpha=0.45, linewidth=0)
        J = {}
        for key, pca, col in [("gt", s["target"][t], "tab:green"), ("pred", pred[t], "tab:red")]:
            r = mano.fk(mano.decode_pca(pca), s["betas"][t], s["wrist_q"][t], s["wrist_t"][t], want_mesh=True)
            V = r["verts"]; J[key] = r
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=mano.faces, color=col, alpha=0.6, linewidth=0)
        # anlik metrikler
        mpjpe = np.linalg.norm(J["gt"]["joints"] - J["pred"]["joints"], axis=1).mean() * 1000
        wrap = cKDTree(ow).query(J["gt"]["fingertips"])[0].mean() * 100
        wrist_obj = np.linalg.norm(s["wrist_t"][t] - ow.mean(0)) * 100
        phase = "GRASP" if wrap < 2.5 else "reach"
        info = (f"AuraXR Controller->Hand  (model A+C: FK-loss + obj-size, test 6.99mm)\n"
                f"object: {s['obj_name']}  | category: {catname}  | split: {s['_split']}\n"
                f"frame {t}  ({i+1}/{len(frames)})   phase: {phase}\n"
                f"GT fingertip->surface: {wrap:4.1f} cm    wrist->obj: {wrist_obj:4.1f} cm\n"
                f"per-frame MPJPE (GT vs pred): {mpjpe:4.1f} mm    [green=GT  red=pred]")
        ax.text2D(0.0, 1.0, info, transform=ax.transAxes, va="top", ha="left",
                  fontsize=8, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ctr = ow.mean(0); rr = 0.13
        ax.set_xlim(ctr[0]-rr, ctr[0]+rr); ax.set_ylim(ctr[1]-rr, ctr[1]+rr); ax.set_zlim(ctr[2]-rr, ctr[2]+rr)
        ax.view_init(elev=16, azim=-60 + 50 * i / max(1, len(frames)))

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000/fps)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2600))
    plt.close()
    return frames


def main():
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ck = torch.load(os.path.join(_REPO, "checkpoints", "c2h_fk_objsize.pt"),
                    map_location=dev, weights_only=False)
    obj_size = ck["args"].get("obj_size", False)
    mean, std = ck["mean"], ck["std"]
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=not ck["args"].get("no_prev", False),
                             feat_dim=16 if obj_size else 13).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    mano = ManoRight()
    os.makedirs(OUT, exist_ok=True)

    # her split'ten en iyi firm-grasp (avucta obje) sec -> generalizasyon + kavrama
    figv = plt.figure(figsize=(15, 5))
    picks = {}
    for ci, sp in enumerate(["train", "val", "test"]):
        segs = [s for s in GraspSegments(sp, mean, std, obj_size=obj_size).segments
                if s["obj_name"] in DEMO]
        for s in segs:
            s["_split"] = sp
        s, a, b, sc = best_window(segs, mano)
        picks[sp] = (s, a, b, sc)
        tg = int(np.argmin([grasp_score(s, t, mano) for t in range(a, b)])) + a
        ov, of = obj_mesh(s["obj_uid"])
        ow = ov @ _quat_to_R(s["obj_q"][tg]).T + s["obj_t"][tg]
        V = mano.fk(mano.decode_pca(s["target"][tg]), s["betas"][tg], s["wrist_q"][tg], s["wrist_t"][tg], want_mesh=True)["verts"]
        ax = figv.add_subplot(1, 3, ci+1, projection="3d")
        ax.plot_trisurf(ow[:, 0], ow[:, 1], ow[:, 2], triangles=of, color="0.55", alpha=0.45, linewidth=0)
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=mano.faces, color="tab:green", alpha=0.6, linewidth=0)
        ax.set_title(f"{sp.upper()}: {s['obj_name']}\nenclosure={sc*100:.1f}cm"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        c = ow.mean(0); r = 0.12
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=16, azim=-60)
        print(f"{sp:5s}: {s['obj_name']:13s} enclosure={sc*100:.1f}cm seg{s['segment_id']} window[{a}:{b}]")
    figv.tight_layout(); figv.savefig(os.path.join(OUT, "verify_grasp.png"), dpi=120); plt.close()
    print("dogrulama -> results/mp4/verify_grasp.png")

    # 3 final mp4 (train/val/test)
    for sp in ["train", "val", "test"]:
        s, a, b, sc = picks[sp]
        out = os.path.join(OUT, f"final_{sp}_{s['obj_name']}.mp4")
        fr = render(s, a, b, model, mano, dev, out)
        print(f"{sp}: {len(fr)} frame -> {out}")


if __name__ == "__main__":
    main()
