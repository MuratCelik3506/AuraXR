"""Kanonik HOT3D segmentini gorsel dogrula: el mesh + obje mesh, reach->grasp.

Done-criterion: el+obje makul; grasp'a dogru dist azalir, parmaklar kapanir; sag-el etiketi.

Kullanim:
    .venv/bin/python src/viz_hot3d_canonical.py --object mug_white
"""
import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import cKDTree

from mano_right import ManoRight

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(_REPO, "data/processed/hot3d_canonical")
ASSET_DIR = os.path.join(_REPO, "data/raw/hot3d/assets/Hot3DAssets_assets_assets")
OUT = os.path.join(_REPO, "results", "viz")


def _quat_to_R(q):
    from mano_right import _quat_to_R as f
    return f(q)


def find_segment(object_name):
    """object_name iceren bir segment bul -> (seq_path, segment_id)."""
    for npz in sorted(glob.glob(os.path.join(CANON, "seq_*.npz"))):
        d = np.load(npz, allow_pickle=True)
        if "obj_name" not in d:
            continue
        names = d["obj_name"].astype(str)
        seg = d["segment_id"]
        mask = names == object_name
        if mask.any():
            # en uzun segmenti sec
            sids = np.unique(seg[mask])
            best = max(sids, key=lambda s: int((seg == s).sum()))
            return npz, int(best)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="mug_white")
    ap.add_argument("--nframes", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    m = ManoRight()

    npz, sid = find_segment(args.object)
    if npz is None:
        print("segment bulunamadi:", args.object); return
    d = np.load(npz, allow_pickle=True)
    idx = np.where(d["segment_id"] == sid)[0]
    print(f"{os.path.basename(npz)} segment {sid}: {len(idx)} frame, obj={args.object}")

    dist = d["dist"][idx, 0]
    contact = d["contact_flag"][idx].astype(bool)
    # parmak acikLigi proxy: parmak-ucu - bilek ortalama mesafe (FK joints'ten)
    print(f"  dist: start={dist[0]:.3f} min={dist.min():.3f} end={dist[-1]:.3f} "
          f"contact_frames={int(contact.sum())}")

    # obje mesh (sabit, ilk frame pozu degisir)
    uid = str(d["obj_uid"][idx[0]])
    omesh = trimesh.load(os.path.join(ASSET_DIR, f"{uid}.glb"), force="mesh")
    ov, of = np.asarray(omesh.vertices), np.asarray(omesh.faces)

    # gosterilecek frame'ler: reach baslangici -> temas
    picks = np.linspace(0, len(idx) - 1, args.nframes).astype(int)
    fig = plt.figure(figsize=(4 * args.nframes, 4.6))
    for c, p in enumerate(picks):
        gi = idx[p]
        aa = d["finger_aa45"][gi]; betas = d["betas"][gi]
        wt = d["wrist_world_t"][gi]; wq = d["wrist_world_q"][gi]
        res = m.fk(aa, betas, wq, wt, want_mesh=True)
        V = res["verts"]
        ot = d["obj_world_t"][gi]; oq = d["obj_world_q"][gi]
        ow = ov @ _quat_to_R(oq).T + ot
        # GERCEK el-yuzeyi <-> obje-yuzeyi minimum mesafe (gorunen temasla tutarli).
        # NOT: d["dist"] = bilek->obje-orijini (~20cm), yuzey mesafesi DEGIL; etiket icin kullanma.
        surf = cKDTree(ow).query(V)[0].min()
        ax = fig.add_subplot(1, args.nframes, c + 1, projection="3d")
        ax.plot_trisurf(ow[:, 0], ow[:, 1], ow[:, 2], triangles=of, color="tab:gray", alpha=0.35, linewidth=0)
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=m.faces, color="tab:red", alpha=0.7, linewidth=0)
        st = "CONTACT" if contact[p] else "reach"
        ax.set_title(f"f{p}/{len(idx)-1} surf={surf*100:.1f}cm {st}\n(red=RIGHT hand)")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        # esit eksen
        allp = np.vstack([V, ow]); ctr = allp.mean(0); r = (allp.max(0) - allp.min(0)).max() / 2
        for setter, ci in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setter(ctr[ci] - r, ctr[ci] + r)

    out = os.path.join(OUT, f"reach_grasp_{args.object}.png")
    fig.suptitle(f"{args.object}  |  {os.path.basename(npz)} seg{sid}  |  reach -> grasp")
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
