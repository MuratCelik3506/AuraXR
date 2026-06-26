"""Best production model (aa45) on real HOT3D + DexYCB segments -> MP4.

green = GT, red = model prediction, gray = real object mesh.
The selected checkpoint is c2h_arch_lstm.pt per results/FAZ_RESULTS.md.
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import trimesh
from scipy.spatial import cKDTree

from dataset import GraspSegments
from model import ControllerToHand
from mano_right import ManoRight, _quat_to_R

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOT3D_ASSET = os.path.join(_REPO, "data/raw/hot3d/assets/Hot3DAssets_assets_assets")
DEXYCB_MODELS = os.path.join(_REPO, "data/raw/dexycb/models")
OUT = os.path.join(_REPO, "results", "mp4")

CATS = ["hook", "power", "wide", "pinch"]
PREFERRED_OBJECTS = {
    "HOT3D": ["mug_white", "bowl", "keyboard", "mouse", "can_parmesan"],
    "DexYCB": ["037_scissors", "006_mustard_bottle", "003_cracker_box", "024_bowl"],
}


def source_name(seg):
    return "HOT3D" if seg["seq_id"].startswith("P") else "DexYCB"


_mesh_cache = {}
def object_mesh(seg, max_faces=3500):
    key = (source_name(seg), seg["obj_uid"], seg["obj_name"])
    if key in _mesh_cache:
        return _mesh_cache[key]

    if source_name(seg) == "HOT3D":
        path = os.path.join(HOT3D_ASSET, f"{seg['obj_uid']}.glb")
    else:
        path = os.path.join(DEXYCB_MODELS, seg["obj_uid"], "textured_simple.obj")
        if not os.path.exists(path):
            path = os.path.join(DEXYCB_MODELS, seg["obj_uid"], "textured.obj")

    mesh = trimesh.load(path, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) > max_faces:
        step = int(np.ceil(len(faces) / max_faces))
        faces = faces[::step]
    _mesh_cache[key] = (verts, faces, path)
    return _mesh_cache[key]


def world_object(seg, t):
    verts, faces, path = object_mesh(seg)
    return verts @ _quat_to_R(seg["obj_q"][t]).T + seg["obj_t"][t], faces, path


def fingertip_surface_cm(seg, t, mano):
    ow, _, _ = world_object(seg, t)
    res = mano.fk(seg["target"][t], seg["betas"][t], seg["wrist_q"][t], seg["wrist_t"][t],
                  want_fingertips=True)
    return float(cKDTree(ow).query(res["fingertips"])[0].mean() * 100.0)


def best_window(seg, mano, pre=32, post=14, max_frames=72):
    L = len(seg["feat"])
    contact = np.asarray(seg["contact"]) > 0.5
    if contact.any():
        runs = []
        i = 0
        while i < L:
            if contact[i]:
                j = i
                while j < L and contact[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        center = (lo + hi - 1) // 2
    else:
        probe = np.linspace(0, L - 1, min(L, 24)).astype(int)
        vals = [fingertip_surface_cm(seg, int(t), mano) for t in probe]
        center = int(probe[int(np.argmin(vals))])

    a = max(0, center - pre)
    b = min(L, center + post)
    if b - a > max_frames:
        b = a + max_frames
    if b - a < min(L, 24):
        need = min(L, 24) - (b - a)
        a = max(0, a - need // 2)
        b = min(L, b + need)
    return a, b


def segment_rank(seg):
    contact = float(np.asarray(seg["contact"]).sum())
    return (contact, len(seg["feat"]))


def pick_segments(ds, mano, per_source=2):
    """Pick real test segments from both sources, preferring varied objects and contact."""
    picks = []
    used_objects = set()
    for src in ["HOT3D", "DexYCB"]:
        candidates = [s for s in ds.segments if source_name(s) == src]
        pref = {name: i for i, name in enumerate(PREFERRED_OBJECTS[src])}
        candidates.sort(key=lambda s: (pref.get(s["obj_name"], 999), -segment_rank(s)[0],
                                       -segment_rank(s)[1], s["obj_name"]))
        src_picks = []
        for seg in candidates:
            if seg["obj_name"] in used_objects:
                continue
            try:
                object_mesh(seg)
                a, b = best_window(seg, mano)
            except Exception as exc:
                print(f"skip mesh/window {src} {seg['obj_name']} {seg['seq_id']} seg{seg['segment_id']}: {exc}")
                continue
            src_picks.append((seg, a, b))
            used_objects.add(seg["obj_name"])
            if len(src_picks) >= per_source:
                break
        picks.extend(src_picks)
    return picks


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    obj_size = ck["args"].get("obj_size", False)
    feat_dim = 16 if obj_size else 13
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=not ck["args"].get("no_prev", False),
                             feat_dim=feat_dim, arch=ck["args"].get("arch", "lstm")).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def render(seg, a, b, pred, mano, out_path, fps=15):
    frames = list(range(a, b))
    catname = CATS[int(seg["cat"][0])] if int(seg["cat"][0]) < len(CATS) else str(int(seg["cat"][0]))
    src = source_name(seg)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    def draw(i):
        t = frames[i]
        ax.clear()
        ow, of, mesh_path = world_object(seg, t)
        ax.plot_trisurf(ow[:, 0], ow[:, 1], ow[:, 2], triangles=of,
                        color="0.55", alpha=0.38, linewidth=0)
        joints = {}
        verts_for_bounds = []
        for key, pose, col in [("GT", seg["target"][t], "tab:green"), ("pred", pred[t], "tab:red")]:
            res = mano.fk(pose, seg["betas"][t], seg["wrist_q"][t], seg["wrist_t"][t], want_mesh=True)
            V = res["verts"]
            joints[key] = res["joints"]
            verts_for_bounds.append(V)
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=mano.faces,
                            color=col, alpha=0.58, linewidth=0)

        mpjpe = np.linalg.norm(joints["GT"] - joints["pred"], axis=1).mean() * 1000.0
        gt_tip = cKDTree(ow).query(mano.fk(seg["target"][t], seg["betas"][t],
                                           seg["wrist_q"][t], seg["wrist_t"][t],
                                           want_fingertips=True)["fingertips"])[0].mean() * 100.0
        pred_tip = cKDTree(ow).query(mano.fk(pred[t], seg["betas"][t],
                                             seg["wrist_q"][t], seg["wrist_t"][t],
                                             want_fingertips=True)["fingertips"])[0].mean() * 100.0
        info = (
            "AuraXR best model: c2h_arch_lstm.pt (aa45, obj-size, FK loss, 7.21mm test)\n"
            f"source: {src} | split: test | object: {seg['obj_name']} | category: {catname}\n"
            f"seq: {seg['seq_id']} | segment: {seg['segment_id']} | frame: {t} ({i+1}/{len(frames)})\n"
            f"GT fingertip->surface: {gt_tip:4.1f} cm | pred fingertip->surface: {pred_tip:4.1f} cm\n"
            f"per-frame MPJPE: {mpjpe:4.1f} mm | green=GT red=pred gray=object"
        )
        ax.text2D(0.0, 1.0, info, transform=ax.transAxes, va="top", ha="left",
                  fontsize=8, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.72", alpha=0.88))

        allp = np.vstack([ow] + verts_for_bounds)
        ctr = np.median(allp, axis=0)
        rr = max(0.12, float((allp.max(0) - allp.min(0)).max()) * 0.58)
        ax.set_xlim(ctr[0] - rr, ctr[0] + rr)
        ax.set_ylim(ctr[1] - rr, ctr[1] + rr)
        ax.set_zlim(ctr[2] - rr, ctr[2] + rr)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=17, azim=-68 + 42 * i / max(1, len(frames) - 1))

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "checkpoints", "c2h_arch_lstm.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model, ck = load_model(args.ckpt, dev)
    mano = ManoRight()
    ds = GraspSegments(args.split, ck["mean"], ck["std"], obj_size=ck["args"].get("obj_size", False))
    picks = pick_segments(ds, mano, per_source=args.per_source)
    if not picks:
        raise RuntimeError("no renderable segments found")

    print(f"model: {os.path.basename(args.ckpt)} | split={args.split} | device={dev}")
    for seg, a, b in picks:
        feat = torch.as_tensor(seg["feat"])[None].to(dev)
        cat = torch.as_tensor(seg["cat"])[None].to(dev)
        with torch.no_grad():
            pred = model.forward_free(feat, cat).cpu().numpy()[0]
        src = source_name(seg).lower()
        out = os.path.join(
            OUT,
            f"best_{src}_{seg['obj_name']}_{seg['seq_id']}_seg{seg['segment_id']}.mp4",
        )
        n = render(seg, a, b, pred, mano, out, fps=args.fps)
        print(f"{source_name(seg):6s} {seg['obj_name']:22s} {seg['seq_id']} seg{seg['segment_id']} "
              f"window[{a}:{b}] {n} frames -> {out}")


if __name__ == "__main__":
    main()
