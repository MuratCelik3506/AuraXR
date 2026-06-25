"""compare_sources_video.py — Generate GT vs Predicted MP4 videos for each data source.

Produces one video per source (HOT3D, ARCTIC, DexYCB) using the same model checkpoint
and HOT3D normalization stats so comparisons are fair.

Usage:
    python src/compare_sources_video.py --hand right
    python src/compare_sources_video.py --hand right --n_seqs 3
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, str(Path(__file__).parent))
from mano_fk import MANOForwardKinematics
from model import SDFLSTMModel

WRIST_DIMS = slice(11, 17)

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15),
]


def infer_feat_dim(sd):
    w = sd.get("feat_proj.0.weight")
    return int(w.shape[1]) if w is not None else 29


def infer_orientation_aware(sd):
    w = sd.get("obj_inj.0.weight")
    return bool(w is not None and int(w.shape[1]) == 99)


def load_sequences(h5_path, split, master_norm, n_seqs=2):
    """Load up to n_seqs sequences, preferring those with contact frames."""
    with h5py.File(h5_path) as f:
        g = f[split]
        features  = g["features"][:]
        sdf_feats = g["sdf_features"][:]
        targets   = g["targets"][:]
        wrist_rot = g["wrist_rot_6d"][:]
        seq_ids   = g["sequence_id"][:]
        frame_idx = g["frame_index"][:]
        is_mirror = g["is_mirror"][:] if "is_mirror" in g else np.zeros_like(seq_ids)
        obj_ids   = g["obj_id"][:]
        contact_name = "contact_v2" if "contact_v2" in g else "contact"
        contact   = g[contact_name][:]

    seq_to_pos: dict = {}
    for i in range(len(seq_ids)):
        key = (int(seq_ids[i]), int(is_mirror[i]))
        seq_to_pos.setdefault(key, []).append(i)
    for positions in seq_to_pos.values():
        positions.sort(key=lambda i: frame_idx[i])

    # Sort sequences by contact frame count (descending)
    seq_list = [(k, pos) for k, pos in seq_to_pos.items()]
    seq_list.sort(key=lambda x: int(contact[x[1]].sum()), reverse=True)

    sequences = []
    for _, positions in seq_list[:n_seqs]:
        sequences.append({
            "features":  features[positions],
            "sdf_feats": sdf_feats[positions],
            "targets":   targets[positions],
            "wrist_rot": wrist_rot[positions],
            "obj_id":    obj_ids[positions],
            "contact":   contact[positions].astype(float),
            "n_contact": int(contact[positions].sum()),
        })
    return sequences, master_norm


@torch.no_grad()
def run_inference(model, seq, master_norm, embed_matrix, bop_ids, device):
    bop_id_to_idx = {int(b): i for i, b in enumerate(bop_ids)}
    emb = torch.from_numpy(embed_matrix.astype(np.float32)).to(device)

    feat_mean = torch.tensor(master_norm["feature_mean"], dtype=torch.float32).to(device)
    feat_std  = torch.tensor(master_norm["feature_std"],  dtype=torch.float32).to(device)
    sdf_mean  = torch.tensor(master_norm["sdf_mean"],     dtype=torch.float32).to(device)
    sdf_std   = torch.tensor(master_norm["sdf_std"],      dtype=torch.float32).to(device)
    tgt_mean  = torch.tensor(master_norm["target_mean"],  dtype=torch.float32).to(device)
    tgt_std   = torch.tensor(master_norm["target_std"],   dtype=torch.float32).to(device)

    feat_np = seq["features"].astype(np.float32)
    sdf_np  = seq["sdf_feats"].astype(np.float32)

    feat_tensor = torch.from_numpy(feat_np).to(device)
    if feat_tensor.shape[1] < feat_mean.shape[0]:
        pad = feat_mean.shape[0] - feat_tensor.shape[1]
        feat_tensor = torch.nn.functional.pad(feat_tensor, (0, pad))
    elif feat_tensor.shape[1] > feat_mean.shape[0]:
        feat_tensor = feat_tensor[:, :feat_mean.shape[0]]
    feat_t = (feat_tensor - feat_mean) / feat_std.clamp_min(1e-6)
    sdf_t  = (torch.from_numpy(sdf_np).to(device)  - sdf_mean)  / sdf_std.clamp_min(1e-6)
    inp_t  = torch.cat([feat_t, sdf_t], dim=-1)

    fdim = model.feat_proj[0].in_features
    if inp_t.shape[1] > fdim:
        inp_t = inp_t[:, :fdim]
    elif inp_t.shape[1] < fdim:
        inp_t = torch.nn.functional.pad(inp_t, (0, fdim - inp_t.shape[1]))

    h, c = model.initial_state(1, device)
    pred_poses, pred_wrists, pred_contacts = [], [], []
    prev_wrist = None

    for t in range(len(seq["features"])):
        emb_idx = bop_id_to_idx.get(int(seq["obj_id"][t]), 0)
        obj_emb = emb[emb_idx].unsqueeze(0)
        frame = inp_t[t:t+1].clone()
        if prev_wrist is not None and frame.shape[1] >= WRIST_DIMS.stop:
            frame[:, WRIST_DIMS] = (prev_wrist - feat_mean[WRIST_DIMS]) / feat_std[WRIST_DIMS].clamp_min(1e-6)
        pose, wrist, cp, h, c = model(frame, obj_emb, h, c)
        pred_poses.append(pose.squeeze(0).cpu().numpy())
        pred_wrists.append(wrist.squeeze(0).cpu())
        pred_contacts.append(float(cp.item()))
        prev_wrist = wrist.detach()

    pred_poses_raw = np.stack(pred_poses) * tgt_std.cpu().numpy() + tgt_mean.cpu().numpy()
    pred_wrists_np = torch.stack(pred_wrists).numpy()
    return pred_poses_raw, pred_wrists_np, np.array(pred_contacts)


def make_video(pred_joints, gt_joints, pred_contacts, gt_contacts,
               out_path, title, fps=15, max_frames=200):
    T = min(len(pred_joints), max_frames)
    pj = pred_joints[:T]
    gj = gt_joints[:T]
    pc = pred_contacts[:T]
    gc = gt_contacts[:T]

    all_pts = np.concatenate([pj, gj], axis=0).reshape(-1, 3)
    center  = all_pts.mean(0)
    span    = max(np.ptp(all_pts, axis=0).max() * 0.6, 150.0)
    lo, hi  = center - span, center + span

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    def draw_skel(joints, color, alpha=0.9):
        lines = []
        for i, j in HAND_EDGES:
            ln, = ax.plot([joints[i,0], joints[j,0]], [joints[i,1], joints[j,1]],
                          [joints[i,2], joints[j,2]], color=color, lw=2, alpha=alpha)
            lines.append(ln)
        sc = ax.scatter(joints[:,0], joints[:,1], joints[:,2],
                        c=color, s=18, alpha=alpha, depthshade=False)
        return lines, sc

    r = max(span * 0.12, 40.0)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    ax.plot_surface(r*np.cos(u)*np.sin(v), r*np.sin(u)*np.sin(v), r*np.cos(v),
                    color="silver", alpha=0.25, linewidth=0)

    pl, ps = draw_skel(pj[0], "#2980b9")
    gl, gs = draw_skel(gj[0], "#27ae60")
    ttl = ax.set_title("", fontsize=10)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="#2980b9", lw=2, label="Predicted"),
        Line2D([0],[0], color="#27ae60", lw=2, label="Ground Truth"),
    ], loc="upper left")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=20, azim=45)

    def update(f):
        for k, (i, j) in enumerate(HAND_EDGES):
            pl[k].set_data([pj[f,i,0], pj[f,j,0]], [pj[f,i,1], pj[f,j,1]])
            pl[k].set_3d_properties([pj[f,i,2], pj[f,j,2]])
            gl[k].set_data([gj[f,i,0], gj[f,j,0]], [gj[f,i,1], gj[f,j,1]])
            gl[k].set_3d_properties([gj[f,i,2], gj[f,j,2]])
        ps._offsets3d = (pj[f,:,0], pj[f,:,1], pj[f,:,2])
        gs._offsets3d = (gj[f,:,0], gj[f,:,1], gj[f,:,2])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        pc_str = "YES" if pc[f] >= 0.5 else "no"
        gc_str = "YES" if int(gc[f]) == 1 else "no"
        ttl.set_text(f"{title}  frame {f+1}/{T}  pred_contact={pc_str}  gt_contact={gc_str}")
        return pl + gl + [ps, gs, ttl]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    print(f"  Saved → {out_path}  ({T} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand",    default="right", choices=["left", "right"])
    p.add_argument("--ckpt",    default=None, type=Path)
    p.add_argument("--out_dir", default=None, type=Path)
    p.add_argument("--n_seqs",  default=2, type=int, help="Videos per source")
    p.add_argument("--fps",     default=15, type=int)
    p.add_argument("--max_frames", default=200, type=int)
    args = p.parse_args()

    root = Path(__file__).parent.parent
    if args.ckpt is None:
        args.ckpt = root / f"checkpoints/lstm_right_v2/best.pt"
    if args.out_dir is None:
        args.out_dir = root / "results/source_comparison"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd   = ckpt["model"]
    model = SDFLSTMModel(
        feat_dim=infer_feat_dim(sd),
        orientation_aware_sdf=infer_orientation_aware(sd),
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}")

    embed_matrix = np.load(root / "data/models/sdf_grids/sdf_embed_matrix.npy")
    bop_ids      = np.load(root / "data/models/sdf_grids/sdf_bop_ids.npy")

    # Always use HOT3D norm stats for all sources
    hot3d_h5 = root / f"data/processed/hot3d_mano/{args.hand}/dataset_mano.h5"
    with h5py.File(hot3d_h5) as f:
        master_norm = json.loads(f.attrs["meta"])

    mano_fk = MANOForwardKinematics(hand=args.hand)
    _ = mano_fk.model

    sources = [
        ("HOT3D",  root / f"data/processed/hot3d_mano/{args.hand}/dataset_mano.h5",  "hot3d"),
        ("ARCTIC", root / f"data/processed/arctic_mano/{args.hand}/dataset_mano.h5", "arctic"),
        ("DexYCB", root / f"data/processed/dexycb_mano/{args.hand}/dataset_mano.h5", "dexycb"),
    ]

    for source_name, h5_path, tag in sources:
        if not h5_path.exists():
            print(f"[SKIP] {source_name}: {h5_path} not found")
            continue

        split = "val"
        print(f"\n── {source_name} [{split}] ──")
        seqs, norm = load_sequences(h5_path, split, master_norm, n_seqs=args.n_seqs)

        for si, seq in enumerate(seqs):
            print(f"  seq {si}: {len(seq['features'])} frames, contact={seq['n_contact']}")
            pred_poses, pred_wrists, pred_contacts = run_inference(
                model, seq, master_norm, embed_matrix, bop_ids, device)

            pred_joints = mano_fk(pred_poses, pred_wrists) * 1000.0
            gt_joints   = mano_fk(seq["targets"], seq["wrist_rot"]) * 1000.0

            out_path = args.out_dir / f"{tag}_seq{si:02d}.mp4"
            make_video(
                pred_joints, gt_joints,
                pred_contacts, seq["contact"],
                out_path,
                title=f"{source_name} seq{si}",
                fps=args.fps, max_frames=args.max_frames,
            )

    print(f"\nDone. Videos in {args.out_dir}/")


if __name__ == "__main__":
    main()
