"""Visualize AuraXR SDF-LSTM predictions in 3D and export MP4 videos.

Usage:
    # Single sequence video
    python src/visualize_lstm.py --hand right --seq_idx 0

    # Filter to one object inside the best sequence for that object
    python src/visualize_lstm.py --hand right --obj_id 31

    # Batch: one video per object (auto-picks best sequence for each)
    python src/visualize_lstm.py --hand right --batch

    # Batch with specific objects
    python src/visualize_lstm.py --hand right --batch --obj_ids 7 8 9 22 27 28 31
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from mano_fk import MANOForwardKinematics
from model import SDFLSTMModel

# ── MANO hand skeleton connectivity (16 joints, smplx) ───────────────────────
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15),
]

WRIST_DIMS = slice(11, 17)

# BOP id → object name (HOT3D objects)
BOP_NAMES = {
    1: "holder_black", 2: "bowl", 3: "plate_bamboo", 4: "spoon_wooden",
    5: "potato_masher", 6: "spatula_red", 7: "coffee_pot", 8: "mug_patterned",
    9: "mug_white", 10: "can_soup", 11: "can_parmesan", 12: "can_tomato_sauce",
    13: "bottle_mustard", 14: "bottle_bbq", 15: "bottle_ranch", 16: "vase",
    17: "carton_milk", 18: "carton_oj", 19: "flask", 20: "food_waffles",
    21: "food_vegetables", 22: "dumbbell_5lb", 23: "aria_small", 24: "cellphone",
    25: "holder_gray", 26: "birdhouse_toy", 27: "dino_toy", 28: "keyboard",
    29: "whiteboard_eraser", 30: "puzzle_toy", 31: "mouse",
    32: "whiteboard_marker", 33: "dvd_remote",
}


# ─────────────────────────────────────────────────────────────────────────────

def infer_feat_dim(state_dict):
    w = state_dict.get("feat_proj.0.weight")
    return int(w.shape[1]) if w is not None else 29


def infer_orientation_aware(state_dict):
    w = state_dict.get("obj_inj.0.weight")
    return bool(w is not None and int(w.shape[1]) == 99)


def build_seq_index(h5_path, split):
    """Return ordered list of (seq_key, [sorted positions])."""
    with h5py.File(h5_path, "r") as f:
        g = f[split]
        seq_ids   = g["sequence_id"][:]
        is_mirror = g["is_mirror"][:] if "is_mirror" in g else np.zeros_like(seq_ids)
        frame_idx = g["frame_index"][:]

    seq_to_pos: dict = {}
    for i in range(len(seq_ids)):
        k = (int(seq_ids[i]), int(is_mirror[i]))
        seq_to_pos.setdefault(k, []).append(i)
    for k, pos in seq_to_pos.items():
        pos.sort(key=lambda i: frame_idx[i])
    return list(seq_to_pos.items())


def best_seq_for_obj(h5_path, split, target_obj_id):
    """Return (seq_idx, local_indices_for_obj) for the seq with most contact frames.

    Falls back to most total frames if no sequence has contact frames.
    Prefers contact-rich sequences so the video shows actual grasping moments.
    """
    index = build_seq_index(h5_path, split)
    with h5py.File(h5_path, "r") as f:
        obj_ids = f[split]["obj_id"][:]
        contact_name = "contact_v2" if "contact_v2" in f[split] else "contact"
        contact = f[split][contact_name][:]

    best_idx, best_pos, best_contact_count = -1, [], 0
    fallback_idx, fallback_pos = -1, []
    for seq_i, (_, positions) in enumerate(index):
        pos_arr = np.array(positions)
        obj_pos = pos_arr[obj_ids[pos_arr] == target_obj_id].tolist()
        if not obj_pos:
            continue
        contact_count = int(contact[obj_pos].sum())
        if contact_count > best_contact_count:
            best_contact_count = contact_count
            best_pos = obj_pos
            best_idx = seq_i
        if len(obj_pos) > len(fallback_pos):
            fallback_pos = obj_pos
            fallback_idx = seq_i

    if best_idx >= 0:
        return best_idx, best_pos
    return fallback_idx, fallback_pos


def load_sequence(h5_path, split, seq_idx):
    index = build_seq_index(h5_path, split)
    _, positions = index[seq_idx]
    with h5py.File(h5_path, "r") as f:
        g = f[split]
        data = {
            "features":  g["features"][positions],
            "sdf_feats": g["sdf_features"][positions],
            "targets":   g["targets"][positions],
            "wrist_rot": g["wrist_rot_6d"][positions],
            "obj_id":    g["obj_id"][positions],
            "contact":   (g["contact_v2"] if "contact_v2" in g else g["contact"])[positions],
            "distances": g["distances"][positions],
        }
        meta = json.loads(f.attrs["meta"])
    return data, meta


@torch.no_grad()
def run_inference(model, data, meta, embed_matrix, bop_ids, device):
    bop_id_to_idx = {int(bid): i for i, bid in enumerate(bop_ids)}
    emb       = torch.from_numpy(embed_matrix.astype(np.float32)).to(device)
    feat_mean = torch.tensor(meta["feature_mean"], dtype=torch.float32).to(device)
    feat_std  = torch.tensor(meta["feature_std"],  dtype=torch.float32).to(device)
    sdf_mean  = torch.tensor(meta["sdf_mean"],     dtype=torch.float32).to(device)
    sdf_std   = torch.tensor(meta["sdf_std"],      dtype=torch.float32).to(device)
    tgt_mean  = torch.tensor(meta["target_mean"],  dtype=torch.float32).to(device)
    tgt_std   = torch.tensor(meta["target_std"],   dtype=torch.float32).to(device)

    feat_t = (torch.from_numpy(data["features"].astype(np.float32)).to(device) - feat_mean) / feat_std
    sdf_t  = (torch.from_numpy(data["sdf_feats"].astype(np.float32)).to(device) - sdf_mean) / sdf_std
    inp_t  = torch.cat([feat_t, sdf_t], dim=-1)
    fdim   = model.feat_proj[0].in_features
    if inp_t.shape[1] > fdim:
        inp_t = inp_t[:, :fdim]
    elif inp_t.shape[1] < fdim:
        inp_t = torch.nn.functional.pad(inp_t, (0, fdim - inp_t.shape[1]))

    h, c = model.initial_state(batch_size=1, device=device)
    pred_poses, pred_wrists, pred_contacts = [], [], []
    prev_wrist = None

    for t in range(len(data["features"])):
        emb_idx = bop_id_to_idx.get(int(data["obj_id"][t]), 0)
        obj_emb = emb[emb_idx].unsqueeze(0)
        frame   = inp_t[t:t+1].clone()
        if prev_wrist is not None and frame.shape[1] >= WRIST_DIMS.stop:
            frame[:, WRIST_DIMS] = (prev_wrist - feat_mean[WRIST_DIMS]) / feat_std[WRIST_DIMS].clamp_min(1e-6)
        pose, wrist, contact_prob, h, c = model(frame, obj_emb, h, c)
        pred_poses.append(pose.squeeze(0).cpu().numpy())
        pred_wrists.append(wrist.squeeze(0).cpu())
        pred_contacts.append(float(contact_prob.item()))
        prev_wrist = wrist.detach()

    pred_poses_raw = np.stack(pred_poses) * tgt_std.cpu().numpy() + tgt_mean.cpu().numpy()
    pred_wrists_np = torch.stack(pred_wrists).numpy()
    return pred_poses_raw, pred_wrists_np, np.array(pred_contacts)


def make_video(
    pred_joints,    # (T, J, 3) mm
    gt_joints,      # (T, J, 3) mm
    pred_contacts,  # (T,)
    gt_contacts,    # (T,)
    wrist_obj_pos,  # (T, 3) metres
    out_path: Path,
    title_prefix: str = "",
    fps: int = 15,
    max_frames: int = 200,
):
    T = min(len(pred_joints), max_frames)
    pred_joints = pred_joints[:T]
    gt_joints   = gt_joints[:T]
    wrist_pos   = wrist_obj_pos[:T]

    pred_world = pred_joints + wrist_pos[:, None, :] * 1000.0
    gt_world   = gt_joints   + wrist_pos[:, None, :] * 1000.0

    all_pts = gt_world.reshape(-1, 3)
    center  = all_pts.mean(0)
    span    = max(np.ptp(all_pts, axis=0).max() * 0.6, 150.0)
    lo, hi  = center - span, center + span

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    def draw_skeleton(joints, color, alpha=0.9, lw=2):
        lines = []
        for i, j in HAND_EDGES:
            line, = ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                [joints[i, 2], joints[j, 2]],
                color=color, lw=lw, alpha=alpha,
            )
            lines.append(line)
        sc = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                        c=color, s=18, alpha=alpha, depthshade=False)
        return lines, sc

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    r = max(span * 0.12, 40.0)   # 12% of scene span so sphere is always visible
    ax.plot_surface(r*np.cos(u)*np.sin(v), r*np.sin(u)*np.sin(v), r*np.cos(v),
                    color="silver", alpha=0.25, linewidth=0)

    pred_lines, pred_sc = draw_skeleton(pred_world[0], "#2980b9")
    gt_lines,   gt_sc   = draw_skeleton(gt_world[0],   "#27ae60")
    title_obj = ax.set_title("", fontsize=11)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#2980b9", lw=2, label="Predicted"),
        Line2D([0], [0], color="#27ae60", lw=2, label="Ground Truth"),
    ], loc="upper left")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")

    def update(frame):
        pj = pred_world[frame]
        gj = gt_world[frame]
        for k, (i, j) in enumerate(HAND_EDGES):
            pred_lines[k].set_data([pj[i,0], pj[j,0]], [pj[i,1], pj[j,1]])
            pred_lines[k].set_3d_properties([pj[i,2], pj[j,2]])
            gt_lines[k].set_data([gj[i,0], gj[j,0]], [gj[i,1], gj[j,1]])
            gt_lines[k].set_3d_properties([gj[i,2], gj[j,2]])
        pred_sc._offsets3d = (pj[:,0], pj[:,1], pj[:,2])
        gt_sc._offsets3d   = (gj[:,0], gj[:,1], gj[:,2])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        pc = "YES" if pred_contacts[frame] >= 0.5 else "no"
        gc = "YES" if int(gt_contacts[frame]) == 1 else "no"
        title_obj.set_text(f"{title_prefix}  frame {frame+1}/{T}  pred_contact={pc}  gt_contact={gc}")
        return pred_lines + gt_lines + [pred_sc, gt_sc, title_obj]

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=20, azim=45)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    print(f"  Saved → {out_path}  ({T} frames @ {fps} fps)")


def run_for_object(obj_id, h5_path, split, model, embed_matrix, bop_ids,
                   mano_fk, device, out_dir, fps=15, max_frames=200):
    obj_name = BOP_NAMES.get(obj_id, f"obj{obj_id}")
    print(f"\n[obj {obj_id:2d}] {obj_name}")

    seq_idx, obj_global_positions = best_seq_for_obj(h5_path, split, obj_id)
    if seq_idx < 0 or len(obj_global_positions) < 10:
        print(f"  skipped (only {len(obj_global_positions)} frames)")
        return

    data, meta = load_sequence(h5_path, split, seq_idx)
    print(f"  seq_idx={seq_idx}  total={len(data['features'])}  obj_frames={len(obj_global_positions)}")

    # Run inference on full sequence (LSTM needs sequential context)
    pred_poses, pred_wrists, pred_contacts = run_inference(
        model, data, meta, embed_matrix, bop_ids, device
    )

    # Extract frames for this object (local indices within the loaded sequence)
    local_mask = (data["obj_id"] == obj_id)
    local_idx  = np.where(local_mask)[0]

    pred_poses_obj    = pred_poses[local_idx]
    pred_wrists_obj   = pred_wrists[local_idx]
    pred_contacts_obj = pred_contacts[local_idx]
    gt_targets_obj    = data["targets"][local_idx]
    gt_contacts_obj   = data["contact"][local_idx].astype(float)
    wrist_obj_pos     = data["features"][local_idx, 25:28]

    gt_wrist_6d = data["wrist_rot"][local_idx]
    pred_joints = mano_fk(pred_poses_obj, pred_wrists_obj) * 1000.0
    gt_joints   = mano_fk(gt_targets_obj, gt_wrist_6d)    * 1000.0

    out_path = out_dir / f"{obj_id:02d}_{obj_name}.mp4"
    make_video(
        pred_joints, gt_joints, pred_contacts_obj, gt_contacts_obj,
        wrist_obj_pos, out_path,
        title_prefix=f"obj {obj_id}: {obj_name}",
        fps=fps, max_frames=max_frames,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default=None,  type=Path)
    parser.add_argument("--data",       default=None,  type=Path)
    parser.add_argument("--hand",       default="right", choices=["left", "right"])
    parser.add_argument("--split",      default="val")
    parser.add_argument("--seq_idx",    default=None,  type=int, help="Single sequence index")
    parser.add_argument("--obj_id",     default=None,  type=int, help="Single BOP object ID")
    parser.add_argument("--batch",      action="store_true", help="One video per object")
    parser.add_argument("--obj_ids",    default=None,  nargs="+", type=int,
                        help="Object IDs for batch (default: all in val)")
    parser.add_argument("--out",        default=None,  type=Path, help="Output path (single mode)")
    parser.add_argument("--out_dir",    default=None,  type=Path, help="Output dir (batch/obj mode)")
    parser.add_argument("--fps",        default=15,    type=int)
    parser.add_argument("--max_frames", default=200,   type=int)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    if args.ckpt is None:
        args.ckpt = root / f"checkpoints/lstm_{args.hand}/best.pt"
    if args.data is None:
        args.data = root / f"data/processed/hot3d_mano/{args.hand}/dataset_mano.h5"
    if args.out_dir is None:
        args.out_dir = root / f"results/per_object_{args.hand}"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt  = torch.load(args.ckpt, map_location="cpu")
    model = SDFLSTMModel(
        feat_dim=infer_feat_dim(ckpt["model"]),
        orientation_aware_sdf=infer_orientation_aware(ckpt["model"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}")

    embed_matrix = np.load(root / "data/models/sdf_grids/sdf_embed_matrix.npy")
    bop_ids      = np.load(root / "data/models/sdf_grids/sdf_bop_ids.npy")

    mano_fk = MANOForwardKinematics(hand=args.hand)
    _ = mano_fk.model
    print(f"MANO FK loaded ({args.hand})")

    # ── Single sequence mode ──────────────────────────────────────────────────
    if args.seq_idx is not None and not args.batch and args.obj_id is None:
        data, meta = load_sequence(args.data, args.split, args.seq_idx)
        print(f"Sequence {args.seq_idx}: {len(data['features'])} frames")
        pred_poses, pred_wrists, pred_contacts = run_inference(
            model, data, meta, embed_matrix, bop_ids, device)
        T = len(pred_poses)
        pred_joints = mano_fk(pred_poses, pred_wrists)              * 1000.0
        gt_joints   = mano_fk(data["targets"], data["wrist_rot"])   * 1000.0
        out = args.out or (args.out_dir / f"seq{args.seq_idx}.mp4")
        make_video(pred_joints, gt_joints, pred_contacts,
                   data["contact"].astype(float), data["features"][:, 25:28],
                   out, fps=args.fps, max_frames=args.max_frames)
        return

    # ── Single object mode ────────────────────────────────────────────────────
    if args.obj_id is not None and not args.batch:
        run_for_object(args.obj_id, args.data, args.split, model, embed_matrix,
                       bop_ids, mano_fk, device, args.out_dir,
                       fps=args.fps, max_frames=args.max_frames)
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        if args.obj_ids:
            target_ids = args.obj_ids
        else:
            with h5py.File(args.data, "r") as f:
                target_ids = sorted(int(x) for x in np.unique(f[args.split]["obj_id"][:]))
        print(f"Batch: {len(target_ids)} objects → {args.out_dir}")
        for oid in target_ids:
            run_for_object(oid, args.data, args.split, model, embed_matrix,
                           bop_ids, mano_fk, device, args.out_dir,
                           fps=args.fps, max_frames=args.max_frames)
        print(f"\nAll done. Videos in {args.out_dir}/")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
