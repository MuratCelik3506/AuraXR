"""C1: Temporal animation — HOT3D sequence predictions as GIF or MP4.

Renders frame-by-frame model output alongside ground truth for 3-5 HOT3D sequences.
Includes at least one failure case (highest geodesic error sequence).

Usage:
  python3 src/evaluation/make_temporal_gif.py --checkpoint checkpoints/full_phase2_best.pt
  python3 src/evaluation/make_temporal_gif.py --checkpoint checkpoints/full_phase2_best.pt --format mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import GraspModel  # noqa: E402
from model.mano_fk import fingertip_positions  # noqa: E402
from utils.paths import RESULTS_DIR  # noqa: E402


def _try_imports(fmt: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not available — pip install matplotlib")
        sys.exit(1)
    if fmt == "gif":
        try:
            import imageio
        except ImportError:
            print("imageio not available — pip install imageio")
            sys.exit(1)
        return plt, imageio, None
    else:
        try:
            import cv2
        except ImportError:
            print("opencv not available — pip install opencv-python")
            sys.exit(1)
        return plt, None, cv2


def render_frame_img(plt, pred_tips: np.ndarray, gt_tips: np.ndarray, obj_pts: np.ndarray,
                     frame_idx: int, geo_err: float) -> np.ndarray:
    fig = plt.figure(figsize=(10, 5))
    sub = obj_pts[np.random.choice(len(obj_pts), min(256, len(obj_pts)), replace=False)]
    for col, (tips, label, color) in enumerate([
        (pred_tips, "Pred", "red"), (gt_tips, "GT", "green")
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c="lightblue", alpha=0.2, s=2)
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], c=color, s=60, marker="o")
        ax.set_title(f"{label}  frame={frame_idx}  geo={geo_err:.1f}°", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--n-seqs", type=int, default=5, dest="n_seqs",
                        help="Number of sequences to render (includes 1 failure case)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    plt, imageio, cv2 = _try_imports(args.format)

    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_args = ckpt.get("args", {})
    model = GraspModel(
        hidden=saved_args.get("hidden", 256),
        z_dim=saved_args.get("z_dim", 64),
        encoder_type=saved_args.get("encoder_type", "gru"),
        obj_encoder_type=saved_args.get("obj_encoder_type", "pointnet"),
        use_attention=saved_args.get("use_attention", True),
        use_film=saved_args.get("use_film", True),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    from data.dataset_hot3d import Hot3DTemporalDataset
    from data.collate import collate_hot3d
    from evaluation.eval_metrics import geodesic_rotation_error
    from torch.utils.data import DataLoader

    ds = Hot3DTemporalDataset(split="test", stride=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_hot3d)

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-sequence frames
    seq_frames: dict[str, list[dict]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            seq_path = (batch.get("seq_path") or ["unknown"])[0]
            frame_feat = batch["frame_feat"].to(device)
            obj_pts_t  = batch["obj_pts"].to(device)
            target     = batch["target_pose"].to(device)
            out = model.infer(frame_feat, obj_pts_t, k=1)
            pred = out["selected_pose"]
            geo  = float(geodesic_rotation_error(pred, target))
            pred_tips = fingertip_positions(pred).squeeze(0).cpu().numpy()
            gt_tips   = fingertip_positions(target).squeeze(0).cpu().numpy()
            obj_pts_np = obj_pts_t.squeeze(0).cpu().numpy()
            target_idx = int(batch.get("target_index", [len(seq_frames[seq_path])])[0])
            seq_frames[seq_path].append({
                "target_idx": target_idx, "geo": geo,
                "pred_tips": pred_tips, "gt_tips": gt_tips, "obj_pts": obj_pts_np,
            })

    if not seq_frames:
        print("No HOT3D test data found.")
        return

    # Select sequences: sort by mean geodesic error, pick best + worst + random
    seq_mean_geo = {s: np.mean([f["geo"] for f in frames]) for s, frames in seq_frames.items()}
    sorted_seqs = sorted(seq_mean_geo, key=seq_mean_geo.get)
    n = args.n_seqs
    selected = []
    if len(sorted_seqs) >= n:
        selected.append(sorted_seqs[0])    # best
        selected.append(sorted_seqs[-1])   # worst (failure case)
        mid = sorted_seqs[len(sorted_seqs)//2]
        if mid not in selected:
            selected.append(mid)
        rng = np.random.default_rng(42)
        rest = [s for s in sorted_seqs if s not in selected]
        extra = rng.choice(rest, size=max(0, n - len(selected)), replace=False).tolist()
        selected += extra
    else:
        selected = sorted_seqs

    print(f"Rendering {len(selected)} sequences to {out_dir}")
    meta = {}
    for seq_path in selected:
        frames = sorted(seq_frames[seq_path], key=lambda f: f["target_idx"])
        label = "failure" if seq_path == sorted_seqs[-1] else "seq"
        safe_name = Path(seq_path).stem if seq_path != "unknown" else f"seq_{hash(seq_path)%10000}"
        mean_geo = seq_mean_geo[seq_path]
        imgs = []
        for f in frames:
            img = render_frame_img(plt, f["pred_tips"], f["gt_tips"], f["obj_pts"],
                                   f["target_idx"], f["geo"])
            imgs.append(img)
        if args.format == "gif":
            out_path = out_dir / f"{label}_{safe_name}.gif"
            imageio.mimsave(str(out_path), imgs, fps=args.fps, loop=0)
        else:
            out_path = out_dir / f"{label}_{safe_name}.mp4"
            h, w = imgs[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
            for img in imgs:
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            writer.release()
        print(f"  {out_path}  n_frames={len(imgs)}  mean_geo={mean_geo:.2f}°")
        meta[str(out_path)] = {"n_frames": len(imgs), "mean_geo_deg": mean_geo,
                                "label": label}

    (out_dir / "animation_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nAll animations saved to {out_dir}")


if __name__ == "__main__":
    main()
