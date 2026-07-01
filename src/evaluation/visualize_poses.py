"""C1: Static pose visualisation — predicted vs ground truth.

Renders MANO finger joint positions (via FK) alongside the object point cloud.
Saves PNG files for: random samples, median-error sample, best, worst, and failure cases.

Usage:
  python3 src/evaluation/visualize_poses.py --checkpoint checkpoints/full_phase2_best.pt
  python3 src/evaluation/visualize_poses.py --checkpoint checkpoints/full_phase2_best.pt --split unseen_test --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import GraspModel  # noqa: E402
from model.mano_fk import fingertip_positions  # noqa: E402
from model.model_io import EVAL_CONTACT_THRESHOLD_M  # noqa: E402
from utils.paths import RESULTS_DIR  # noqa: E402


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        return plt
    except ImportError:
        print("matplotlib not available — install with: pip install matplotlib")
        sys.exit(1)


def render_sample(
    plt,
    pred_pose: np.ndarray,   # (45,) axis-angle
    gt_pose: np.ndarray,     # (45,)
    obj_pts: np.ndarray,     # (N,3)
    title: str,
    out_path: Path,
) -> None:
    pred_t = torch.from_numpy(pred_pose).unsqueeze(0).float()
    gt_t   = torch.from_numpy(gt_pose).unsqueeze(0).float()

    pred_tips = fingertip_positions(pred_t).squeeze(0).numpy()  # (5,3)
    gt_tips   = fingertip_positions(gt_t).squeeze(0).numpy()    # (5,3)

    # Compute contact distances for colour coding
    pred_dists = np.linalg.norm(
        pred_tips[:, None, :] - obj_pts[None, :, :], axis=-1
    ).min(axis=1)  # (5,)
    in_contact = pred_dists < EVAL_CONTACT_THRESHOLD_M
    penetrating = pred_dists < 0.003

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    for ax, tips, label, color in [
        (ax1, pred_tips, "Predicted", "red"),
        (ax2, gt_tips,   "Ground Truth", "green"),
    ]:
        sub_pts = obj_pts[np.random.choice(len(obj_pts), min(512, len(obj_pts)), replace=False)]
        ax.scatter(sub_pts[:, 0], sub_pts[:, 1], sub_pts[:, 2],
                   c="lightblue", alpha=0.3, s=2, label="Object")
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2],
                   c=color, s=80, marker="o", zorder=5, label=label)
        if label == "Predicted":
            for i, (tip, ic, pen) in enumerate(zip(tips, in_contact, penetrating)):
                c = "orange" if pen else ("lime" if ic else color)
                ax.scatter(*tip, c=c, s=120, marker="*", zorder=6)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="seen_test",
                        choices=["seen_test", "unseen_test", "val", "train"])
    parser.add_argument("--source", type=str, default="oakink", choices=["oakink", "hot3d"])
    parser.add_argument("--n", type=int, default=20, help="Number of samples to collect for statistics")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir")
    args = parser.parse_args()

    plt = _try_import_matplotlib()

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

    # Load dataset
    if args.source == "oakink":
        from data.dataset_oakink import OakInkStaticDataset
        from data.collate import collate_oakink as collate_fn
        ds = OakInkStaticDataset(split=args.split, augment=False)
    else:
        from data.dataset_hot3d import Hot3DTemporalDataset
        from data.collate import collate_hot3d as collate_fn
        ds = Hot3DTemporalDataset(split="test")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all samples with geodesic error
    from evaluation.eval_metrics import geodesic_rotation_error
    samples = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n:
                break
            frame_feat = batch["frame_feat"].to(device)
            obj_pts_t  = batch["obj_pts"].to(device)
            target     = batch["target_pose"].to(device)
            prev_pose  = batch.get("prev_pose")
            if prev_pose is not None:
                prev_pose = prev_pose.to(device)
            out = model.infer(frame_feat, obj_pts_t, prev_pose=prev_pose, k=1)
            pred = out["selected_pose"]
            geo  = float(geodesic_rotation_error(pred, target))
            samples.append({
                "idx": i,
                "geo_err": geo,
                "pred": pred.squeeze(0).cpu().numpy(),
                "gt":   target.squeeze(0).cpu().numpy(),
                "obj_pts": obj_pts_t.squeeze(0).cpu().numpy(),
                "obj_name": batch.get("obj_name", ["unknown"])[0],
            })

    if not samples:
        print("No samples collected.")
        return

    errs = np.array([s["geo_err"] for s in samples])
    median_idx  = int(np.argsort(np.abs(errs - np.median(errs)))[0])
    best_idx    = int(np.argmin(errs))
    worst_idx   = int(np.argmax(errs))
    # Random 3 (excluding best/worst/median)
    exclude = {median_idx, best_idx, worst_idx}
    rand_pool = [i for i in range(len(samples)) if i not in exclude]
    rng = np.random.default_rng(42)
    rand_idxs = rng.choice(rand_pool, size=min(3, len(rand_pool)), replace=False).tolist()

    to_render = [
        (best_idx,   "best"),
        (median_idx, "median"),
        (worst_idx,  "worst"),
    ] + [(i, f"random_{j}") for j, i in enumerate(rand_idxs)]

    for sidx, label in to_render:
        s = samples[sidx]
        title = f"{label}  geo={s['geo_err']:.2f}°  obj={s['obj_name']}"
        out_path = out_dir / f"{args.split}_{label}.png"
        render_sample(plt, s["pred"], s["gt"], s["obj_pts"], title, out_path)
        print(f"Saved {out_path}  geo={s['geo_err']:.2f}°")

    print(f"\nAll visualizations saved to {out_dir}")
    meta = {"split": args.split, "n_samples": len(samples),
            "median_geo": float(np.median(errs)), "mean_geo": float(errs.mean())}
    (out_dir / f"viz_meta_{args.split}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
