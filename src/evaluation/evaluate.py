"""Offline evaluation pipeline (docs/B-grasp-model.md B10, D4, D6).

Runs the full B10 metric suite over a val/test split and writes a JSON results file.

Usage:
  python3 src/evaluation/evaluate.py --checkpoint checkpoints/grasp_best.pt --split val
  python3 src/evaluation/evaluate.py --checkpoint checkpoints/grasp_best.pt --split test --k 5
  python3 src/evaluation/evaluate.py --checkpoint checkpoints/grasp_best.pt --source both --split val

Output: results/eval_<source>_<split>_<timestamp>.json
Each file contains a "meta" block (checkpoint, epoch, git commit, timestamp, etc.)
and a "metrics" block. Never overwrites previous results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.dataset_hot3d import Hot3DTemporalDataset, collate_hot3d  # noqa: E402
from data.dataset_oakink import OakInkStaticDataset, collate_oakink  # noqa: E402
from evaluation.eval_metrics import (  # noqa: E402
    binary_auc,
    contact_ratio,
    diversity_score,
    expected_calibration_error,
    fingertip_position_error,
    geodesic_rotation_error,
    geodesic_velocity,
    jitter_acceleration,
    jitter_score,
    jitter_velocity,
    joint_limit_saturation_rate,
    joint_limit_violation_rate,
    mpjpe,
    penetration_depth_mm,
    spearman_correlation,
)
from model.grasp_model import GraspModel  # noqa: E402
from utils.paths import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _safe_mean(values: list[float]) -> float:
    """Mean with NaN filtering. Returns NaN if all values are NaN or list is empty."""
    arr = np.array([v for v in values if not (isinstance(v, float) and np.isnan(v))],
                   dtype=np.float64)
    return float(np.mean(arr)) if len(arr) > 0 else float("nan")


@torch.no_grad()
def evaluate_loader(
    model: GraspModel,
    loader: DataLoader,
    dev: torch.device,
    source: str,
    k: int,
) -> dict:
    model.eval()

    # Per-batch scalar accumulators
    geo_errs: list[float] = []
    mpjpe_errs: list[float] = []
    tip_errs: list[float] = []
    limit_viols: list[float] = []
    limit_sats: list[float] = []
    contact_ratios: list[float] = []
    pen_depths: list[float] = []
    quality_scores_all: list[float] = []
    quality_labels_all: list[float] = []
    diversity_scores: list[float] = []

    # Per-object accumulators: obj_name -> list of per-sample values
    obj_geo:    dict[str, list[float]] = defaultdict(list)
    obj_mpjpe:  dict[str, list[float]] = defaultdict(list)
    obj_tip:    dict[str, list[float]] = defaultdict(list)
    obj_lim:    dict[str, list[float]] = defaultdict(list)
    obj_pen:    dict[str, list[float]] = defaultdict(list)
    obj_contact: dict[str, list[float]] = defaultdict(list)

    # For HOT3D sequence-level jitter: collect (seq_path, target_index, pred) per frame.
    # After all batches, sort by seq then index and compute temporal metrics per sequence.
    seq_preds: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    has_seq_info = False

    obj_pts_contact_warned = False

    for batch in loader:
        frame_feat = batch["frame_feat"].to(dev)
        obj_pts    = batch["obj_pts"].to(dev)

        # obj_pts_contact is in wrist frame (correct for FK comparison).
        # If absent, contact/penetration metrics will be skipped with a warning.
        obj_pts_contact = batch.get("obj_pts_contact")
        if obj_pts_contact is not None:
            obj_pts_contact = obj_pts_contact.to(dev)
        elif not obj_pts_contact_warned:
            warnings.warn(
                "obj_pts_contact absent — contact/penetration metrics skipped. "
                "Re-run preprocessing to include wrist-frame object points.",
                stacklevel=2,
            )
            obj_pts_contact_warned = True

        target      = batch["target_pose"].to(dev)
        contact_flag = batch.get("contact_flag")
        if contact_flag is not None:
            contact_flag = contact_flag.to(dev)
        quality_label = batch.get("quality_label")

        prev_pose_eval = batch.get("prev_pose")
        if prev_pose_eval is not None:
            prev_pose_eval = prev_pose_eval.to(dev)
        out  = model.infer(frame_feat, obj_pts, prev_pose=prev_pose_eval, contact_flag=contact_flag, k=k)
        pred = out["selected_pose"]
        q_score = out["selected_quality_score"].squeeze(-1)

        geo_errs.append(float(geodesic_rotation_error(pred, target)))
        mpjpe_errs.append(float(mpjpe(pred, target)))
        tip_errs.append(float(fingertip_position_error(pred, target)))
        limit_viols.append(float(joint_limit_violation_rate(pred)))
        limit_sats.append(float(joint_limit_saturation_rate(pred)))

        if obj_pts_contact is not None:
            contact_ratios.append(float(contact_ratio(pred, obj_pts_contact)))
            pen_depths.append(float(penetration_depth_mm(pred, obj_pts_contact)))

        quality_scores_all.extend(q_score.cpu().float().numpy().tolist())
        if quality_label is not None:
            quality_labels_all.extend(quality_label.cpu().float().numpy().reshape(-1).tolist())

        # Per-object accumulation
        obj_names_batch = batch.get("obj_name")
        if obj_names_batch is not None:
            B = pred.shape[0]
            for i in range(B):
                oname = obj_names_batch[i]
                p_i = pred[i : i + 1]
                t_i = target[i : i + 1]
                obj_geo[oname].append(float(geodesic_rotation_error(p_i, t_i)))
                obj_mpjpe[oname].append(float(mpjpe(p_i, t_i)))
                obj_tip[oname].append(float(fingertip_position_error(p_i, t_i)))
                obj_lim[oname].append(float(joint_limit_violation_rate(p_i)))
                if obj_pts_contact is not None:
                    opc_i = obj_pts_contact[i : i + 1]
                    obj_pen[oname].append(float(penetration_depth_mm(p_i, opc_i)))
                    obj_contact[oname].append(float(contact_ratio(p_i, opc_i)))

        if k > 1 and "candidate_poses" in out:
            cand = out["candidate_poses"]
            diversity_scores.append(float(diversity_score(cand)))

        # Collect for sequence-level jitter (HOT3D provides seq_path + target_index).
        seq_paths    = batch.get("seq_path")
        target_idxs  = batch.get("target_index")
        if seq_paths is not None and target_idxs is not None:
            has_seq_info = True
            pred_np = pred.cpu().float().numpy()
            for b, (sp, ti) in enumerate(zip(seq_paths, target_idxs.tolist())):
                seq_preds[sp][int(ti)] = pred_np[b]

    if len(geo_errs) == 0:
        return {
            "failed_run": True,
            "reason": "n_batches=0 — dataset or split returned no samples",
            "n_batches": 0,
        }

    metrics: dict = {
        "failed_run": False,
        "n_batches": len(geo_errs),
        "geodesic_err_deg":          _safe_mean(geo_errs),
        "mpjpe_mm":                  _safe_mean(mpjpe_errs),
        "fingertip_err_mm":          _safe_mean(tip_errs),
        "joint_limit_violation_rate":   _safe_mean(limit_viols),
        "joint_limit_saturation_rate":  _safe_mean(limit_sats),
    }

    if contact_ratios:
        metrics["contact_ratio"]       = _safe_mean(contact_ratios)
        metrics["penetration_depth_mm"] = _safe_mean(pen_depths)
    else:
        metrics["contact_ratio"]       = float("nan")
        metrics["penetration_depth_mm"] = float("nan")

    # Quality calibration metrics
    if quality_labels_all:
        qs = np.array(quality_scores_all, dtype=np.float32)
        ql = np.array(quality_labels_all, dtype=np.float32)
        binary_labels = (ql > 0.5).astype(int)
        metrics["quality_auc"]      = binary_auc(qs, binary_labels)
        metrics["quality_ece"]      = expected_calibration_error(qs, binary_labels)
        metrics["quality_spearman"] = spearman_correlation(qs, ql)

    # Diversity
    if diversity_scores:
        metrics["diversity_score"] = _safe_mean(diversity_scores)

    # Sequence-level jitter metrics (HOT3D only, requires seq_path in batch)
    if has_seq_info and seq_preds:
        jitter_scores_list:   list[float] = []
        jitter_vel_list:      list[float] = []
        jitter_acc_list:      list[float] = []
        geo_vel_list:         list[float] = []

        for sp, frame_dict in seq_preds.items():
            sorted_indices = sorted(frame_dict.keys())
            if len(sorted_indices) < 2:
                continue
            poses = torch.from_numpy(
                np.stack([frame_dict[i] for i in sorted_indices])
            ).unsqueeze(0)  # (1, T, 45)
            jitter_scores_list.append(float(jitter_score(poses)))
            jitter_vel_list.append(float(jitter_velocity(poses)))
            if len(sorted_indices) >= 3:
                jitter_acc_list.append(float(jitter_acceleration(poses)))
            geo_vel_list.append(float(geodesic_velocity(poses)))

        if jitter_scores_list:
            metrics["jitter_score"]          = _safe_mean(jitter_scores_list)
            metrics["jitter_velocity"]       = _safe_mean(jitter_vel_list)
            metrics["geodesic_velocity_deg"] = _safe_mean(geo_vel_list)
        if jitter_acc_list:
            metrics["jitter_acceleration"] = _safe_mean(jitter_acc_list)

    # Mark failed if any primary metric is NaN
    primary = ["geodesic_err_deg", "mpjpe_mm", "fingertip_err_mm"]
    if any(np.isnan(metrics.get(p, float("nan"))) for p in primary):
        metrics["failed_run"] = True
        metrics["reason"] = "primary metric(s) are NaN"

    # Per-object breakdown (sorted by mean geodesic error descending)
    if obj_geo:
        per_obj: dict[str, dict] = {}
        for oname in obj_geo:
            entry: dict = {
                "n_samples": len(obj_geo[oname]),
                "geodesic_err_deg": _safe_mean(obj_geo[oname]),
                "mpjpe_mm": _safe_mean(obj_mpjpe[oname]),
                "fingertip_err_mm": _safe_mean(obj_tip[oname]),
                "joint_limit_violation_rate": _safe_mean(obj_lim[oname]),
            }
            if obj_pen.get(oname):
                entry["penetration_depth_mm"] = _safe_mean(obj_pen[oname])
                entry["contact_ratio"] = _safe_mean(obj_contact[oname])
            per_obj[oname] = entry
        metrics["per_object"] = dict(
            sorted(per_obj.items(), key=lambda x: x[1]["geodesic_err_deg"], reverse=True)
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--source",      choices=["oakink", "hot3d", "both"], default="hot3d")
    parser.add_argument("--split",       choices=["val", "test", "all"], default="val")
    parser.add_argument("--phase",       type=int, default=None,
                        help="Training phase (1/2/3) for metadata — informational only.")
    parser.add_argument("--batch",       type=int, default=64)
    parser.add_argument("--n_points",    type=int, default=1024)
    parser.add_argument("--window",      type=int, default=16)
    parser.add_argument("--workers",     type=int, default=0)
    parser.add_argument("--k",           type=int, default=1)
    parser.add_argument("--out",         type=str, default=None,
                        help="Output path. Default: results/eval_<source>_<split>_<ts>.json")
    args = parser.parse_args()

    dev = get_device()
    model = GraspModel()

    # Load checkpoint and extract epoch if available
    epoch: int | None = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = ckpt["model"] if "model" in ckpt else ckpt
        epoch = int(ckpt["epoch"]) if "epoch" in ckpt else None
        model.load_state_dict(state)
    model.eval().to(dev)

    # Build metadata block
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        "eval_timestamp": ts,
        "git_commit":     git_commit(),
        "checkpoint":     str(args.checkpoint) if args.checkpoint else None,
        "epoch":          epoch,
        "phase":          args.phase,
        "source":         args.source,
        "split":          args.split,
        "k":              args.k,
        "window":         args.window,
        "device":         str(dev),
    }

    results: dict = {"meta": meta}

    if args.source in ("oakink", "both"):
        oak_split = "val" if args.split == "val" else "test"
        ds = OakInkStaticDataset(split=oak_split, n_points=args.n_points, augment=False)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_oakink)
        results["oakink"] = evaluate_loader(model, loader, dev, "oakink", args.k)
        print("OakInk:", json.dumps(results["oakink"], indent=2))

    if args.source in ("hot3d", "both"):
        ds = Hot3DTemporalDataset(split=args.split, window=args.window,
                                  n_points=args.n_points, stride=4)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_hot3d)
        results["hot3d"] = evaluate_loader(model, loader, dev, "hot3d", args.k)
        print("HOT3D:", json.dumps(results["hot3d"], indent=2))

    # Write — timestamped filename so runs never overwrite each other
    if args.out:
        out_path = Path(args.out)
    else:
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"eval_{args.source}_{args.split}_{ts_file}.json"

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Print failed_run summary
    for src in ("oakink", "hot3d"):
        if src in results and results[src].get("failed_run"):
            reason = results[src].get("reason", "unknown")
            print(f"  WARNING: {src} failed_run=True — {reason}")


if __name__ == "__main__":
    main()
