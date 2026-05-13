"""
12_evaluate.py — Compute full evaluation metrics on the test split.

Metrics computed:
  MPJPE        Mean Per-Joint Position Error on wrist (mm) — proxy until FK available
  PA-MPJPE     Procrustes-aligned MPJPE on wrist
  Pose MSE     Mean-squared error on MANO θ (degrees² equivalent)
  Beta MAE     Mean absolute error on β coefficients
  Wrist-T Err  Wrist translation error (mm)
  Delta-T Err  Controller-to-wrist offset error (mm)
  Wrist-Q Err  Geodesic wrist orientation error (degrees)

Ablation study (--ablation flag):
  geometry_only   Zero out the visual embedding channel (cols 32:96)
  controller_only Zero out the object context channels (cols 18:32)
  no_temporal     Replace each window with single last frame (replicated T times)

Usage:
    python 12_evaluate.py
    python 12_evaluate.py --checkpoint data/checkpoints/best.pt
    python 12_evaluate.py --ablation geometry_only
    python 12_evaluate.py --ablation controller_only
    python 12_evaluate.py --all_ablations
    python 12_evaluate.py --model gru    # evaluate GRU baseline
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import importlib.util
import sys
from pathlib import Path as _Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import numbered modules
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_here = _Path(__file__).parent
_im   = _load_module("intentformer_mod2", _here / "10_intentformer.py")
_tr   = _load_module("train_mod",         _here / "11_train.py")

IntentFormer    = _im.IntentFormer
SingleFrameMLP  = _im.SingleFrameMLP
GRUBaseline     = _im.GRUBaseline
TARGET_DIM      = _im.TARGET_DIM
F_IN            = _im.F_IN
T               = _im.T
compute_losses  = _tr.compute_losses

from hot3d_dataset import HOT3DDataset

DATA_FILE = Path("../data/hot3d_training.h5")
CKPT_DIR  = Path("../data/checkpoints")


# ---------------------------------------------------------------------------
# Procrustes alignment (for PA-MPJPE)
# ---------------------------------------------------------------------------

def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Align pred to gt via optimal rotation (no scale).
    pred, gt: [N, 3]
    Returns aligned pred [N, 3].
    """
    pred_c = pred - pred.mean(0, keepdims=True)
    gt_c   = gt   - gt.mean(0, keepdims=True)
    H = pred_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return pred_c @ R.T + gt.mean(0, keepdims=True)


# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

class MetricAccumulator:
    def __init__(self):
        self.values: dict[str, list] = {}

    def add(self, key: str, val: float):
        self.values.setdefault(key, []).append(val)

    def mean(self, key: str) -> float:
        v = self.values.get(key, [0.])
        return float(np.mean(v)) if v else float("nan")

    def summary(self) -> dict[str, float]:
        return {k: self.mean(k) for k in self.values}


def quat_geodesic_deg(pred_q: np.ndarray, gt_q: np.ndarray) -> float:
    """Geodesic quaternion angle in degrees. pred_q, gt_q: [4] (w,x,y,z)."""
    pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-8)
    gt_q   = gt_q   / (np.linalg.norm(gt_q)   + 1e-8)
    dot    = np.clip(abs(np.dot(pred_q, gt_q)), -1 + 1e-6, 1 - 1e-6)
    return float(np.degrees(2 * np.arccos(dot)))


# ---------------------------------------------------------------------------
# Ablation input transforms
# ---------------------------------------------------------------------------

ABLATION_TRANSFORMS = {
    "geometry_only":   lambda x: x * torch.cat([torch.ones(32), torch.zeros(64)]).to(x.device),
    "controller_only": lambda x: x * torch.cat([torch.ones(18), torch.zeros(14), torch.ones(64)]).to(x.device),
    "no_temporal":     lambda x: x[:, -1:, :].expand_as(x),
}


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    ablation:  Optional[str] = None,
) -> dict[str, float]:
    """
    Run full evaluation on a DataLoader.
    Returns dict of metric name → mean value.
    """
    model.eval()
    acc = MetricAccumulator()

    transform = None
    if ablation and ablation in ABLATION_TRANSFORMS:
        transform = ABLATION_TRANSFORMS[ablation]

    for feat, tgt in loader:
        feat, tgt = feat.to(device), tgt.to(device)

        if transform is not None:
            feat = transform(feat)

        pred = model(feat)
        params_p = model.predict_hand_params(pred)
        params_g = model.predict_hand_params(tgt)

        B = feat.size(0)
        for b in range(B):
            for hand in ("h0", "h1"):
                # Pose MSE
                pose_mse = float(nn.functional.mse_loss(
                    params_p[f"mano_pose_{hand}"][b],
                    params_g[f"mano_pose_{hand}"][b]
                ).item())
                acc.add(f"pose_mse_{hand}", pose_mse)

                # Beta MAE
                beta_mae = float((params_p[f"mano_betas_{hand}"][b] -
                                  params_g[f"mano_betas_{hand}"][b]).abs().mean().item())
                acc.add(f"beta_mae_{hand}", beta_mae)

                # Wrist translation error (mm)
                pt = params_p[f"wrist_t_{hand}"][b].cpu().numpy()
                gt_t = params_g[f"wrist_t_{hand}"][b].cpu().numpy()
                wrist_err_mm = float(np.linalg.norm(pt - gt_t) * 1000)
                acc.add(f"wrist_t_mm_{hand}", wrist_err_mm)

                # MPJPE proxy (wrist translation, mm)
                acc.add(f"mpjpe_{hand}", wrist_err_mm)

                # PA-MPJPE (Procrustes align wrist + unit vector as proxy joints)
                # Use wrist + 3 virtual joints along predicted pose directions
                pred_pts = np.stack([pt,
                                     pt + params_p[f"mano_pose_{hand}"][b, 0:3].cpu().numpy() * 0.1,
                                     pt + params_p[f"mano_pose_{hand}"][b, 3:6].cpu().numpy() * 0.1,
                                     pt + params_p[f"mano_pose_{hand}"][b, 6:9].cpu().numpy() * 0.1])
                gt_pts   = np.stack([gt_t,
                                     gt_t + params_g[f"mano_pose_{hand}"][b, 0:3].cpu().numpy() * 0.1,
                                     gt_t + params_g[f"mano_pose_{hand}"][b, 3:6].cpu().numpy() * 0.1,
                                     gt_t + params_g[f"mano_pose_{hand}"][b, 6:9].cpu().numpy() * 0.1])
                aligned  = procrustes_align(pred_pts, gt_pts)
                pa_err   = float(np.linalg.norm(aligned - gt_pts, axis=-1).mean() * 1000)
                acc.add(f"pa_mpjpe_{hand}", pa_err)

                # Delta offset error (mm)
                pd = params_p[f"delta_t_{hand}"][b].cpu().numpy()
                gd = params_g[f"delta_t_{hand}"][b].cpu().numpy()
                acc.add(f"delta_t_mm_{hand}", float(np.linalg.norm(pd - gd) * 1000))

                # Wrist quaternion geodesic (degrees)
                pq = params_p[f"wrist_q_{hand}"][b].cpu().numpy()
                gq = params_g[f"wrist_q_{hand}"][b].cpu().numpy()
                acc.add(f"wrist_q_deg_{hand}", quat_geodesic_deg(pq, gq))

    return acc.summary()


def print_report(metrics: dict, ablation: Optional[str] = None):
    prefix = f"[{ablation}] " if ablation else "[Full model] "
    print(f"\n{prefix}Evaluation Results")
    print("=" * 55)

    for hand in ("h0", "h1"):
        label = "Hand-0" if hand == "h0" else "Hand-1"
        print(f"\n  {label}:")
        print(f"    MPJPE (wrist proxy)  : {metrics.get(f'mpjpe_{hand}', float('nan')):>8.2f} mm")
        print(f"    PA-MPJPE             : {metrics.get(f'pa_mpjpe_{hand}', float('nan')):>8.2f} mm")
        print(f"    Wrist-T error        : {metrics.get(f'wrist_t_mm_{hand}', float('nan')):>8.2f} mm")
        print(f"    Wrist-Q error        : {metrics.get(f'wrist_q_deg_{hand}', float('nan')):>8.2f} deg")
        print(f"    Delta-T error        : {metrics.get(f'delta_t_mm_{hand}', float('nan')):>8.2f} mm")
        print(f"    Pose MSE             : {metrics.get(f'pose_mse_{hand}', float('nan')):>8.4f}")
        print(f"    Beta MAE             : {metrics.get(f'beta_mae_{hand}', float('nan')):>8.4f}")

    # Average across hands
    mpjpe_avg = np.mean([metrics.get(f"mpjpe_{h}", float("nan")) for h in ("h0","h1")])
    pa_avg    = np.mean([metrics.get(f"pa_mpjpe_{h}", float("nan")) for h in ("h0","h1")])
    print(f"\n  MPJPE (both hands avg) : {mpjpe_avg:>8.2f} mm")
    print(f"  PA-MPJPE (both hands)  : {pa_avg:>8.2f} mm")
    print("=" * 55)


def load_model(ckpt_path: Path, model_name: str, device: torch.device) -> nn.Module:
    if model_name == "intentformer":
        model = IntentFormer()
    elif model_name == "gru":
        model = GRUBaseline()
    else:
        model = SingleFrameMLP()

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        mpjpe = ckpt.get("best_mpjpe", float("nan"))
        print(f"Loaded checkpoint: epoch={epoch}, best_mpjpe={mpjpe:.2f} mm")
    else:
        print(f"[WARN] Checkpoint {ckpt_path} not found — using random weights.")

    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  type=str, default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--model",       choices=["intentformer", "gru", "mlp"],
                    default="intentformer")
    ap.add_argument("--batch",       type=int, default=64)
    ap.add_argument("--workers",     type=int, default=4)
    ap.add_argument("--ablation",    type=str, default=None,
                    choices=list(ABLATION_TRANSFORMS.keys()))
    ap.add_argument("--all_ablations", action="store_true",
                    help="Run all ablations sequentially")
    ap.add_argument("--split",       default="test", choices=["test", "val", "train"])
    ap.add_argument("--device",      type=str, default="auto")
    ap.add_argument("--save_json",   type=str, default=None,
                    help="Save results to a JSON file")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps"  if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available()          else "cpu")
    else:
        device = torch.device(args.device)

    if not DATA_FILE.exists():
        print(f"[ERROR] {DATA_FILE} not found.")
        return

    ds     = HOT3DDataset(DATA_FILE, args.split, normalise=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f"Evaluating on {args.split}: {len(ds):,} windows")

    model = load_model(Path(args.checkpoint), args.model, device)

    all_results = {}

    # Full model evaluation
    metrics = evaluate(model, loader, device, ablation=None)
    print_report(metrics)
    all_results["full"] = metrics

    # Ablations
    ablations_to_run = list(ABLATION_TRANSFORMS.keys()) if args.all_ablations else []
    if args.ablation:
        ablations_to_run = [args.ablation]

    for abl in ablations_to_run:
        m = evaluate(model, loader, device, ablation=abl)
        print_report(m, ablation=abl)
        all_results[abl] = m

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
