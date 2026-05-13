"""
11_train.py — Training loop for IntentFormer.

Reads data/hot3d_training.h5 (produced by 09_build_dataset.py).
Supports IntentFormer, SingleFrameMLP, and GRUBaseline.

Loss functions (all weighted and summed):
  joint_mse       λ=1.0   MSE on MANO pose θ (primary accuracy target)
  beta_mse        λ=0.5   MSE on shape β
  wrist_t_mse     λ=1.0   MSE on wrist translation
  wrist_q_loss    λ=1.0   Geodesic quaternion loss on wrist orientation
  delta_t_mse     λ=0.4   MSE on controller offset translation
  delta_q_loss    λ=0.4   Geodesic quaternion loss on controller offset
  velocity_smooth λ=0.3   Penalises jitter between consecutive predictions (batch-level)
  beta_smooth     λ=0.1   Encourages β to be stable (near zero mean across batch)

Training setup:
  Optimizer : AdamW
  LR        : 1e-4 with cosine annealing decay
  Epochs    : 100 (early stopping on val MPJPE)
  Batch     : 32
  Checkpoint: data/checkpoints/best.pt

Usage:
    python 11_train.py
    python 11_train.py --model gru            # use GRU baseline
    python 11_train.py --model mlp            # use single-frame MLP baseline
    python 11_train.py --epochs 200 --batch 64
    python 11_train.py --resume data/checkpoints/best.pt
"""

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import importlib.util
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path as _Path

# Import numbered module (Python cannot import filenames starting with a digit directly)
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_here = _Path(__file__).parent
_im   = _load_module("intentformer_mod", _here / "10_intentformer.py")
IntentFormer    = _im.IntentFormer
SingleFrameMLP  = _im.SingleFrameMLP
GRUBaseline     = _im.GRUBaseline
TARGET_DIM      = _im.TARGET_DIM
F_IN            = _im.F_IN
T               = _im.T

from hot3d_dataset import HOT3DDataset

DATA_FILE  = Path("../data/hot3d_training.h5")
CKPT_DIR   = Path("../data/checkpoints")
LOG_DIR    = Path("../data/logs")




# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def geodesic_quat_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance between two unit quaternions.
    pred_q, gt_q: [B, 4] (w,x,y,z)
    Returns mean scalar loss.
    """
    pred_q = nn.functional.normalize(pred_q, dim=-1)
    dot    = (pred_q * gt_q).sum(dim=-1).abs().clamp(-1 + 1e-6, 1 - 1e-6)
    return (2 * torch.acos(dot)).mean()


def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    model,
    λ_joint:    float = 1.0,
    λ_beta:     float = 0.5,
    λ_wrist_t:  float = 1.0,
    λ_wrist_q:  float = 1.0,
    λ_delta_t:  float = 0.4,
    λ_delta_q:  float = 0.4,
    λ_vel:      float = 0.3,
    λ_beta_sm:  float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute all loss components.
    pred, target: [B, TARGET_DIM]
    Returns (total_loss, loss_dict).
    """
    p = model.predict_hand_params(pred)
    g = model.predict_hand_params(target)

    mse = nn.functional.mse_loss

    # Joint pose (MANO θ)
    l_joint = (mse(p["mano_pose_h0"], g["mano_pose_h0"]) +
               mse(p["mano_pose_h1"], g["mano_pose_h1"])) * 0.5

    # Shape β
    l_beta  = (mse(p["mano_betas_h0"], g["mano_betas_h0"]) +
               mse(p["mano_betas_h1"], g["mano_betas_h1"])) * 0.5

    # Wrist translation
    l_wrist_t = (mse(p["wrist_t_h0"], g["wrist_t_h0"]) +
                 mse(p["wrist_t_h1"], g["wrist_t_h1"])) * 0.5

    # Wrist quaternion (geodesic)
    l_wrist_q = (geodesic_quat_loss(p["wrist_q_h0"], g["wrist_q_h0"]) +
                 geodesic_quat_loss(p["wrist_q_h1"], g["wrist_q_h1"])) * 0.5

    # Controller-to-wrist offset
    l_delta_t = (mse(p["delta_t_h0"], g["delta_t_h0"]) +
                 mse(p["delta_t_h1"], g["delta_t_h1"])) * 0.5

    l_delta_q = (geodesic_quat_loss(p["delta_q_h0"], g["delta_q_h0"]) +
                 geodesic_quat_loss(p["delta_q_h1"], g["delta_q_h1"])) * 0.5

    # Velocity smoothness: penalise large changes in pose across the batch
    # (a proxy for temporal jitter — exact smoothness needs sequence ordering)
    l_vel = (mse(p["mano_pose_h0"][1:], p["mano_pose_h0"][:-1]) +
             mse(p["mano_pose_h1"][1:], p["mano_pose_h1"][:-1])) * 0.5

    # Beta stability: shape should be near zero mean (regularisation)
    l_beta_sm = (p["mano_betas_h0"].mean().pow(2) +
                 p["mano_betas_h1"].mean().pow(2)) * 0.5

    total = (λ_joint   * l_joint   +
             λ_beta    * l_beta    +
             λ_wrist_t * l_wrist_t +
             λ_wrist_q * l_wrist_q +
             λ_delta_t * l_delta_t +
             λ_delta_q * l_delta_q +
             λ_vel     * l_vel     +
             λ_beta_sm * l_beta_sm)

    losses = {
        "joint":   l_joint.item(),
        "beta":    l_beta.item(),
        "wrist_t": l_wrist_t.item(),
        "wrist_q": l_wrist_q.item(),
        "delta_t": l_delta_t.item(),
        "delta_q": l_delta_q.item(),
        "vel":     l_vel.item(),
        "beta_sm": l_beta_sm.item(),
        "total":   total.item(),
    }
    return total, losses


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def mpjpe_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Proxy MPJPE on the wrist translation (mm).
    True MPJPE requires FK from θ/β; this is a fast per-batch proxy.
    Full MPJPE computed in 12_evaluate.py after training.
    """
    # wrist_t_h0 is at indices 25:28 within the first 39 dims
    pred_t  = pred[:, 25:28]
    gt_t    = target[:, 25:28]
    return float((pred_t - gt_t).norm(dim=-1).mean().item() * 1000)   # metres → mm


def train_epoch(model, loader, optimiser, device, use_amp: bool = False):
    model.train()
    total_loss = 0.
    n_batches  = 0
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else contextlib.nullcontext()
    for feat, tgt in loader:
        feat, tgt = feat.to(device), tgt.to(device)
        with amp_ctx:
            pred = model(feat)
            loss, _ = compute_losses(pred, tgt, model)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, use_amp: bool = False):
    model.eval()
    total_loss = 0.
    all_mpjpe  = []
    loss_comps = {k: 0. for k in ("joint","beta","wrist_t","wrist_q",
                                   "delta_t","delta_q","vel","beta_sm")}
    n_batches  = 0
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else contextlib.nullcontext()
    for feat, tgt in loader:
        feat, tgt = feat.to(device), tgt.to(device)
        with amp_ctx:
            pred = model(feat)
            loss, comps = compute_losses(pred, tgt, model)
        total_loss += loss.item()
        for k in loss_comps:
            loss_comps[k] += comps[k]
        all_mpjpe.append(mpjpe_metric(pred, tgt))
        n_batches += 1
    n = max(n_batches, 1)
    return (total_loss / n,
            {k: v / n for k, v in loss_comps.items()},
            float(np.mean(all_mpjpe)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   choices=["intentformer", "gru", "mlp"],
                    default="intentformer")
    ap.add_argument("--epochs",  type=int,   default=100)
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--lr",      type=float, default=1e-4)
    ap.add_argument("--d_model", type=int,   default=256)
    ap.add_argument("--workers", type=int,   default=6)
    ap.add_argument("--resume",  type=str,   default=None)
    ap.add_argument("--no_norm", action="store_true",
                    help="Disable feature/target normalisation")
    ap.add_argument("--device",  type=str,   default="auto")
    ap.add_argument("--patience",type=int,   default=15,
                    help="Early stopping patience (epochs without val improvement)")
    ap.add_argument("--subset",  type=int,   default=None,
                    help="Use only the first N samples from train and val (POC / smoke test)")
    ap.add_argument("--no_aug",  action="store_true",
                    help="Disable data augmentation (use for baselines / ablation)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model for faster execution (adds ~1 min warm-up)")
    ap.add_argument("--amp",     action="store_true",
                    help="Enable float16 mixed precision (faster on MPS/CUDA)")
    return ap.parse_args()


def build_model(name: str, d_model: int) -> nn.Module:
    if name == "intentformer":
        return IntentFormer(d_model=d_model)
    elif name == "gru":
        return GRUBaseline()
    else:
        return SingleFrameMLP()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("mps"  if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available()          else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    if not DATA_FILE.exists():
        print(f"[ERROR] {DATA_FILE} not found. Run 09_build_dataset.py first.")
        return

    # Datasets
    from torch.utils.data import Subset
    use_aug  = (not args.no_aug) and (args.model == "intentformer")
    train_ds = HOT3DDataset(DATA_FILE, "train", normalise=not args.no_norm, augment=use_aug)
    val_ds   = HOT3DDataset(DATA_FILE, "val",   normalise=not args.no_norm, augment=False)

    if args.subset:
        train_ds = Subset(train_ds, range(min(args.subset, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(args.subset // 5, len(val_ds))))
        print(f"[Subset mode] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    else:
        print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    print(f"Augmentation: {'ON  (pos noise + beta perturb + mirror flip)' if use_aug else 'OFF'}")

    # Workers: safe for all devices now that h5py is opened lazily per worker.
    # pin_memory only helps CUDA (PCIe DMA); skip for MPS (unified memory).
    use_pin = device.type == "cuda"

    # persistent_workers keeps worker processes alive between epochs (faster
    # epoch start), but requires workers > 0.
    use_persistent = args.workers > 0

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=use_pin,
                              persistent_workers=use_persistent)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=use_pin,
                              persistent_workers=use_persistent)

    # Model
    model = build_model(args.model, args.d_model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  ({n_params:,} params)")

    if args.compile:
        print("Compiling model (first batch will be slow — this is normal)…")
        try:
            model = torch.compile(model, backend="aot_eager", fullgraph=False)
            print("torch.compile: OK")
        except Exception as e:
            print(f"torch.compile failed ({e}) — continuing without compile")

    use_amp = args.amp and device.type in ("mps", "cuda")
    if use_amp:
        print(f"Mixed precision: float16 ON  (device={device.type})")
    else:
        print(f"Mixed precision: OFF")

    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    start_epoch  = 0
    best_mpjpe   = math.inf
    patience_ctr = 0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{args.model}_training_log.jsonl"

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mpjpe  = ckpt.get("best_mpjpe", math.inf)
        print(f"Resumed from epoch {start_epoch}, best MPJPE={best_mpjpe:.2f} mm")

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'MPJPE(mm)':>12}  {'LR':>10}  {'Time':>7}")
    print("-" * 65)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimiser, device, use_amp)
        val_loss, comps, val_mpjpe = eval_epoch(model, val_loader, device, use_amp)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch+1:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{val_mpjpe:>12.2f}  {lr:>10.2e}  {elapsed:>6.1f}s")

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_mpjpe":  val_mpjpe,
            "lr": lr,
            "augmentation": use_aug,
            **{f"val_{k}": v for k, v in comps.items()},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Checkpoint
        is_best = val_mpjpe < best_mpjpe
        if is_best:
            best_mpjpe   = val_mpjpe
            patience_ctr = 0
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimiser":  optimiser.state_dict(),
                "best_mpjpe": best_mpjpe,
                "args":       vars(args),
            }, CKPT_DIR / "best.pt")
            print(f"  ✓ Saved best.pt  (MPJPE={best_mpjpe:.2f} mm)")
        else:
            patience_ctr += 1

        # Always save latest
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimiser":  optimiser.state_dict(),
            "best_mpjpe": best_mpjpe,
            "args":       vars(args),
        }, CKPT_DIR / "latest.pt")

        if patience_ctr >= args.patience:
            print(f"\n[Early stop] No improvement for {args.patience} epochs.")
            break

    print(f"\n[DONE] Best val MPJPE: {best_mpjpe:.2f} mm")
    print(f"Checkpoint : {CKPT_DIR / 'best.pt'}")
    print(f"Log        : {log_path}")
    print("Next: run 12_evaluate.py for full MPJPE and contact metrics.")


if __name__ == "__main__":
    main()
