"""
Training Orchestrator — Intent-Aware XR Framework
=================================================

This script handles the optimization loop for the IntentFormer model, 
supporting standalone (H2O or HOT3D) or combined (multi-dataset) training.

Optimization Targets:
---------------------
1. Early Prediction Accuracy: The model is evaluated on how correctly it 
   identifies intent with only 20%, 25%, and 30% of the movement observed.
   
2. Low Latency: Designed to achieve <5ms inference on Apple Silicon.

Hardware Acceleration:
---------------------
The script is optimized for Apple M-series chips (M1/M2/M3 Max). 
It leverages the `mps` (Metal Performance Shaders) backend for 
fast tensor operations on the integrated GPU.

Key Args:
---------
- --dataset: h2o | hot3d | combined
- --fusion:  shared_head (3 classes) | concat (36 classes)
- --device:  mps (default) | cuda | cpu

Output:
-------
Checkpoints are saved to `checkpoints/`, including the best model per 
validation accuracy and the final state.
"""


import os
import csv
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.h2o_dataset    import get_dataloaders,         NUM_CLASSES         as H2O_NUM_CLASSES
from src.data.hot3d_dataset  import get_hot3d_dataloaders,   NUM_CLASSES_HOT3D
from src.data.combined_dataset import (
    get_combined_dataloaders,
    num_classes_for_fusion,
)
from src.models.intent_former import IntentFormer, EarlyPredictionLoss
from src.evaluate import compute_metrics


# ─────────────────────────────────────────────────────────
# Device selection (MPS → CUDA → CPU)
# ─────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[train] Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[train] Using CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[train] Using CPU")
    return device


# ─────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, pose_weight=5.0):
    model.train()
    total_loss, total_intent_loss, total_pose_loss = 0.0, 0.0, 0.0
    correct, total = 0, 0

    mse_loss = torch.nn.MSELoss()

    for batch in loader:
        hand   = batch["hand_flat"].to(device, non_blocking=True)   # (B, T, 126)
        obj    = batch["obj_rt"].to(device,    non_blocking=True)   # (B, T, 16)
        obs    = batch["obs_ratio"].to(device, non_blocking=True)   # (B,)
        labels = batch["label"].to(device,     non_blocking=True)   # (B,)
        target_pose = batch["target_pose"].to(device, non_blocking=True) # (B, 126)

        # ── Data Augmentation: Add subtle Gaussian noise ──────
        if model.training:
            # 2mm noise standard deviation (0.002m)
            hand = hand + torch.randn_like(hand) * 0.002

        optimizer.zero_grad(set_to_none=True)

        # ── Forward Pass ──────────────────────────────────────
        # IntentFormer now returns a tuple (logits, pred_pose)
        logits, pred_pose = model(hand, obj, obs)
        
        # ── Multi-task Loss ───────────────────────────────────
        intent_loss = criterion(logits, labels, obs)
        pose_loss   = mse_loss(pred_pose, target_pose)
        loss        = intent_loss + pose_weight * pose_loss

        if scaler is not None:                              # AMP (CUDA/MPS)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss        += loss.item() * labels.size(0)
        total_intent_loss += intent_loss.item() * labels.size(0)
        total_pose_loss   += pose_loss.item() * labels.size(0)
        correct           += (logits.argmax(1) == labels).sum().item()
        total             += labels.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        hand   = batch["hand_flat"].to(device, non_blocking=True)
        obj    = batch["obj_rt"].to(device,    non_blocking=True)
        obs    = batch["obs_ratio"].to(device, non_blocking=True)
        labels = batch["label"].to(device,     non_blocking=True)

        # Handle tuple output
        logits, _ = model(hand, obj, obs)
        loss      = criterion(logits, labels, obs)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds_all  = torch.cat(all_preds)
    labels_all = torch.cat(all_labels)
    metrics    = compute_metrics(preds_all, labels_all, num_classes)
    return total_loss / total, correct / total, metrics


# ─────────────────────────────────────────────────────────
# Dataset / loader factory
# ─────────────────────────────────────────────────────────

def build_loaders(args):
    """
    Return (train_loader, val_loader, test_loader, num_classes)
    depending on --dataset choice.
    """
    obs_ratios = [float(r) for r in args.obs_ratios.split(",")]

    if args.dataset == "h2o":
        print(f"[train] Dataset: H2O  root={args.data_root}")
        train_loader, val_loader, test_loader = get_dataloaders(
            root_dir    = args.data_root,
            batch_size  = args.batch_size,
            window_size = args.window_size,
            obs_ratios  = obs_ratios,
            num_workers = args.num_workers,
            dense       = args.dense,
            stride      = args.stride,
        )
        return train_loader, val_loader, test_loader, H2O_NUM_CLASSES

    elif args.dataset == "hot3d":
        print(f"[train] Dataset: HOT3D  root={args.hot3d_root}")
        if not args.hot3d_root:
            raise ValueError("--hot3d_root must be set when --dataset hot3d")
        train_loader, test_loader = get_hot3d_dataloaders(
            root_dir    = args.hot3d_root,
            batch_size  = args.batch_size,
            window_size = args.window_size,
            obs_ratios  = obs_ratios,
            num_workers = args.num_workers,
            max_clips   = args.hot3d_max_clips,
        )
        # HOT3D has no official val split — reuse test as val
        return train_loader, test_loader, test_loader, NUM_CLASSES_HOT3D

    elif args.dataset == "combined":
        if not args.hot3d_root:
            raise ValueError("--hot3d_root must be set when --dataset combined")
        print(
            f"[train] Dataset: Combined (H2O + HOT3D)  "
            f"fusion={args.fusion}  "
            f"h2o={args.data_root}  hot3d={args.hot3d_root}"
        )
        train_loader, val_loader, test_loader = get_combined_dataloaders(
            h2o_root    = args.data_root,
            hot3d_root  = args.hot3d_root,
            fusion      = args.fusion,
            batch_size  = args.batch_size,
            window_size = args.window_size,
            obs_ratios  = obs_ratios,
            num_workers = args.num_workers,
            hot3d_max_clips = args.hot3d_max_clips,
        )
        n_cls = num_classes_for_fusion(args.fusion)
        return train_loader, val_loader, test_loader, n_cls

    else:
        raise ValueError(f"Unknown --dataset: {args.dataset}")


# ─────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────

def train(args):
    device  = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────
    train_loader, val_loader, _, num_classes = build_loaders(args)

    # ── Model ─────────────────────────────────────────────
    model = IntentFormer(
        input_dim       = 378 + 16,     # (126*3) hand + 16 obj_rt
        d_model         = args.d_model,
        nhead           = args.nhead,
        num_layers      = args.num_layers,
        dim_feedforward = args.dim_ff,
        num_classes     = num_classes,
        window_size     = args.window_size,
        dropout         = args.dropout,
    ).to(device)
    print(f"[train] IntentFormer params: {model.num_parameters():,}  "
          f"num_classes={num_classes}")

    # ── Loss, Optimizer, Scheduler ────────────────────────
    criterion = EarlyPredictionLoss(alpha=args.ep_alpha, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Metrics CSV ───────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "dataset", "fusion",
                         "train_loss", "train_acc",
                         "val_loss",   "val_acc",
                         "val_precision", "secs"])

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            pose_weight=args.pose_weight
        )
        val_loss, val_acc, metrics = validate(
            model, val_loader, criterion, device, num_classes
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_prec={metrics['precision']:.4f}  {elapsed:.1f}s"
        )

        # ── Save metrics ──────────────────────────────────────
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, args.dataset, getattr(args, "fusion", "N/A"),
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}",   f"{val_acc:.6f}",
                f"{metrics['precision']:.6f}", f"{elapsed:.2f}",
            ])

        # ── Save checkpoints ──────────────────────────────────
        last_ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_acc":     val_acc,
            "metrics":     metrics,
            "num_classes": num_classes,
            "dataset":     args.dataset,
            "fusion":      getattr(args, "fusion", "N/A"),
        }
        torch.save(last_ckpt, out_dir / "last_model.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(last_ckpt, out_dir / "best_model.pt")
            print(f"  ✓ New best val_acc = {best_val_acc:.4f}  (saved)")

    print(f"\n[train] Finished. Best val_acc = {best_val_acc:.4f}")
    print(f"[train] Checkpoints → {out_dir}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train IntentFormer on H2O, HOT3D, or both"
    )

    # ── Dataset selection ─────────────────────────────────
    p.add_argument("--dataset",    default="h2o",
                   choices=["h2o", "hot3d", "combined"],
                   help="Which dataset(s) to train on")
    p.add_argument("--data_root",  default="data/h2o",
                   help="Path to H2O root directory")
    p.add_argument("--hot3d_root", default="",
                   help="Path to HOT3D-Clips root directory")
    p.add_argument("--fusion",     default="concat",
                   choices=["concat", "shared_head"],
                   help="Label fusion strategy for combined training")
    p.add_argument("--hot3d_max_clips", type=int, default=None,
                   help="Limit HOT3D clips loaded (for quick tests)")

    # ── Training hyperparameters ─────────────────────────
    p.add_argument("--out_dir",      default="checkpoints")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--window_size",  type=int,   default=30)
    p.add_argument("--obs_ratios",   default="0.2,0.25,0.3,0.4,0.5",
                   help="Comma-separated observation ratios")
    p.add_argument("--dense",        action="store_true",
                   help="If set, samples windows throughout the action (sliding window train)")
    p.add_argument("--stride",       type=int,   default=5,
                   help="Stride for dense sampling")

    # ── Model hyperparameters ────────────────────────────
    p.add_argument("--d_model",      type=int,   default=256)
    p.add_argument("--nhead",        type=int,   default=8)
    p.add_argument("--num_layers",   type=int,   default=6)
    p.add_argument("--dim_ff",       type=int,   default=1024)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--ep_alpha",     type=float, default=2.0)
    p.add_argument("--pose_weight",  type=float, default=5.0)
    p.add_argument("--num_workers",  type=int,   default=4)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
