"""train.py — Step 3: train AuraXRModel and save checkpoints.

Run:
    python train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/
    python train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/

Hardware: auto-detects MPS (Apple Silicon) > CUDA > CPU.
Speed: entire dataset pre-loaded onto GPU — ~0.3s/epoch on M2 Max for 500 epochs ≈ 3 min.
"""

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from model import AuraXRModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Train AuraXR hand pose model.")
    p.add_argument("--data_dir",     required=True,  type=Path)
    p.add_argument("--output_dir",   required=True,  type=Path)
    p.add_argument("--epochs",       default=500,    type=int)
    p.add_argument("--batch_size",   default=4096,   type=int)
    p.add_argument("--lr",           default=1e-3,   type=float)
    p.add_argument("--weight_decay", default=1e-4,   type=float)
    p.add_argument("--grip_weight",  default=6.0,    type=float,
                   help="Loss multiplier for grip frames (distance < 10 cm). "
                        "Only 2.2%% of data — needs strong signal.")
    p.add_argument("--approach_weight", default=2.0, type=float,
                   help="Loss multiplier for approach frames (distance > 45 cm). "
                        "Synthetic augmentation samples — teach open-hand at far range.")
    p.add_argument("--warmup_epochs",default=20,     type=int,
                   help="Linear LR warmup before cosine decay kicks in.")
    p.add_argument("--hidden_dim",   default=128,    type=int)
    p.add_argument("--embedding_dim",default=64,     type=int)
    p.add_argument("--dropout",      default=0.40,   type=float)
    p.add_argument("--seed",         default=42,     type=int)
    return p.parse_args()


# Joints 20–21 are always 0.0 in HOT3D — mask them from the loss.
_ACTIVE = torch.tensor(AuraXRModel.ACTIVE_JOINTS)  # indices 0–19


def weighted_huber(pred: torch.Tensor, target: torch.Tensor,
                   weights: torch.Tensor, active: torch.Tensor,
                   beta: float = 0.5) -> torch.Tensor:
    """Weighted Huber (smooth-L1) over active joints — robust to outlier poses."""
    diff = (pred[:, active] - target[:, active]).abs()
    huber = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return (weights * huber.mean(dim=-1)).mean()


def load_split_to_device(hdf5_path: Path, split: str, meta: dict, device: torch.device):
    feat_mean = torch.tensor(meta["feature_mean"], dtype=torch.float32)
    feat_std  = torch.tensor(meta["feature_std"],  dtype=torch.float32)
    tgt_mean  = torch.tensor(meta["target_mean"],  dtype=torch.float32)
    tgt_std   = torch.tensor(meta["target_std"],   dtype=torch.float32)

    with h5py.File(hdf5_path, "r") as hf:
        feat_np = hf[split]["features"][:]
        tgt_np  = hf[split]["targets"][:]
        dist_np = hf[split]["distances"][:]

    feat = ((torch.from_numpy(feat_np) - feat_mean) / (feat_std + 1e-8)).to(device)
    tgt  = ((torch.from_numpy(tgt_np)  - tgt_mean)  / (tgt_std  + 1e-8)).to(device)
    dist = torch.from_numpy(dist_np).to(device)
    return feat, tgt, dist


def make_weights(dist: torch.Tensor, grip_weight: float, approach_weight: float) -> torch.Tensor:
    w = torch.ones_like(dist)
    w = torch.where(dist < 0.10, torch.full_like(dist, grip_weight), w)    # grip close-range
    w = torch.where(dist > 0.45, torch.full_like(dist, approach_weight), w) # synthetic approach
    return w


def run_epoch(model, feat, tgt, dist, optimizer, grip_weight, approach_weight, batch_size,
              active, training: bool) -> float:
    model.train(training)
    N = len(feat)
    total_loss = 0.0

    if training:
        idx = torch.randperm(N, device=feat.device)
    else:
        idx = torch.arange(N, device=feat.device)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for start in range(0, N, batch_size):
            b = idx[start:start + batch_size]
            f, t, d = feat[b], tgt[b], dist[b]
            sp, ob = AuraXRModel.split_feature(f)
            pred = model(sp, ob)
            loss = weighted_huber(pred, t, make_weights(d, grip_weight, approach_weight), active)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(f)

    return total_loss / N


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}", flush=True)

    hdf5_path = args.data_dir / "dataset.h5"
    if not hdf5_path.exists():
        print(f"[ERROR] {hdf5_path} not found.")
        return

    with h5py.File(hdf5_path, "r") as hf:
        meta = json.loads(hf.attrs["meta"])

    print("Loading dataset onto device …", flush=True)
    train_feat, train_tgt, train_dist = load_split_to_device(hdf5_path, "train", meta, device)
    val_feat,   val_tgt,   val_dist   = load_split_to_device(hdf5_path, "val",   meta, device)
    active = _ACTIVE.to(device)

    n_train, n_val = len(train_feat), len(val_feat)
    grip_train    = (train_dist < 0.10).sum().item()
    approach_train = (train_dist > 0.45).sum().item()
    print(f"Train: {n_train}  Val: {n_val}", flush=True)
    print(f"  Grip (<10cm):     {grip_train:>7}  ({100*grip_train/n_train:.1f}%)", flush=True)
    print(f"  Pre-shape (real): {n_train - grip_train - approach_train:>7}  ({100*(n_train-grip_train-approach_train)/n_train:.1f}%)", flush=True)
    print(f"  Approach (synth): {approach_train:>7}  ({100*approach_train/n_train:.1f}%)", flush=True)
    print(f"Model: hidden={args.hidden_dim}  emb={args.embedding_dim}  "
          f"dropout={args.dropout}  active_joints=20/22", flush=True)

    model = AuraXRModel(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup → cosine decay
    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        progress = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))  # cosine to ~0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch    = 0
    log = []

    print(f"\nTraining {args.epochs} epochs (warmup={args.warmup_epochs}) …", flush=True)
    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_feat, train_tgt, train_dist,
                               optimizer, args.grip_weight, args.approach_weight,
                               args.batch_size, active, True)
        val_loss   = run_epoch(model, val_feat, val_tgt, val_dist,
                               None, args.grip_weight, args.approach_weight,
                               args.batch_size, active, False)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.2f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    total_time = time.time() - t_total
    print(f"\nDone in {total_time:.1f}s  ({total_time/args.epochs:.2f}s/epoch)", flush=True)
    print(f"Best val loss: {best_val_loss:.6f}  (epoch {best_epoch})", flush=True)

    with open(args.output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Save meta — update architecture fields to match actual model
    meta["architecture"].update({
        "hidden_dim":    args.hidden_dim,
        "embedding_dim": args.embedding_dim,
    })
    with open(args.output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {args.output_dir}/best_model.pt", flush=True)
    print(f"Saved: {args.output_dir}/model_meta.json", flush=True)


if __name__ == "__main__":
    main()
