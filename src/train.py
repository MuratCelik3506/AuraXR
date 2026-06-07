"""train.py — train AuraXRModel and save checkpoints.

Run:
    python train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/
    python train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/

Expects dataset.h5 (15-dim features: dir_world(3)+dir_obj_local(3)+dist(1)+approach_speed(1)+grip_oh(4)+bbox(3)).
Build with: python build_dataset.py --hand right --output_dir ../data/right/

Hardware: auto-detects MPS (Apple Silicon) > CUDA > CPU.
Class balance: grip frames oversampled 10x in dataset.
Loss: compound_loss = weighted Huber + grip phase 2x + range penalty + DIP-PIP coupling + grip classifier.
"""

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from model import AuraXRModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Train AuraXR hand pose model.")
    p.add_argument("--data_dir",      required=True,  type=Path)
    p.add_argument("--output_dir",    required=True,  type=Path)
    p.add_argument("--epochs",        default=50000,  type=int)
    p.add_argument("--batch_size",    default=131072, type=int)
    p.add_argument("--lr",            default=5e-3,   type=float)
    p.add_argument("--weight_decay",  default=3e-4,   type=float)
    p.add_argument("--hidden_dim",    default=512,    type=int)
    p.add_argument("--embedding_dim", default=256,    type=int)
    p.add_argument("--dropout",       default=0.25,   type=float)
    p.add_argument("--patience",      default=4000,   type=int)
    p.add_argument("--warmup_epochs", default=200,    type=int)
    p.add_argument("--seed",          default=42,     type=int)
    p.add_argument("--resume",        action="store_true",
                   help="Load best_model.pt from output_dir before training")
    p.add_argument("--no_amp",        action="store_true",
                   help="Disable bfloat16 autocast")
    p.add_argument("--no_compile",    action="store_true",
                   help="Disable torch.compile (use eager mode)")
    return p.parse_args()


# ── Joint weight constants ───────────────────────────────────────────────────
# One weight per active joint (indices 0–19), in UME order:
#   Thumb  [0-3]:  CMC-flex, abduction, MCP, DIP
#   Index  [4-7]:  abduction, MCP, PIP, DIP
#   Middle [8-11]: abduction, MCP, PIP, DIP
#   Ring   [12-15]:abduction, MCP, PIP, DIP
#   Pinky  [16-19]:abduction, MCP, PIP, DIP
_JOINT_WEIGHTS_RAW = [
    2.0, 0.5, 1.5, 1.5,   # Thumb:  CMC-flex high, abduction low
    0.5, 1.5, 2.5, 1.5,   # Index:  PIP high (most visible)
    0.5, 1.5, 2.5, 1.5,   # Middle: PIP high
    0.5, 1.5, 2.5, 1.5,   # Ring:   PIP high
    0.5, 3.5, 3.0, 1.5,   # Pinky:  MCP raised 1.8→3.5 (23.6° error, worst joint)
]

# Indices into the 22-dim pred tensor for range/coupling penalties
_FLEX_IDX = [0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]  # all non-abduction
_PIP_IDX  = [6, 10, 14, 18]   # PIP joints (Index/Mid/Ring/Pinky)
_DIP_IDX  = [7, 11, 15, 19]   # DIP joints (Index/Mid/Ring/Pinky)

# Joints 20–21 are always 0.0 in HOT3D — mask them from the loss.
_ACTIVE = torch.tensor(AuraXRModel.ACTIVE_JOINTS)  # indices 0–19


def compound_loss(
    pred,         # (B, 22) normalized model output
    tgt,          # (B, 22) normalized targets
    dist,         # (B,)    raw distance in metres
    grip_logits,  # (B, 4)  from forward_train()
    grip_labels,  # (B,)    long, grip category index
    joint_weights,# (20,)   per-joint weights tensor
    tgt_mean,     # (22,)   target normalisation mean
    tgt_std,      # (22,)   target normalisation std
    active,       # (20,)   active joint indices (0–19)
    beta: float = 0.5,
) -> torch.Tensor:
    # 1. Per-joint weighted Huber over active joints
    diff  = (pred[:, active] - tgt[:, active]).abs()
    huber = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    per_sample = (huber * joint_weights).mean(dim=-1)   # (B,)

    # 2. Grip phase 2x weight — emphasise close contact frames on top of oversampling
    grip_w = torch.where(dist < 0.10,
                         torch.full_like(dist, 2.0),
                         torch.ones_like(dist))
    angle_loss = (per_sample * grip_w).mean()

    # 3. Range penalty in raw angle space — penalise impossible poses
    #    Flexion joints should be in [0, 2.0] radians.
    flex_tgt_std  = tgt_std[_FLEX_IDX]
    flex_tgt_mean = tgt_mean[_FLEX_IDX]
    pred_flex_raw = pred[:, _FLEX_IDX] * flex_tgt_std + flex_tgt_mean
    range_pen = (F.relu(-pred_flex_raw) + F.relu(pred_flex_raw - 2.0)).mean()

    # 4. DIP-PIP coupling — anatomical ratio DIP ≈ 0.67 × PIP
    pip_raw = pred[:, _PIP_IDX] * tgt_std[_PIP_IDX] + tgt_mean[_PIP_IDX]
    dip_raw = pred[:, _DIP_IDX] * tgt_std[_DIP_IDX] + tgt_mean[_DIP_IDX]
    coupling = (dip_raw - 0.67 * pip_raw).abs().mean()

    # 5. Auxiliary grip classification (CE loss)
    cls_loss = F.cross_entropy(grip_logits, grip_labels)

    return angle_loss + 0.3 * range_pen + 0.2 * coupling + 0.15 * cls_loss


def load_split_to_device(hdf5_path: Path, split: str, meta: dict, device: torch.device):
    feat_mean = torch.tensor(meta["feature_mean"],   dtype=torch.float32)
    feat_std  = torch.tensor(meta["feature_std"],    dtype=torch.float32)
    tgt_mean  = torch.tensor(meta["target_mean"],    dtype=torch.float32)
    tgt_std   = torch.tensor(meta["target_std"],     dtype=torch.float32)
    rot_mean  = torch.tensor(meta["wrist_rot_mean"], dtype=torch.float32)
    rot_std   = torch.tensor(meta["wrist_rot_std"],  dtype=torch.float32)

    with h5py.File(hdf5_path, "r") as hf:
        feat_np    = hf[split]["features"][:]
        tgt_np     = hf[split]["targets"][:]
        rot_np     = hf[split]["wrist_rot_6d"][:]
        dist_np    = hf[split]["distances"][:]

    # Grip label = argmax of one-hot at feat[:, 8:12], before normalization
    grip_labels = torch.from_numpy(feat_np[:, 8:12].argmax(axis=1)).long().to(device)

    feat = ((torch.from_numpy(feat_np) - feat_mean) / (feat_std + 1e-8)).to(device)
    tgt  = ((torch.from_numpy(tgt_np)  - tgt_mean)  / (tgt_std  + 1e-8)).to(device)
    rot  = ((torch.from_numpy(rot_np)  - rot_mean)   / (rot_std  + 1e-8)).to(device)
    dist = torch.from_numpy(dist_np).to(device)
    return feat, tgt, rot, dist, grip_labels


def run_epoch(
    model, feat, tgt, rot, dist, grip_labels,
    optimizer, batch_size, active,
    joint_weights, tgt_mean, tgt_std,
    training: bool, device: torch.device, use_amp: bool = True,
) -> float:
    model.train(training)
    N = len(feat)
    total_loss = 0.0

    amp_enabled = use_amp and device.type in ("mps", "cuda")

    idx = torch.randperm(N, device=feat.device) if training else torch.arange(N, device=feat.device)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for start in range(0, N, batch_size):
            b   = idx[start:start + batch_size]
            f, t, r, d, gl = feat[b], tgt[b], rot[b], dist[b], grip_labels[b]
            if training:
                f = f.clone()
                f[:, :3]  += 0.02 * torch.randn_like(f[:, :3])    # dir_world
                f[:, 3:6] += 0.02 * torch.randn_like(f[:, 3:6])   # dir_obj_local
                perturb_d = 1.0 + 0.10 * (2.0 * torch.rand(len(f), device=f.device) - 1.0)
                f[:, 6]  = f[:, 6] * perturb_d                     # distance ±10%
                f[:, 7]  += 0.05 * torch.randn_like(f[:, 7])       # approach_speed additive noise
                perturb_b = 1.0 + 0.05 * (2.0 * torch.rand(len(f), 3, device=f.device) - 1.0)
                f[:, 12:15] = f[:, 12:15] * perturb_b              # bbox ±5%
            sp, ob = AuraXRModel.split_feature(f)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                pred_joints, pred_rot, grip_logits = model.forward_train(sp, ob)

            # Loss in float32 for numerical stability regardless of AMP dtype
            angle_loss = compound_loss(
                pred_joints.float(), t, d, grip_logits.float(), gl,
                joint_weights, tgt_mean, tgt_std, active,
            )
            wrist_loss = F.mse_loss(pred_rot.float(), r)
            loss = angle_loss + 0.3 * wrist_loss

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
    train_feat, train_tgt, train_rot, train_dist, train_grip = load_split_to_device(
        hdf5_path, "train", meta, device)
    val_feat, val_tgt, val_rot, val_dist, val_grip = load_split_to_device(
        hdf5_path, "val", meta, device)

    active = _ACTIVE.to(device)

    # Joint weights tensor on device (20 active joints)
    joint_weights = torch.tensor(_JOINT_WEIGHTS_RAW, dtype=torch.float32, device=device)

    # Target normalisation stats on device (for range penalty + coupling in raw space)
    tgt_mean = torch.tensor(meta["target_mean"], dtype=torch.float32, device=device)
    tgt_std  = torch.tensor(meta["target_std"],  dtype=torch.float32, device=device)

    n_train, n_val = len(train_feat), len(val_feat)
    grip_train = (train_dist < 0.10).sum().item()
    print(f"Train: {n_train}  Val: {n_val}", flush=True)
    print(f"  Grip (<10cm):  {grip_train:>7}  ({100*grip_train/n_train:.1f}%)", flush=True)
    print(f"  Pre-shape:     {n_train-grip_train:>7}  ({100*(n_train-grip_train)/n_train:.1f}%)", flush=True)
    print(f"Model: hidden={args.hidden_dim}  emb={args.embedding_dim}  "
          f"dropout={args.dropout}  active_joints=20/22", flush=True)

    model = AuraXRModel(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    if args.resume:
        ckpt = args.output_dir / "best_model.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            # Strip _orig_mod. prefix if checkpoint was saved from a compiled model
            if any(k.startswith("_orig_mod.") for k in state):
                state = {k[len("_orig_mod."):]: v for k, v in state.items()}
            model.load_state_dict(state)
            print(f"Resumed weights from {ckpt}", flush=True)
        else:
            print(f"[WARN] --resume set but {ckpt} not found — training from scratch.", flush=True)

    use_amp = not args.no_amp
    # bfloat16 on MPS is not faster than float32 for hidden=256 matrices and
    # causes NaN overflow on small trailing batches — always use float32 on MPS
    if device.type == "mps" and use_amp:
        use_amp = False
        print("AMP: disabled on MPS (float32 equally fast, more stable)", flush=True)
    elif use_amp:
        print(f"AMP: bfloat16 autocast enabled ({device.type})", flush=True)

    if not args.no_compile:
        try:
            # aot_eager: works on MPS without Triton; eliminates Python dispatch overhead
            model = torch.compile(model, backend="aot_eager", fullgraph=False)
            print("torch.compile: enabled (aot_eager backend)", flush=True)
        except Exception as e:
            print(f"torch.compile: skipped ({e})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Linear warmup, then CosineAnnealingWarmRestarts
    # T_0=4000: scaled for 8× larger batch (fewer grad steps/epoch) — restarts at 4k, 12k, 28k
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=4000, T_mult=2, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    log = []

    print(f"\nTraining up to {args.epochs} epochs (early-stop patience={args.patience}) …", flush=True)
    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model, train_feat, train_tgt, train_rot, train_dist, train_grip,
            optimizer, args.batch_size, active,
            joint_weights, tgt_mean, tgt_std, training=True,
            device=device, use_amp=use_amp,
        )
        val_loss = run_epoch(
            model, val_feat, val_tgt, val_rot, val_dist, val_grip,
            None, args.batch_size, active,
            joint_weights, tgt_mean, tgt_std, training=False,
            device=device, use_amp=use_amp,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})

        if epoch % 200 == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.2f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            # Always save bare weights (strip _orig_mod. prefix from compiled models)
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw.state_dict(), args.output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).", flush=True)
                break

    total_time    = time.time() - t_total
    actual_epochs = len(log)
    print(f"\nDone in {total_time:.1f}s  ({total_time/max(1, actual_epochs):.2f}s/epoch)", flush=True)
    print(f"Best val loss: {best_val_loss:.6f}  (epoch {best_epoch})", flush=True)

    with open(args.output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

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
