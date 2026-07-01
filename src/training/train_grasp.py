"""3-phase training for TemporalGeometryConditionedGraspModel (docs/B-grasp-model.md B1).

Phase 1: OakInk static only (T=1, frame_feat), CVAE beta KL warmup 0->1e-3.
Phase 2: HOT3D temporal (T=window), GT velocity/acceleration loss terms activated.
Phase 3: Object encoder (PointNet) frozen, success_weight activated if success_labels present.

Usage:
  python3 src/training/train_grasp.py --phase 1 --epochs 30
  python3 src/training/train_grasp.py --phase 2 --epochs 50 --checkpoint checkpoints/grasp_best.pt
  python3 src/training/train_grasp.py --phase 3 --epochs 20 --checkpoint checkpoints/grasp_best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.dataset_hot3d import Hot3DTemporalDataset, collate_hot3d  # noqa: E402
from data.dataset_oakink import OakInkStaticDataset, collate_oakink  # noqa: E402
from model.grasp_model import GraspModel, grasp_loss  # noqa: E402
from utils.paths import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def kl_schedule(epoch: int, warmup_epochs: int, max_kl: float) -> float:
    """Linearly ramp KL weight from 0 to max_kl over warmup_epochs."""
    if warmup_epochs <= 0:
        return max_kl
    return min(1.0, epoch / warmup_epochs) * max_kl


def run_batch(
    model: GraspModel,
    batch: dict,
    dev: torch.device,
    args: argparse.Namespace,
    kl_weight: float,
    phase: int,
) -> tuple[torch.Tensor, dict]:
    obj_pts = batch["obj_pts"].to(dev)
    # obj_pts_contact: object surface in wrist frame (metres) — correct frame for FK comparison.
    # Falls back to None (disables contact/penetration loss) if absent, e.g. legacy batches.
    obj_pts_contact = batch.get("obj_pts_contact")
    obj_pts_contact = obj_pts_contact.to(dev) if obj_pts_contact is not None else None
    target = batch["target_pose"].to(dev)
    quality_label = batch.get("quality_label")
    quality_label = quality_label.to(dev).unsqueeze(-1) if quality_label is not None else None
    success_label = batch.get("success_label")
    success_label = success_label.to(dev).unsqueeze(-1) if success_label is not None else None

    frame_feat = batch["frame_feat"].to(dev)
    contact_flag = batch.get("contact_flag")
    if contact_flag is not None:
        contact_flag = contact_flag.to(dev)

    prev_pose = batch.get("prev_pose")
    prev2_pose = batch.get("prev2_pose")
    prev_pose = prev_pose.to(dev) if prev_pose is not None else None
    prev2_pose = prev2_pose.to(dev) if prev2_pose is not None else None

    vel_weight = args.vel_weight if phase >= 2 else 0.0
    acc_weight = args.acc_weight if phase >= 2 else 0.0
    success_weight = args.success_weight if (phase >= 3 and success_label is not None) else 0.0

    # Phase 2: run prev/prev2 passes BEFORE the main pass so their inplace GRU/FK ops
    # do not corrupt the main computation graph. All prev passes are no_grad + detached.
    prev_pred_pose = None
    prev2_pred_pose = None
    if vel_weight > 0 and prev_pose is not None:
        prev_frame_feat  = batch.get("prev_frame_feat")
        prev2_frame_feat = batch.get("prev2_frame_feat") if acc_weight > 0 else None
        with torch.no_grad():
            if prev_frame_feat is not None:
                # prev_out is conditioned on prev2_pose (the step before prev_pose)
                prev_out = model.forward_train(
                    prev_frame_feat.to(dev), obj_pts.clone(), prev_pose,
                    prev_pose=prev2_pose,
                    contact_flag=contact_flag,
                )
                prev_pred_pose = prev_out["pred"].detach()
            if prev2_frame_feat is not None and prev2_pose is not None:
                prev2_out = model.forward_train(
                    prev2_frame_feat.to(dev), obj_pts.clone(), prev2_pose,
                    prev_pose=None,  # no known prior before prev2; model uses zeros
                    contact_flag=contact_flag,
                )
                prev2_pred_pose = prev2_out["pred"].detach()

    out = model.forward_train(frame_feat, obj_pts, target,
                              prev_pose=prev_pose,
                              contact_flag=contact_flag)
    pred_pose = out["pred"]

    loss, info = grasp_loss(
        pred_pose,
        target,
        out["mu"],
        out["logvar"],
        out["quality_score"],
        obj_pts=obj_pts_contact,
        quality_label=quality_label,
        success_prob=out["success_prob"] if success_weight > 0 else None,
        success_label=success_label,
        prev_target_pose=prev_pose,
        prev_pred_pose=prev_pred_pose,
        prev2_target_pose=prev2_pose,
        prev2_pred_pose=prev2_pred_pose,
        kl_weight=kl_weight,
        limit_weight=args.limit_weight,
        contact_weight=args.contact_weight,
        penetration_weight=args.penetration_weight,
        quality_weight=args.quality_weight,
        success_weight=success_weight,
        vel_weight=vel_weight,
        acc_weight=acc_weight,
        tip_weight=args.tip_weight,
    )
    return loss, info


def freeze_object_encoder(model: GraspModel) -> None:
    for p in model.obj_encoder.parameters():
        p.requires_grad_(False)


def unfreeze_object_encoder(model: GraspModel) -> None:
    for p in model.obj_encoder.parameters():
        p.requires_grad_(True)


def make_loader(ds, batch_size: int, shuffle: bool, workers: int, collate_fn) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_fn)


class MixedDataLoader:
    """Phase 2 için HOT3D (%hot3d_ratio) + OakInk (1-hot3d_ratio) karışık loader.

    HOT3D'nin bir tam geçişi bir epoch'u tanımlar. OakInk loader, epoch başında
    sonsuz döngüyle sıfırlanır — küçük olduğu için bu gereklidir.
    Batch'ler HOT3D ve OakInk'in farklı alanlarını taşır; run_batch her ikisini
    de opsiyonel alanlarla (prev_frame_feat, contact_flag vb.) doğru şekilde işler:
      - HOT3D batch gelince vel/acc loss aktif olur.
      - OakInk batch gelince prev_frame_feat=None, vel/acc skip edilir.
    """
    def __init__(
        self,
        hot3d_loader: DataLoader,
        oakink_loader: DataLoader,
        hot3d_ratio: float = 0.7,
    ) -> None:
        self.hot3d_loader  = hot3d_loader
        self.oakink_loader = oakink_loader
        self.hot3d_ratio   = hot3d_ratio
        # HOT3D bir epoch tanımlar; OakInk o epoch içinde kaç batch vermeli?
        n_hot = len(hot3d_loader)
        self._n_oak = max(1, round(n_hot * (1.0 - hot3d_ratio) / hot3d_ratio))
        self._len   = n_hot + self._n_oak

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        hot_iter = iter(self.hot3d_loader)
        oak_cycle = self._oak_cycle()

        n_hot = len(self.hot3d_loader)
        # Hangi konumların OakInk olacağını belirle (eşit aralıklı)
        oak_positions: set[int] = set(
            round(i * (n_hot + self._n_oak) / self._n_oak)
            for i in range(self._n_oak)
        )

        pos = 0
        hot_exhausted = False
        while not hot_exhausted:
            if pos in oak_positions:
                yield next(oak_cycle)
            else:
                try:
                    yield next(hot_iter)
                except StopIteration:
                    hot_exhausted = True
            pos += 1

    def _oak_cycle(self):
        """OakInk loader'ı sonsuz döngüyle tekrarla."""
        while True:
            yield from self.oakink_loader


@torch.no_grad()
def run_val(model: GraspModel, loader: DataLoader, dev: torch.device, args: argparse.Namespace, phase: int) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    n_total = 0
    for batch in loader:
        _, info = run_batch(model, batch, dev, args, kl_weight=args.kl_weight, phase=phase)
        bs = int(len(batch["target_pose"]))
        for k, v in info.items():
            totals[k] = totals.get(k, 0.0) + v * bs
        n_total += bs
    return {k: v / max(1, n_total) for k, v in totals.items()}


def train_phase(
    model: GraspModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    args: argparse.Namespace,
    phase: int,
    start_epoch: int = 0,
) -> dict:
    dev = get_device()
    model = model.to(dev)
    if phase >= 3:
        freeze_object_encoder(model)
    else:
        unfreeze_object_encoder(model)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6,
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    log: list[dict] = []
    best_val = float("inf")
    no_improve = 0
    prefix = getattr(args, "checkpoint_prefix", None) or "grasp"
    best_path = CHECKPOINT_DIR / f"{prefix}_phase{phase}_best.pt"
    log_path = RESULTS_DIR / f"train_{prefix}_phase{phase}_log.json"

    for epoch in range(args.epochs):
        model.train()
        kl_w = kl_schedule(epoch, args.kl_warmup_epochs, args.kl_weight)
        t0 = time.time()
        totals: dict[str, float] = {}
        n_total = 0
        for batch in train_loader:
            loss, info = run_batch(model, batch, dev, args, kl_weight=kl_w, phase=phase)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = int(len(batch["target_pose"]))
            for k, v in info.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            n_total += bs

        entry: dict = {
            "phase": phase,
            "epoch": start_epoch + epoch,
            "kl_weight": kl_w,
            "lr": opt.param_groups[0]["lr"],
            "time_s": round(time.time() - t0, 2),
        }
        entry.update({f"train_{k}": v / max(1, n_total) for k, v in totals.items()})
        val_total = entry["train_total"]

        if val_loader is not None:
            vals = run_val(model, val_loader, dev, args, phase)
            entry.update({f"val_{k}": v for k, v in vals.items()})
            val_total = vals.get("total", val_total)

        scheduler.step(val_total)

        log.append(entry)
        log_path.write_text(json.dumps(log, indent=2))
        print(json.dumps(entry))

        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": start_epoch + epoch, "phase": phase}
        torch.save(ckpt, CHECKPOINT_DIR / f"{prefix}_phase{phase}_latest.pt")
        if val_total < best_val:
            best_val = val_total
            no_improve = 0
            torch.save(ckpt, best_path)
        else:
            no_improve += 1
            if args.early_stopping > 0 and no_improve >= args.early_stopping:
                print(f"Early stopping: {no_improve} epoch iyileşme yok (best_val={best_val:.4f})")
                break

    return {"best_val": best_val, "best_path": str(best_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-2)
    parser.add_argument("--kl_warmup_epochs", type=int, default=10)
    parser.add_argument("--limit_weight", type=float, default=1.0)
    parser.add_argument("--contact_weight", type=float, default=0.3)
    parser.add_argument("--penetration_weight", type=float, default=0.3)
    parser.add_argument("--quality_weight", type=float, default=0.1)
    parser.add_argument("--success_weight", type=float, default=0.1)
    parser.add_argument("--vel_weight", type=float, default=0.1)
    parser.add_argument("--acc_weight", type=float, default=0.05)
    parser.add_argument("--tip_weight", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-prefix", type=str, default="grasp", dest="checkpoint_prefix",
                        help="prefix for checkpoint filenames, used by orchestrate.py to separate experiments")
    parser.add_argument("--hot3d_split", choices=["train", "all"], default="train")
    parser.add_argument("--val_stride", type=int, default=8)
    parser.add_argument("--oakink_replay_ratio", type=float, default=0.3,
                        help="Phase 2'de OakInk batch oranı (0=devre dışı, 0.3=%%30 OakInk)")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="ReduceLROnPlateau: bu kadar epoch iyileşme yoksa LR yarıya")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Bu kadar epoch iyileşme yoksa dur (0=kapalı)")
    args = parser.parse_args()

    model = GraspModel(hidden=args.hidden, z_dim=args.z_dim)
    start_epoch = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    if args.phase == 1:
        train_ds = OakInkStaticDataset(split="train", n_points=args.n_points, augment=True)
        val_ds = OakInkStaticDataset(split="val", n_points=args.n_points, augment=False)
        train_loader = make_loader(train_ds, args.batch, True, args.workers, collate_oakink)
        val_loader = make_loader(val_ds, args.batch * 2, False, args.workers, collate_oakink)
    elif args.phase == 2:
        hot3d_train_ds = Hot3DTemporalDataset(split=args.hot3d_split, window=args.window, n_points=args.n_points)
        val_ds = Hot3DTemporalDataset(split="val", window=args.window, n_points=args.n_points, stride=args.val_stride)
        hot3d_loader = make_loader(hot3d_train_ds, args.batch, True, args.workers, collate_hot3d)
        val_loader = make_loader(val_ds, args.batch * 2, False, args.workers, collate_hot3d) if len(val_ds) > 0 else None
        if args.oakink_replay_ratio > 0:
            oak_replay_ds = OakInkStaticDataset(split="train", n_points=args.n_points, augment=True)
            oak_loader = make_loader(oak_replay_ds, args.batch, True, args.workers, collate_oakink)
            train_loader = MixedDataLoader(hot3d_loader, oak_loader,
                                           hot3d_ratio=1.0 - args.oakink_replay_ratio)
            print(f"Phase 2 mixed: HOT3D {len(hot3d_train_ds)} + OakInk replay {len(oak_replay_ds)} "
                  f"(ratio {1-args.oakink_replay_ratio:.0%}/{args.oakink_replay_ratio:.0%})")
        else:
            train_loader = hot3d_loader
    else:  # phase 3: mix for success_label fine-tuning (optional, skip if no success_label in NPZ)
        train_ds = Hot3DTemporalDataset(split=args.hot3d_split, window=args.window, n_points=args.n_points)
        train_loader = make_loader(train_ds, args.batch, True, args.workers, collate_hot3d)
        val_loader = None

    print(f"Phase {args.phase}: {len(train_loader)} train batches | device={get_device()}")
    train_phase(model, train_loader, val_loader, args, args.phase, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
