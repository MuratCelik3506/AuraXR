"""Faz 2A baseline egitim: ogrenilebilirlik kaniti (SOTA degil).

loss = pose_mse(pca15) + lambda_vel * vel_mse  (maskeli)
log  = bilesen-bazli train/val loss + val MPJPE(mm, teacher-forced)
"""
import os
import json
import time
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GraspSegments, collate, load_stats
from model import ControllerToHand, masked_mse, masked_vel_mse, POSE_DIM
from mano_right import ManoRight
from mano_torch import ManoTorchFK

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_REPO, "checkpoints")
RESULTS = os.path.join(_REPO, "results")


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mpjpe_mm(pred_pca, betas, wq, wt, gt_joints, mano, cap=3000):
    """numpy FK ile MPJPE (mm). frame'ler (N,*). cap kadar ornek."""
    N = len(pred_pca)
    idx = np.random.choice(N, min(cap, N), replace=False)
    errs = []
    for i in idx:
        aa = pred_pca[i]   # cikti zaten aa45 (decode yok)
        j = mano.fk(aa, betas[i], wq[i], wt[i], shaped=mano.shaped_cached(betas[i]))["joints"]
        errs.append(np.linalg.norm(j - gt_joints[i], axis=1).mean())
    return float(np.mean(errs) * 1000.0)


def gather_teacher(model, loader, dev, free=True):
    """preds + gt alanlarini frame duzeyinde topla (maskeli). free=True -> free-running."""
    model.eval()
    keys = dict(pred=[], target=[], betas=[], wq=[], wt=[], joints=[], cat=[])
    with torch.no_grad():
        for b in loader:
            feat = b["feat"].to(dev); cat = b["cat"].to(dev); tgt = b["target"].to(dev)
            if free:
                pred = model.forward_free(feat, cat).cpu().numpy()
            else:
                pred = model.forward_teacher(feat, cat, tgt).cpu().numpy()
            m = b["mask"].numpy().astype(bool)
            keys["pred"].append(pred[m]); keys["target"].append(b["target"].numpy()[m])
            keys["betas"].append(b["betas"].numpy()[m]); keys["wq"].append(b["wrist_q"].numpy()[m])
            keys["wt"].append(b["wrist_t"].numpy()[m]); keys["joints"].append(b["fk_joints"].numpy()[m])
            keys["cat"].append(b["cat"].numpy()[m])
    return {k: np.concatenate(v) for k, v in keys.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lambda_vel", type=float, default=1.0)
    ap.add_argument("--lambda_fk", type=float, default=0.0, help="FK pozisyon loss agirligi (m->katsayi)")
    ap.add_argument("--ss", action="store_true", help="scheduled sampling (AR icin)")
    ap.add_argument("--ss_max", type=float, default=0.5)
    ap.add_argument("--obj_size", action="store_true", help="obje bbox extent koşullama (+3 feat)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_prev", action="store_true", help="prev_pose'u kaldir (non-autoregressive)")
    ap.add_argument("--arch", default="lstm", choices=["lstm", "tcn", "transformer"])
    ap.add_argument("--warmup", type=int, default=2, help="warmup epoch (lineer LR rampasi)")
    ap.add_argument("--patience", type=int, default=8, help="erken durdurma sabri (val MPJPE)")
    ap.add_argument("--tag", default="best")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(CKPT, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    dev = device()
    print("device:", dev)

    mean, std = load_stats()
    feat_dim = 16 if args.obj_size else 13
    tr = DataLoader(GraspSegments("train", mean, std, obj_size=args.obj_size), batch_size=args.batch,
                    shuffle=True, collate_fn=collate)
    va = DataLoader(GraspSegments("val", mean, std, obj_size=args.obj_size), batch_size=args.batch,
                    shuffle=False, collate_fn=collate)
    mano = ManoRight()
    # FK loss CPU'da: MPS'te 16-eklem zinciri (tiny matmul) kernel-launch overhead'iyle cok yavas.
    mano_fk = ManoTorchFK() if args.lambda_fk > 0 else None
    model = ControllerToHand(hidden=args.hidden, layers=args.layers,
                             use_prev=not args.no_prev, feat_dim=feat_dim,
                             arch=args.arch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_factor(ep):  # ep: 1..epochs. warmup lineer -> cosine decay.
        if ep <= args.warmup:
            return ep / max(1, args.warmup)
        import math
        prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    print(f"params: {sum(p.numel() for p in model.parameters())} | feat_dim={feat_dim} "
          f"use_prev={not args.no_prev} fk={args.lambda_fk} ss={args.ss} "
          f"warmup={args.warmup} patience={args.patience}")

    FK_SUB = 256   # FK loss derin zincir; her batch rastgele alt-kume (stokastik tahmin, ~10x ucuz)

    def fk_loss(pred, b, mask):
        v = b["mask"].reshape(-1).bool()                      # CPU mask
        idx = v.nonzero(as_tuple=True)[0]
        if len(idx) > FK_SUB:
            idx = idx[torch.randperm(len(idx))[:FK_SUB]]
        P = pred.reshape(-1, POSE_DIM).cpu()[idx]             # MPS->CPU (autograd korunur)
        Bb = b["betas"].reshape(-1, 10)[idx]
        Q = b["wrist_q"].reshape(-1, 4)[idx]
        T = b["wrist_t"].reshape(-1, 3)[idx]
        GJ = b["fk_joints"].reshape(-1, 16, 3)[idx]
        J = mano_fk.joints(P, Bb, Q, T)
        return (J - GJ).norm(dim=2).mean().to(pred.device)

    log = []
    best = 1e9; bad = 0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        ss_prob = args.ss_max * min(1.0, (ep - 1) / max(1, 0.6 * args.epochs)) if args.ss else 0.0
        tl = tlp = tlv = tlf = 0.0; nb = 0
        for b in tr:
            feat = b["feat"].to(dev); cat = b["cat"].to(dev)
            tgt = b["target"].to(dev); mask = b["mask"].to(dev)
            if args.ss and not args.no_prev:
                pred = model.forward_scheduled(feat, cat, tgt, ss_prob)
            else:
                pred = model.forward_teacher(feat, cat, tgt)
            lp = masked_mse(pred, tgt, mask)
            lv = masked_vel_mse(pred, tgt, mask)
            lf = fk_loss(pred, b, mask) if mano_fk is not None else torch.zeros((), device=dev)
            loss = lp + args.lambda_vel * lv + args.lambda_fk * lf
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); tlp += lp.item(); tlv += lv.item(); tlf += float(lf); nb += 1
        # val (free-running -> gercek metrik)
        g = gather_teacher(model, va, dev, free=True)
        vpose = float(np.mean((g["pred"] - g["target"]) ** 2))
        vmpjpe = mpjpe_mm(g["pred"], g["betas"], g["wq"], g["wt"], g["joints"], mano)
        rec = dict(epoch=ep, train_loss=tl/nb, train_pose=tlp/nb, train_vel=tlv/nb,
                   train_fk=tlf/nb, ss_prob=round(ss_prob, 3),
                   val_pose_mse=vpose, val_mpjpe_mm=vmpjpe, sec=round(time.time()-t0, 1))
        log.append(rec)
        sched.step()
        flag = ""
        if vmpjpe < best - 1e-3:
            best = vmpjpe; bad = 0
            torch.save(dict(model=model.state_dict(), args=vars(args),
                            mean=mean, std=std), os.path.join(CKPT, f"c2h_{args.tag}.pt"))
            flag = " *best"
        else:
            bad += 1
        print(f"ep{ep:02d} lr={opt.param_groups[0]['lr']:.1e} train={rec['train_loss']:.4f} "
              f"(pose={rec['train_pose']:.4f} vel={rec['train_vel']:.4f}) | val_pose={vpose:.4f} "
              f"val_MPJPE={vmpjpe:.1f}mm {rec['sec']}s{flag}")
        if bad >= args.patience:
            print(f"erken durdurma (ep{ep}, {args.patience} epoch iyilesme yok)")
            break

    json.dump(log, open(os.path.join(RESULTS, f"train_log_{args.tag}.json"), "w"), indent=2)
    print("best val MPJPE (mm):", round(best, 2), f"| log -> results/train_log_{args.tag}.json")

    # --- en iyi checkpoint'i yukle + TEST'te kapsamli otomatik degerlendirme ---
    from eval_metrics import full_eval, print_report
    bestck = torch.load(os.path.join(CKPT, f"c2h_{args.tag}.pt"), map_location=dev, weights_only=False)
    eval_model = ControllerToHand(hidden=args.hidden, layers=args.layers,
                                  use_prev=not args.no_prev, feat_dim=feat_dim,
                                  arch=args.arch).to(dev)
    eval_model.load_state_dict(bestck["model"])
    rep = full_eval(eval_model, mano, mean, std, args.obj_size, dev, split="test", tag=args.tag)
    print_report(rep)
    print(f"otomatik eval -> results/eval_{args.tag}.json + by_object_{args.tag}.csv")


if __name__ == "__main__":
    main()
