"""Tum varyantlari TEST'te free-running MPJPE ile karsilastir (ayni harness).

baseline(noprev), A(fk), C(objsize), A+C(fk_objsize), B(ss) + statik baseline'lar.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GraspSegments, collate, load_stats
from model import ControllerToHand
from mano_right import ManoRight
import grasp_taxonomy as tax
from evaluate import gather, mpjpe_frames, jerk, device

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_REPO, "checkpoints")
RESULTS = os.path.join(_REPO, "results")

VARIANTS = [
    ("baseline(noprev)", "c2h_noprev.pt"),
    ("A: fk_loss", "c2h_fk.pt"),
    ("C: obj_size", "c2h_objsize.pt"),
    ("A+C: fk+obj_size", "c2h_fk_objsize.pt"),
    ("B: AR+sched_samp", "c2h_ss.pt"),
]


def eval_ckpt(path, mano, dev, idx, mean, std):
    ck = torch.load(path, map_location=dev, weights_only=False)
    obj_size = ck["args"].get("obj_size", False)
    feat_dim = 16 if obj_size else 13
    use_prev = not ck["args"].get("no_prev", False)
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=use_prev, feat_dim=feat_dim).to(dev)
    model.load_state_dict(ck["model"])
    ds = GraspSegments("test", mean, std, obj_size=obj_size)
    G = gather(model, DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate), dev)
    e = mpjpe_frames(G["free"], G["betas"], G["wq"], G["wt"], G["joints"], idx, mano)
    cats = G["cat"][idx]
    bycat = {tax.CATEGORIES[c]: round(float(e[cats == c].mean()), 2)
             for c in range(3) if (cats == c).any()}
    # smoothness (free-running) - ilk 200 segment
    js = []
    for s in ds.segments[:200]:
        with torch.no_grad():
            fr = model.forward_free(torch.as_tensor(s["feat"])[None].to(dev),
                                    torch.as_tensor(s["cat"])[None].to(dev)).cpu().numpy()[0]
        js.append(jerk(fr))
    return dict(mpjpe=round(float(e.mean()), 2), per_cat=bycat,
                jerk=round(float(np.mean(js)), 4)), G


def main():
    dev = device()
    mano = ManoRight()
    mean, std = load_stats()
    # statik baseline'lar (train kategori-ortalama)
    tr = GraspSegments("train", mean, std)
    tp = np.concatenate([s["target"] for s in tr.segments])
    tc = np.concatenate([s["cat"] for s in tr.segments])
    cat_mean = {c: tp[tc == c].mean(0) for c in range(3)}

    # ortak idx (ilk varyantin N'i)
    ck0 = torch.load(os.path.join(CKPT, VARIANTS[0][1]), map_location=dev, weights_only=False)
    G0 = gather(ControllerToHand(hidden=ck0["args"]["hidden"], layers=ck0["args"]["layers"],
                use_prev=False, feat_dim=13).to(dev).eval().requires_grad_(False),
                DataLoader(GraspSegments("test", mean, std), batch_size=32, collate_fn=collate), dev) \
        if False else None
    # N: test frame sayisi
    dstmp = GraspSegments("test", mean, std)
    N = sum(len(s["feat"]) for s in dstmp.segments)
    np.random.seed(0); idx = np.random.choice(N, min(8000, N), replace=False)

    # statik baseline MPJPE (cat_mean) - bir varyanttan GT al
    table = {}
    GT = None
    for name, fn in VARIANTS:
        p = os.path.join(CKPT, fn)
        if not os.path.exists(p):
            print("ATLA (yok):", fn); continue
        res, G = eval_ckpt(p, mano, dev, idx, mean, std)
        table[name] = res
        GT = G
    # cat_mean baseline
    catmean_pred = np.stack([cat_mean[c] for c in GT["cat"]])
    e_cm = mpjpe_frames(catmean_pred, GT["betas"], GT["wq"], GT["wt"], GT["joints"], idx, mano)
    table["baseline: cat_mean"] = dict(mpjpe=round(float(e_cm.mean()), 2), per_cat={}, jerk=None)

    json.dump(table, open(os.path.join(RESULTS, "eval_compare.json"), "w"), indent=2)
    print(f"\n=== TEST free-running MPJPE (mm) [{len(idx)} frame] ===")
    order = sorted(table.items(), key=lambda kv: kv[1]["mpjpe"])
    for name, r in order:
        pc = " ".join(f"{k}={v}" for k, v in r["per_cat"].items())
        jk = f"jerk={r['jerk']}" if r["jerk"] is not None else ""
        print(f"  {name:22s} {r['mpjpe']:6.2f}   {pc}   {jk}")
    print("\nGT jerk ref ~0.167 | rapor -> results/eval_compare.json")


if __name__ == "__main__":
    main()
