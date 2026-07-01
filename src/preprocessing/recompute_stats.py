"""Faz 3: birlesik (HOT3D+DexYCB) TRAIN uzerinden 13-dim girdi stats + extents birlestir.

Onkosul: make_split.py calistirilmis (split.json guncel).
"""
import os
import json
import glob
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANON = os.path.join(_REPO, "data", "processed", "hot3d_canonical")


def merge_extents():
    base = json.load(open(os.path.join(CANON, "object_extents.json")))
    dxp = os.path.join(CANON, "object_extents_dexycb.json")
    if os.path.exists(dxp):
        base.update(json.load(open(dxp)))
    json.dump(base, open(os.path.join(CANON, "object_extents.json"), "w"), indent=2)
    print(f"extents birlesti -> {len(base)} nesne")


def recompute_stats():
    import csv
    manifest_path = os.path.join(CANON, "manifest.csv")
    train_ids = set()
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row.get("split") in ("train", "backbone_train"):
                train_ids.add(row["seq_id"])
    feats = []
    n_h = n_d = 0
    for npz in sorted(glob.glob(os.path.join(CANON, "seq_*.npz"))):
        sid = os.path.basename(npz)[4:-4]
        if sid not in train_ids:
            continue
        d = np.load(npz, allow_pickle=True)
        f = np.concatenate([d["rel_pos"], d["rel_rot6d"], d["rel_vel"], d["dist"]], axis=1)
        feats.append(f)
        if sid.startswith("DX"):
            n_d += len(f)
        else:
            n_h += len(f)
    F = np.concatenate(feats).astype(np.float64)
    mean = F.mean(0); std = F.std(0)
    std[std < 1e-6] = 1.0
    stats = json.load(open(os.path.join(CANON, "stats.json")))
    stats["input_mean"] = mean.tolist()
    stats["input_std"] = std.tolist()
    stats["train_frames_for_stats"] = int(len(F))
    stats["sources"] = dict(hot3d_train_frames=int(n_h), dexycb_train_frames=int(n_d))
    json.dump(stats, open(os.path.join(CANON, "stats.json"), "w"), indent=2)
    print(f"stats yeniden hesaplandi: train frames={len(F)} (HOT3D={n_h} DexYCB={n_d})")
    print("  mean[:3]=", np.round(mean[:3], 4), "std[:3]=", np.round(std[:3], 4))


if __name__ == "__main__":
    merge_extents()
    recompute_stats()
