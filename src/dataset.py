"""Segment-bazli torch Dataset (Faz 2A).

Her ornek = bir reach->grasp segmenti (degisken uzunluk). collate pad + mask.
Girdi 13-dim [rel_pos3, rel_rot6d6, rel_vel3, dist1] stats.json ile normalize.
Hedef finger_pca15. Eval/FK icin ek alanlar (fk_joints, betas, wrist/obj poz) tasinir.
"""
import os
import json
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(_REPO, "data", "processed", "hot3d_canonical")

INPUT_DIM = 13
POSE_DIM = 15


def load_stats():
    s = json.load(open(os.path.join(CANON, "stats.json")))
    mean = np.asarray(s["input_mean"], dtype=np.float32)
    std = np.asarray(s["input_std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _seq_ids(split):
    sp = json.load(open(os.path.join(CANON, "split.json")))
    return set(sp["sequences"][split])


def load_extents():
    p = os.path.join(CANON, "object_extents.json")
    ext = json.load(open(p))
    arr = np.asarray(list(ext.values()), dtype=np.float32)
    emean, estd = arr.mean(0), arr.std(0)
    estd[estd < 1e-6] = 1.0
    return ext, emean.astype(np.float32), estd.astype(np.float32)


class GraspSegments(Dataset):
    def __init__(self, split, mean=None, std=None, obj_size=False):
        self.split = split
        if mean is None:
            mean, std = load_stats()
        self.mean, self.std = mean, std
        self.obj_size = obj_size
        if obj_size:
            self.ext, self.emean, self.estd = load_extents()
        want = _seq_ids(split)
        self.segments = []
        for npz in sorted(glob.glob(os.path.join(CANON, "seq_*.npz"))):
            sid = os.path.basename(npz)[4:-4]
            if sid not in want:
                continue
            d = np.load(npz, allow_pickle=True)
            seg = d["segment_id"]
            feat = np.concatenate([d["rel_pos"], d["rel_rot6d"], d["rel_vel"], d["dist"]], axis=1).astype(np.float32)
            feat = (feat - self.mean) / self.std
            names = d["obj_name"].astype(str)
            for s in np.unique(seg):
                m = seg == s
                if m.sum() < 2:
                    continue
                fseg = feat[m]
                if self.obj_size:
                    nm = names[m][0]
                    e = (np.asarray(self.ext[nm], np.float32) - self.emean) / self.estd
                    fseg = np.concatenate([fseg, np.tile(e, (len(fseg), 1))], axis=1)
                self.segments.append(dict(
                    feat=fseg,
                    cat=d["category_id"][m].astype(np.int64),
                    target=d["finger_pca15"][m].astype(np.float32),
                    aa45=d["finger_aa45"][m].astype(np.float32),
                    fk_joints=d["fk_joints"][m].astype(np.float32),
                    betas=d["betas"][m].astype(np.float32),
                    wrist_t=d["wrist_world_t"][m].astype(np.float32),
                    wrist_q=d["wrist_world_q"][m].astype(np.float32),
                    obj_t=d["obj_world_t"][m].astype(np.float32),
                    obj_q=d["obj_world_q"][m].astype(np.float32),
                    contact=d["contact_flag"][m].astype(np.float32),
                    seq_id=sid, segment_id=int(s),
                    obj_name=str(d["obj_name"][m][0]),
                    obj_uid=str(d["obj_uid"][m][0]),
                ))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        return self.segments[i]


_PAD_KEYS = ["feat", "cat", "target", "aa45", "fk_joints", "betas",
             "wrist_t", "wrist_q", "obj_t", "obj_q", "contact"]


def collate(batch):
    L = max(len(b["feat"]) for b in batch)
    B = len(batch)
    out = {}
    lengths = torch.tensor([len(b["feat"]) for b in batch], dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.float32)
    for k in _PAD_KEYS:
        sample = batch[0][k]
        shp = (B, L) + sample.shape[1:]
        dt = torch.long if sample.dtype == np.int64 else torch.float32
        t = torch.zeros(shp, dtype=dt)
        for i, b in enumerate(batch):
            n = len(b[k])
            t[i, :n] = torch.as_tensor(b[k])
        out[k] = t
    for i, b in enumerate(batch):
        mask[i, :len(b["feat"])] = 1.0
    out["mask"] = mask
    out["lengths"] = lengths
    out["meta"] = [(b["seq_id"], b["segment_id"], b["obj_name"]) for b in batch]
    return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    for split in ["train", "val", "test"]:
        ds = GraspSegments(split)
        n = sum(len(s["feat"]) for s in ds.segments)
        print(f"{split}: {len(ds)} segment, {n} frame")
    dl = DataLoader(GraspSegments("val"), batch_size=8, collate_fn=collate, shuffle=True)
    b = next(iter(dl))
    print("batch feat", b["feat"].shape, "target", b["target"].shape,
          "cat", b["cat"].shape, "mask", b["mask"].shape)
