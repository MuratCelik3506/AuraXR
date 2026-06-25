"""Subject-held-out train/val/test split (Faz 2A).

Yeni-kisiye genelleme testi: val/test subject'leri train'de gorulmez.
7 subject, hepsinde 3 kategori var. -> data/processed/split.json
"""
import os
import csv
import json
import collections

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(_REPO, "data", "processed", "hot3d_canonical")

# Sabit atama (tekrar-uretilebilirlik). Hepsinde hook/power/wide var.
SUBJECTS = {
    "train": ["P0002", "P0003", "P0010", "P0013"],
    "val":   ["P0018"],
    "test":  ["P0017", "P0021"],
}


def main():
    rows = list(csv.DictReader(open(os.path.join(CANON, "manifest.csv"))))
    subj2split = {s: sp for sp, subs in SUBJECTS.items() for s in subs}
    seqs = collections.defaultdict(list)
    frames = collections.Counter()
    for r in rows:
        s = r["seq_id"].split("_")[0]
        sp = subj2split.get(s)
        if sp is None:
            continue
        seqs[sp].append(r["seq_id"])
        frames[sp] += int(r["n_frames"])

    out = dict(method="subject-held-out", note="val/test subjects train'de yok (yeni-kisi genelleme)",
               subjects=SUBJECTS, sequences={k: sorted(v) for k, v in seqs.items()},
               frame_counts=dict(frames))
    p = os.path.join(CANON, "split.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("yazildi ->", p)
    for sp in ["train", "val", "test"]:
        print(f"  {sp:5s}: {len(seqs[sp]):3d} seq  {frames[sp]:7d} frame  subjects={SUBJECTS[sp]}")


if __name__ == "__main__":
    main()
