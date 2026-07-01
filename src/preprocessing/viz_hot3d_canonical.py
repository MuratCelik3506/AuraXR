"""Textual per-frame summary of a HOT3D canonical seq_*.npz file."""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    n = len(data["finger_aa45"])
    print(f"file={args.npz} frames={n}")
    print("fields:", ", ".join(data.files))
    for i in range(min(args.limit, n)):
        obj = data["obj_name"][i] if "obj_name" in data.files else "?"
        dist = float(data["dist"][i][0])
        contact = bool(data["contact_flag"][i]) if "contact_flag" in data.files else False
        seg = int(data["segment_id"][i]) if "segment_id" in data.files else -1
        print(f"{i:04d} seg={seg:03d} obj={obj} dist={dist:.3f} contact={int(contact)}")


if __name__ == "__main__":
    main()
