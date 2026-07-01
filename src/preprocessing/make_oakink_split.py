"""Category-stratified train/val/test split for OakInk canonical data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.paths import OAKINK_CANON  # noqa: E402


def make_split(categories: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for category in sorted(set(categories.astype(str))):
        idx = np.where(categories.astype(str) == category)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        n_val = max(1, int(len(idx) * val_ratio))
        test.extend(idx[:n_test].tolist())
        val.extend(idx[n_test:n_test + n_val].tolist())
        train.extend(idx[n_test + n_val:].tolist())
    return {"train": train, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OAKINK_CANON)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = np.load(args.root / "dataset.npz", allow_pickle=True)
    split = make_split(data["category"], args.val, args.test, args.seed)
    out = args.out or args.root / "split.json"
    out.write_text(json.dumps(split, indent=2))
    print(f"wrote {out}: { {k: len(v) for k, v in split.items()} }")


if __name__ == "__main__":
    main()
