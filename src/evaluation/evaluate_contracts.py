"""Dataset contract smoke-tests for both OakInk and HOT3D loaders."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.dataset_hot3d import Hot3DTemporalDataset, assert_hot3d_contract, collate_hot3d  # noqa: E402
from data.dataset_oakink import OakInkStaticDataset, assert_oakink_contract, collate_oakink  # noqa: E402
from model.model_io import DEFAULT_N_POINTS, FINGER_POSE_DIM, HOT3D_FRAME_DIM  # noqa: E402


def check_oakink(n_points: int = DEFAULT_N_POINTS, batch_size: int = 8) -> None:
    ds = OakInkStaticDataset(split="val", n_points=n_points, augment=False)
    if len(ds) == 0:
        print("[OakInk] SKIP (no val samples)")
        return
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_oakink)
    batch = next(iter(loader))
    assert_oakink_contract(batch)
    assert batch["frame_feat"].shape == (batch_size, 1, HOT3D_FRAME_DIM), batch["frame_feat"].shape
    assert batch["target_pose"].shape == (batch_size, FINGER_POSE_DIM), batch["target_pose"].shape
    assert "quality_label" in batch, "missing quality_label"
    print(f"[OakInk] OK  n={len(ds)}  batch frame_feat={batch['frame_feat'].shape}  quality_label={batch['quality_label'].shape}")


def check_hot3d(window: int = 8, n_points: int = DEFAULT_N_POINTS, batch_size: int = 4) -> None:
    ds = Hot3DTemporalDataset(split="all", window=window, n_points=n_points)
    if len(ds) == 0:
        print("[HOT3D] SKIP (no samples)")
        return
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_hot3d)
    batch = next(iter(loader))
    assert_hot3d_contract(batch)
    assert batch["frame_feat"].shape == (batch_size, window, HOT3D_FRAME_DIM), batch["frame_feat"].shape
    assert batch["target_pose"].shape == (batch_size, FINGER_POSE_DIM), batch["target_pose"].shape
    assert "prev_pose" in batch, "missing prev_pose (needed for B8 velocity loss)"
    assert "prev2_pose" in batch, "missing prev2_pose (needed for B8 acceleration loss)"
    assert "quality_label" in batch, "missing quality_label"
    print(f"[HOT3D] OK  n={len(ds)}  batch frame_feat={batch['frame_feat'].shape}  prev_pose={batch['prev_pose'].shape}")


if __name__ == "__main__":
    check_oakink()
    check_hot3d()
