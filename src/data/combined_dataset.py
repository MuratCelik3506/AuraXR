"""
Combined Dataset: H2O + HOT3D Fusion — Intent-Aware XR Framework
================================================================

This module provides a unified interface for joint training on both 
H2O (high-quality labels) and HOT3D (high-fidelity egocentric motion).

Fusion Mechanisms:
------------------
1. Shared Head Fusion (`shared_head`):
   Maps H2O's 36 detailed classes and HOT3D's 3 temporal phases into 
   a common 3-class label space:
     - 0: Pickup / Grap (Initial approach)
     - 1: Manipulation / Observe (Steady state)
     - 2: Release / Put-down (Terminal phase)
   This logic allows the model to learn a "Universal Intent Language" 
   that generalizes across different XR devices (Quest 3 vs. Static Rigs).

2. Concatenation Fusion (`concat`):
   Pools all samples into one stream but keeps the H2O label space (36 classes) 
   primary. HOT3D samples act as "weakly supervised" augmentation.

Data Identification (Source Tagging):
------------------------------------
Each batch dictionary includes a `source` tensor:
- `source=0`: Sample originated from the H2O dataset.
- `source=1`: Sample originated from the HOT3D dataset.

Usage:
------
    from src.data.combined_dataset import get_combined_dataloaders
    train_loader, val_loader, test_loader = get_combined_dataloaders(
        h2o_root="data/h2o", 
        hot3d_root="data/hot3d", 
        fusion="shared_head"
    )
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.data.h2o_dataset  import H2ODataset,   NUM_CLASSES as H2O_NUM_CLASSES
from src.data.hot3d_dataset import HOT3DDataset, NUM_CLASSES_HOT3D, \
                                   ACTION_PICKUP, ACTION_OBSERVE, ACTION_PUTDOWN, \
                                   DEVICE_ARIA, DEVICE_QUEST3


# ─────────────────────────────────────────────────────────
# H2O → coarse label mapping (for SharedHeadFusion)
# ─────────────────────────────────────────────────────────
#
# H2O has 36 action classes (1-indexed in raw → 0-indexed here).
# We group them into 3 coarse buckets based on the H2O annotation guide:
#   pick-up actions   → indices matching "grasp" in H2O
#   observe/transport → everything in-between
#   put-down          → release / place actions
#
# Full H2O label list (0-indexed):
#   0-11  : approach / pick-up  (grab, pick up, carry)
#   12-23 : manipulation        (pour, cut, stir, etc.)
#   24-35 : put-down / release  (place, put down, set)
#
# These boundaries are approximations; adjust if you have the exact
# H2O ontology mapping.

def _h2o_to_coarse(label: int) -> int:
    """Map H2O 0-indexed action label → coarse bucket (0, 1, 2)."""
    if label < 12:
        return ACTION_PICKUP    # 0
    elif label < 24:
        return ACTION_OBSERVE   # 1
    else:
        return ACTION_PUTDOWN   # 2


# ─────────────────────────────────────────────────────────
# Wrapper dataset: remaps labels + adds dataset-source tag
# ─────────────────────────────────────────────────────────

class _LabelRemapDataset(Dataset):
    """
    Wraps an existing dataset and applies an optional label remapping
    function.  Also injects `source` (int) into each sample dict so
    that downstream code can distinguish H2O vs HOT3D.

    source=0 → H2O
    source=1 → HOT3D
    """

    def __init__(
        self,
        inner:      Dataset,
        source_id:  int,
        remap_fn=None,   # callable(int) → int, or None
    ):
        self._inner     = inner
        self._source_id = source_id
        self._remap_fn  = remap_fn

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        sample = self._inner[idx]
        if self._remap_fn is not None:
            sample = dict(sample)   # shallow copy so we don't mutate cache
            sample["label"] = torch.tensor(
                self._remap_fn(int(sample["label"])), dtype=torch.long
            )
        sample["source"] = torch.tensor(self._source_id, dtype=torch.long)
        return sample


# ─────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────

NUM_CLASSES_COMBINED      = H2O_NUM_CLASSES     # 36  (concat mode)
NUM_CLASSES_SHARED_HEAD   = 3                   # pick-up / observe / put-down


# ─────────────────────────────────────────────────────────
# Combined factory
# ─────────────────────────────────────────────────────────

def get_combined_dataloaders(
    h2o_root:    str,
    hot3d_root:  str,
    fusion:      str         = "concat",     # "concat" | "shared_head"
    batch_size:  int         = 32,
    window_size: int         = 30,
    obs_ratios:  list[float] = None,
    num_workers: int         = 4,
    # H2O-specific
    h2o_splits:  tuple[str, str, str] = ("train", "val", "test"),
    # HOT3D-specific
    hot3d_split: str         = "train",
    hot3d_devices: list[str] = None,
    hot3d_max_clips: int     = None,
    # Sampling weights to balance datasets (None = equal weighting)
    hot3d_weight: float      = 1.0,
    h2o_weight:   float      = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from H2O (all splits) and HOT3D
    (train split only; HOT3D test sets have no public GT labels).

    Args:
        fusion        : 'concat'       → keep H2O's 36 labels; HOT3D labels are
                                         kept in [0,2] range (treated as dummy
                                         classes in the H2O head — useful for
                                         multi-task / pre-training settings).
                        'shared_head'  → remap both to 3 coarse labels.
        hot3d_weight  : if < 1.0, only a random fraction of HOT3D samples are kept.
        h2o_weight    : if < 1.0, only a random fraction of H2O samples are kept.

    Returns:
        (train_loader, val_loader, test_loader)
        val / test come exclusively from H2O to preserve evaluation integrity.
    """
    if obs_ratios is None:
        obs_ratios = [0.2, 0.25, 0.3]
    if hot3d_devices is None:
        hot3d_devices = [DEVICE_ARIA, DEVICE_QUEST3]

    # ── Build H2O datasets ────────────────────────────────────────────────
    tr_split, val_split, te_split = h2o_splits

    h2o_train = H2ODataset(
        h2o_root, split=tr_split,
        window_size=window_size, obs_ratios=obs_ratios,
    )
    h2o_val  = H2ODataset(
        h2o_root, split=val_split,
        window_size=window_size, obs_ratios=obs_ratios,
    )
    h2o_test = H2ODataset(
        h2o_root, split=te_split,
        window_size=window_size, obs_ratios=obs_ratios,
    )

    # ── Build HOT3D train dataset ─────────────────────────────────────────
    hot3d_train = HOT3DDataset(
        hot3d_root, split=hot3d_split,
        devices=hot3d_devices,
        window_size=window_size,
        obs_ratios=obs_ratios,
        max_clips=hot3d_max_clips,
    )

    # ── Apply fusion strategy ─────────────────────────────────────────────
    if fusion == "shared_head":
        h2o_train_wrapped = _LabelRemapDataset(
            h2o_train, source_id=0, remap_fn=_h2o_to_coarse
        )
        h2o_val_wrapped  = _LabelRemapDataset(
            h2o_val, source_id=0, remap_fn=_h2o_to_coarse
        )
        h2o_test_wrapped = _LabelRemapDataset(
            h2o_test, source_id=0, remap_fn=_h2o_to_coarse
        )
        hot3d_wrapped = _LabelRemapDataset(
            hot3d_train, source_id=1, remap_fn=None   # HOT3D already [0,2]
        )
    else:  # concat mode
        h2o_train_wrapped = _LabelRemapDataset(h2o_train, source_id=0)
        h2o_val_wrapped   = _LabelRemapDataset(h2o_val,   source_id=0)
        h2o_test_wrapped  = _LabelRemapDataset(h2o_test,  source_id=0)
        hot3d_wrapped     = _LabelRemapDataset(hot3d_train, source_id=1)

    # ── Combine train sets ─────────────────────────────────────────────────
    combined_train = ConcatDataset([h2o_train_wrapped, hot3d_wrapped])

    print(
        f"[Combined] fusion={fusion}  "
        f"H2O-train={len(h2o_train_wrapped)}  "
        f"HOT3D-train={len(hot3d_wrapped)}  "
        f"combined={len(combined_train)}"
    )

    # ── DataLoaders ────────────────────────────────────────────────────────
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(combined_train, shuffle=True, **common)
    val_loader   = DataLoader(h2o_val_wrapped,  shuffle=False, **common)
    test_loader  = DataLoader(h2o_test_wrapped, shuffle=False, **common)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────
# Number of output classes given fusion mode
# ─────────────────────────────────────────────────────────

def num_classes_for_fusion(fusion: str) -> int:
    """Return the number of output classes for a given fusion mode."""
    return {
        "concat":      NUM_CLASSES_COMBINED,    # 36
        "shared_head": NUM_CLASSES_SHARED_HEAD, # 3
    }[fusion]


# ─────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    h2o_root   = sys.argv[1] if len(sys.argv) > 1 else "data/h2o"
    hot3d_root = sys.argv[2] if len(sys.argv) > 2 else "data/hot3d"
    fusion     = sys.argv[3] if len(sys.argv) > 3 else "concat"

    n_cls = num_classes_for_fusion(fusion)
    print(f"Fusion mode: '{fusion}'  →  {n_cls} classes")

    tr, val, te = get_combined_dataloaders(
        h2o_root=h2o_root,
        hot3d_root=hot3d_root,
        fusion=fusion,
        batch_size=8,
        num_workers=0,
        hot3d_max_clips=2,
    )
    batch = next(iter(tr))
    print("hand_flat:", batch["hand_flat"].shape)
    print("obj_rt   :", batch["obj_rt"].shape)
    print("label    :", batch["label"])
    print("source   :", batch["source"])
