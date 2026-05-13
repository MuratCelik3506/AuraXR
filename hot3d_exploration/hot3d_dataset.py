"""hot3d_dataset.py — Shared PyTorch Dataset for HOT3D training windows (HDF5)."""

import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class HOT3DDataset(Dataset):
    """
    Streams training windows from the HDF5 file produced by 09_build_dataset.py.
    Applies feature/target normalisation stored in the file's meta attribute.
    Caches the whole split in RAM if it fits in < 500 MB.

    Augmentation (train split only, augment=True):
      1. Controller position noise   — ±1 cm uniform offset per window (both hands)
      2. Beta perturbation           — ±0.5 Gaussian on MANO shape β (both hands)
      3. Left/right mirror flip      — 50% chance: swap hand-0 ↔ hand-1 slots in
                                       features AND targets (pure slot-swap, no
                                       coordinate reflection needed)

    Feature layout reference (96 dims, see 09_build_dataset.py):
      [0..2]   ctrl_pos_h0   [3..6]  ctrl_rot_h0   [7] grip_h0  [8] trig_h0
      [9..17]  same for h1
      [18..24] nearest_obj_h0 (centroid xyz + bbox xyz + category)
      [25..31] same for h1
      [32..95] visual embedding (zeros)

    Target layout (78 dims):
      [0..14]  mano_pose_h0 (15)   [15..24] mano_betas_h0 (10)
      [25..27] wrist_t_h0   (3)    [28..31] wrist_q_h0    (4)
      [32..34] delta_t_h0   (3)    [35..38] delta_q_h0    (4)   → 39 dims / hand
      [39..77] same for h1
    """

    # Feature index ranges used by augmentation
    _CTRL_POS_H0  = slice(0,  3)    # left  controller xyz
    _CTRL_POS_H1  = slice(9,  12)   # right controller xyz
    _CTRL_SLOT_H0 = slice(0,  9)    # full left  controller slot
    _CTRL_SLOT_H1 = slice(9,  18)   # full right controller slot
    _OBJ_SLOT_H0  = slice(18, 25)   # nearest-object slot (left)
    _OBJ_SLOT_H1  = slice(25, 32)   # nearest-object slot (right)

    _BETA_H0 = slice(15, 25)        # MANO β hand-0 in target vector
    _BETA_H1 = slice(54, 64)        # MANO β hand-1 in target vector
    _HAND_H0 = slice(0,  39)        # full hand-0 block in target
    _HAND_H1 = slice(39, 78)        # full hand-1 block in target

    def __init__(self, hf_path: Path, split: str,
                 normalise: bool = True, augment: bool = False,
                 cache_gb: float = 12.0):
        self.hf_path   = str(hf_path)
        self.split     = split
        self.normalise = normalise
        self.augment   = augment

        with h5py.File(self.hf_path, "r") as hf:
            self.N      = hf[split]["features"].shape[0]
            feat_bytes  = hf[split]["features"].nbytes
            meta        = json.loads(hf.attrs["meta"])

        if normalise:
            self.feat_mean = torch.tensor(meta["feature_mean"], dtype=torch.float32)
            self.feat_std  = torch.tensor(meta["feature_std"],  dtype=torch.float32)
            self.tgt_mean  = torch.tensor(meta["target_mean"],  dtype=torch.float32)
            self.tgt_std   = torch.tensor(meta["target_std"],   dtype=torch.float32)
        else:
            self.feat_mean = self.feat_std = self.tgt_mean = self.tgt_std = None

        if feat_bytes < cache_gb * 1024 ** 3:
            # Load entire split into RAM — multiple workers can read tensors safely
            print(f"  Caching {split} split into RAM ({feat_bytes/1e9:.1f} GB)…", flush=True)
            with h5py.File(self.hf_path, "r") as hf:
                self._feat = torch.from_numpy(hf[split]["features"][:]).share_memory_()
                self._tgt  = torch.from_numpy(hf[split]["targets"][:]).share_memory_()
            self._cached = True
            print(f"  {split} cached.", flush=True)
        else:
            # Too large to cache — each worker opens its own h5py handle lazily
            # (do NOT open h5py here; file handles cannot be shared across processes)
            self._cached = False
            self._hf     = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self._cached:
            feat = self._feat[idx].clone()
            tgt  = self._tgt[idx].clone()
        else:
            # Lazy open: each DataLoader worker process opens its own handle on
            # first access, so file handles are never shared across processes.
            if self._hf is None:
                self._hf = h5py.File(self.hf_path, "r")
            feat = torch.from_numpy(self._hf[self.split]["features"][idx])
            tgt  = torch.from_numpy(self._hf[self.split]["targets"][idx])

        # Augmentation runs on raw (un-normalised) data so noise is in real units
        if self.augment:
            feat, tgt = self._augment(feat, tgt)

        if self.normalise:
            feat = (feat - self.feat_mean) / self.feat_std
            tgt  = (tgt  - self.tgt_mean)  / self.tgt_std

        return feat, tgt

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _augment(self, feat: torch.Tensor, tgt: torch.Tensor):
        """
        In-place augmentation on raw (un-normalised) tensors.
        feat : [T=16, 96]   tgt : [78]
        """

        # 1. Controller position noise — one uniform offset per window so the
        #    whole trajectory shifts without adding per-frame jitter.
        feat[:, self._CTRL_POS_H0] = feat[:, self._CTRL_POS_H0] + (torch.rand(3) - 0.5) * 0.02  # ±1 cm
        feat[:, self._CTRL_POS_H1] = feat[:, self._CTRL_POS_H1] + (torch.rand(3) - 0.5) * 0.02

        # 2. Beta perturbation — shape parameters are individual scalars with
        #    HOT3D std ≈ 1, so ±0.5 is a moderate perturbation.
        tgt[self._BETA_H0] = tgt[self._BETA_H0] + torch.randn(10) * 0.5
        tgt[self._BETA_H1] = tgt[self._BETA_H1] + torch.randn(10) * 0.5

        # 3. Mirror flip — swap hand-0 ↔ hand-1 slots with probability 0.5.
        #    Pure slot swap (no coordinate reflection) treats each hand's
        #    gesture as interchangeable, effectively doubling the dataset.
        if torch.rand(1).item() < 0.5:
            h0_feat = feat[:, self._CTRL_SLOT_H0].clone()
            feat[:, self._CTRL_SLOT_H0] = feat[:, self._CTRL_SLOT_H1]
            feat[:, self._CTRL_SLOT_H1] = h0_feat

            h0_obj = feat[:, self._OBJ_SLOT_H0].clone()
            feat[:, self._OBJ_SLOT_H0] = feat[:, self._OBJ_SLOT_H1]
            feat[:, self._OBJ_SLOT_H1] = h0_obj

            h0_tgt = tgt[self._HAND_H0].clone()
            tgt[self._HAND_H0] = tgt[self._HAND_H1]
            tgt[self._HAND_H1] = h0_tgt

        return feat, tgt
