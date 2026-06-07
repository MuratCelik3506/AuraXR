"""hot3d_dataset.py — PyTorch Dataset that streams from the HDF5 built by build_dataset.py."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HOT3DDataset(Dataset):
    """Loads train or val split from dataset.h5.

    Feature vector (15 dims):
        [0:3]  dir_world     — unit vector from wrist to object, HOT3D world frame
        [3:6]  dir_obj_local — same vector rotated into object-local frame
        [6]    distance      — hand-object distance in metres
        [7]    approach_speed — dot(wrist_velocity, dir_world), positive = moving toward object
        [8:12] grip_oh       — grip category one-hot (Power/Precision/Palmar/Pinch)
        [12:15] bbox         — object bbox half-extents (x, y, z) in metres

    Target vector (22 dims):
        UmeTrack joint angles for one hand (left or right).

    Args:
        hdf5_path: Path to dataset.h5 produced by build_dataset.py.
        split:     "train" or "val".
        normalise: If True, applies z-score normalisation using stored stats.
    """

    def __init__(self, hdf5_path: Path, split: str, normalise: bool = True):
        self.hdf5_path = str(hdf5_path)
        self.split     = split
        self.normalise = normalise

        with h5py.File(self.hdf5_path, "r") as hf:
            self.N = hf[split]["features"].shape[0]
            meta   = json.loads(hf.attrs["meta"])

        if normalise:
            self.feat_mean = torch.tensor(meta["feature_mean"], dtype=torch.float32)
            self.feat_std  = torch.tensor(meta["feature_std"],  dtype=torch.float32)
            self.tgt_mean  = torch.tensor(meta["target_mean"],  dtype=torch.float32)
            self.tgt_std   = torch.tensor(meta["target_std"],   dtype=torch.float32)
        else:
            self.feat_mean = self.feat_std = self.tgt_mean = self.tgt_std = None

        # Cache in RAM if small enough (< 2 GB)
        with h5py.File(self.hdf5_path, "r") as hf:
            feat_bytes = hf[split]["features"].nbytes
        if feat_bytes < 2 * 1024 ** 3:
            with h5py.File(self.hdf5_path, "r") as hf:
                self._feat = torch.from_numpy(hf[split]["features"][:]).share_memory_()
                self._tgt  = torch.from_numpy(hf[split]["targets"][:]).share_memory_()
                self._dist = torch.from_numpy(hf[split]["distances"][:]).share_memory_()
            self._cached = True
        else:
            self._cached = False
            self._hf = None

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        if self._cached:
            feat = self._feat[idx].clone()
            tgt  = self._tgt[idx].clone()
            dist = self._dist[idx].clone()
        else:
            if self._hf is None:
                self._hf = h5py.File(self.hdf5_path, "r")
            feat = torch.from_numpy(self._hf[self.split]["features"][idx])
            tgt  = torch.from_numpy(self._hf[self.split]["targets"][idx])
            dist = torch.tensor(self._hf[self.split]["distances"][idx])

        if self.normalise:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)
            tgt  = (tgt  - self.tgt_mean)  / (self.tgt_std  + 1e-8)

        return feat, tgt, dist

    def get_norm_stats(self) -> dict:
        """Return normalisation stats as a dict (for denormalising predictions)."""
        with h5py.File(self.hdf5_path, "r") as hf:
            return json.loads(hf.attrs["meta"])
