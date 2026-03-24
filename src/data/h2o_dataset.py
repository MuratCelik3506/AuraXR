"""
H2O Dataset Loader — Intent-Aware XR Framework
==============================================

This module provides a robust loader for the H2O (Human-to-Object)
dataset, focusing on 3D skeletal and object pose data for XR
applications. It avoids RGB frames to minimize I/O overhead on 
Apple M2 Max/ANE hardware.

Core Logic:
-----------
1. Window-based Loading:
   Takes long action segments (e.g., a "coffee pouring" event) and 
   extracts early observation windows (e.g., the first 20%-30% 
   of the motion) to train the model for real-time proactive intent.

2. Wrist-Relative Normalization:
   Subtracts the wrist (root) coordinate from all other 20 joints. 
   This makes the trajectory independent of the user's global 
   position in the room, focusing entirely on hand morphology 
   and relative object pose.

3. Action Prediction:
   Maps H2O's 36 detailed action classes (0-indexed internally).

Data Layout Expected:
--------------------
  annotations/Subject{N}/{RIG}/{ID}/cam4/
    hand_pose/   <- 2 * 21 * 3 floats (vis, x, y, z) per hand
    obj_pose_rt/ <- flattened 4x4 matrix per frame
    action_label/<- single int

Features Produced:
-----------------
- hand_flat : (T, 126) – 2 hands × 21 joints × 3 coords, normalized.
- obj_rt    : (T, 16)  – 4×4 rigid transform relative to camera.
- obs_ratio : float    – Observation ratio used (0.2, 0.25, etc.).
- label     : int      – Ground truth action class.
"""

import os
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────
# Constants (derived from inspecting the raw files)
# ─────────────────────────────────────────────────────────
NUM_JOINTS       = 21          # joints per hand (MANO topology)
NUM_HANDS        = 2           # left + right
JOINT_DIM        = 3          # x, y, z in camera space
TOKEN_PER_HAND   = 1 + NUM_JOINTS * JOINT_DIM   # visibility flag + 63 floats
HAND_POSE_LEN    = NUM_HANDS * TOKEN_PER_HAND    # 128 total floats per frame

# Camera rig names used in annotations
CAMERA_RIG       = "cam4"

# Scene (take) folders per subject
SCENES           = ["h1", "h2", "k1", "k2", "o1", "o2"]

# Number of action classes in H2O
NUM_CLASSES      = 36          # labels 1..36 → index 0..35

# ─────────────────────────────────────────────────────────
# Low-level parsers
# ─────────────────────────────────────────────────────────

def _parse_hand_pose(path: str) -> np.ndarray:
    """
    Parse a hand_pose/*.txt file.
    Each file is a single line with (2 × (1 + 21×3)) = 128 floats.
    Returns ndarray of shape (2, 21, 3) after stripping visibility flags.
    Positions are in camera-space metres.
    """
    vals = np.fromstring(open(path).read().strip(), dtype=np.float32, sep=" ")
    if vals.size != HAND_POSE_LEN:
        # If file is malformed or empty, return zeros
        return np.zeros((NUM_HANDS, NUM_JOINTS, JOINT_DIM), dtype=np.float32)

    out = np.zeros((NUM_HANDS, NUM_JOINTS, JOINT_DIM), dtype=np.float32)
    for h in range(NUM_HANDS):
        offset = h * TOKEN_PER_HAND
        # First value is visibility flag, skip it
        joints_flat = vals[offset + 1 : offset + 1 + NUM_JOINTS * JOINT_DIM]
        out[h] = joints_flat.reshape(NUM_JOINTS, JOINT_DIM)
    return out


def _parse_obj_pose_rt(path: str) -> np.ndarray:
    """
    Parse an obj_pose_rt/*.txt file.
    Format: obj_id  R(9 floats)  t(3 floats)  rest…
    We read the first 17 values: obj_id + 16-element flattened 4×4 matrix.
    Returns ndarray of shape (16,).
    """
    try:
        vals = np.fromstring(open(path).read().strip(), dtype=np.float32, sep=" ")
        if vals.size < 17:
            return np.zeros(16, dtype=np.float32)
        return vals[1:17]          # skip obj_id, take 4×4 RT matrix
    except Exception:
        return np.zeros(16, dtype=np.float32)


def _parse_action_label(path: str) -> int:
    try:
        return int(open(path).read().strip())
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────
# Wrist-Relative Normalization
# ─────────────────────────────────────────────────────────

def wrist_relative_normalize(joints: np.ndarray) -> np.ndarray:
    """
    Make joint positions independent of global hand position.

    Args:
        joints: (T, 2, 21, 3)  raw camera-space coordinates
    Returns:
        normalized: (T, 2, 21, 3)  positions relative to wrist (joint 0)
    """
    wrist = joints[:, :, 0:1, :]        # (T, 2, 1, 3) – wrist is joint 0
    return joints - wrist                # broadcast subtract


# ─────────────────────────────────────────────────────────
# Sequence reader
# ─────────────────────────────────────────────────────────

def load_sequence(seq_dir: str, camera: str = CAMERA_RIG) -> dict:
    """
    Load all frames for one take (e.g. 'subject1/h1/0') from the annotation directory.

    Returns a dict:
        hand_poses  : (F, 2, 21, 3)
        obj_poses_rt: (F, 16)
        action_labels: (F,)   int
    """
    cam_dir  = os.path.join(seq_dir, camera)
    hp_dir   = os.path.join(cam_dir, "hand_pose")
    op_dir   = os.path.join(cam_dir, "obj_pose_rt")
    al_dir   = os.path.join(cam_dir, "action_label")

    # Improved Adaptive Loader (Phase 1 fix): 
    # Try 4-digit (0000) and 6-digit (000000) patterns separately
    hp_files_4 = sorted(glob.glob(os.path.join(hp_dir, "[0-9][0-9][0-9][0-9].txt")))
    hp_files_6 = sorted(glob.glob(os.path.join(hp_dir, "[0-9][0-9][0-9][0-9][0-9][0-9].txt")))
    
    # Decide which set to use based on which one has more files (avoids pollution)
    if len(hp_files_6) >= len(hp_files_4):
        hp_files = hp_files_6
    else:
        hp_files = hp_files_4

    F = len(hp_files)
    if F == 0:
        return None

    hand_poses   = np.zeros((F, NUM_HANDS, NUM_JOINTS, JOINT_DIM), dtype=np.float32)
    obj_poses_rt = np.zeros((F, 16),                                dtype=np.float32)
    action_labels = np.zeros(F,                                     dtype=np.int64)

    for i, hp_path in enumerate(hp_files):
        frame_name = os.path.basename(hp_path)

        hand_poses[i]    = _parse_hand_pose(hp_path)
        op_path = os.path.join(op_dir, frame_name)
        if os.path.exists(op_path):
            obj_poses_rt[i] = _parse_obj_pose_rt(op_path)
        al_path = os.path.join(al_dir, frame_name)
        if os.path.exists(al_path):
            action_labels[i] = _parse_action_label(al_path)

    raw_hand_poses = hand_poses.copy()
    # Apply wrist-relative normalization (Section 2 of instruction.md)
    hand_poses = wrist_relative_normalize(hand_poses)

    return {
        "hand_poses":    hand_poses,
        "raw_hand_poses": raw_hand_poses,
        "obj_poses_rt":  obj_poses_rt,
        "action_labels": action_labels,
        "num_frames":    F,
    }


# ─────────────────────────────────────────────────────────
# Window extractor (early prediction, 20-30 % of motion)
# ─────────────────────────────────────────────────────────

def extract_window(
    seq_data:    dict,
    start_act:   int,
    end_act:     int,
    obs_ratio:   float,
    window_size: int,
) -> dict | None:
    """
    Slice a fixed-length observation window from the first `obs_ratio` portion
    of an action segment [start_act, end_act].

    If the action segment is too short, return None.
    """
    action_len   = end_act - start_act + 1
    obs_end      = start_act + max(1, int(action_len * obs_ratio))
    obs_end      = min(obs_end, end_act + 1)
    obs_start    = max(obs_end - window_size, start_act)

    if obs_end - obs_start < 1:
        return None

    hand_snippet = seq_data["hand_poses"][obs_start:obs_end]
    obj_snippet  = seq_data["obj_poses_rt"][obs_start:obs_end]

    T = hand_snippet.shape[0]
    # Pad or truncate to exactly window_size
    if T < window_size:
        pad = window_size - T
        hand_snippet = np.pad(hand_snippet, ((pad, 0), (0,0), (0,0), (0,0)))
        obj_snippet  = np.pad(obj_snippet,  ((pad, 0), (0,0)))
    else:
        hand_snippet = hand_snippet[-window_size:]
        obj_snippet  = obj_snippet[-window_size:]

    # Flatten both hands to (T, 2*21*3) = (T, 126)
    T_out = hand_snippet.shape[0]
    hand_flat = hand_snippet.reshape(T_out, -1)           # (T, 126)

    # Next pose for auxiliary task (regression)
    # Target frame is obs_end (one frame after the window) if it exists
    if obs_end < seq_data["num_frames"]:
        target_pose = seq_data["hand_poses"][obs_end].reshape(-1)
    else:
        target_pose = seq_data["hand_poses"][obs_end-1].reshape(-1)

    return {
        "hand_flat":   hand_flat.astype(np.float32),       # (T, 126)
        "obj_rt":      obj_snippet.astype(np.float32),     # (T, 16)
        "target_pose": target_pose.astype(np.float32),     # (126,)
        "obs_ratio":   np.float32(obs_ratio),
    }


# ─────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────

class H2ODataset(Dataset):
    """
    PyTorch Dataset for H2O early action prediction.

    Each item is a dict:
        hand_flat  : Tensor (T, 126)  wrist-relative joint positions (2 hands)
        obj_rt     : Tensor (T, 16)   flattened 4×4 RT matrix of the target object
        obs_ratio  : Tensor scalar    fraction of action observed
        label      : Tensor long      0-indexed action class (0..35)

    Args:
        root_dir   : path to `data/h2o` (the parent of `annotations/` and `models/`)
        split      : 'train' | 'val' | 'test'
        window_size: number of frames in the observation window (default 30)
        obs_ratios : list of observation ratios to try per action segment
        camera     : camera rig to read (default 'cam4')

    Notes:
        • No RGB images are loaded.  Only skeletal (.txt) files.
        • Wrist-relative normalization is applied automatically.
        • Sequences are cached in memory after first load.
    """

    def __init__(
        self,
        root_dir:    str,
        split:       str = "train",
        window_size: int = 30,
        obs_ratios:  list[float] | None = None,
        camera:      str = CAMERA_RIG,
        dense:       bool = False,  # If True, sample every N frames of action
        stride:      int = 5,       # Stride for dense sampling
    ):
        super().__init__()
        self.root_dir    = Path(root_dir)
        self.split       = split
        self.window_size = window_size
        self.obs_ratios  = obs_ratios if obs_ratios is not None else [0.2, 0.25, 0.3]
        self.camera      = camera
        self.dense       = dense
        self.stride      = stride

        self._seq_cache: dict[str, dict] = {}
        self.samples: list[dict]         = []

        self._build_sample_list()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_file(self) -> Path:
        fname = {"train": "action_train.txt",
                 "val":   "action_val.txt",
                 "test":  "action_test.txt"}[self.split]
        return self.root_dir / "models" / "label_split" / fname

    def _anno_dir(self, rel_path: str) -> Path:
        return self.root_dir / "annotations" / rel_path

    def _get_sequence(self, rel_path: str) -> dict | None:
        if rel_path not in self._seq_cache:
            seq_dir = str(self._anno_dir(rel_path))
            data    = load_sequence(seq_dir, self.camera)
            self._seq_cache[rel_path] = data
        return self._seq_cache[rel_path]

    def _build_sample_list(self):
        split_file = self._split_file()
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            lines = f.readlines()

        # header: id path action_label start_act end_act start_frame end_frame
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            _, path, action_label, start_act, end_act, _, _ = parts
            start_act    = int(start_act)
            end_act      = int(end_act)
            action_label = int(action_label) - 1   # → 0-indexed

            # Standard Split: specific ratios
            if not self.dense:
                for obs_ratio in self.obs_ratios:
                    self.samples.append({
                        "rel_path":     path,
                        "start_act":    start_act,
                        "end_act":      end_act,
                        "label":        action_label,
                        "obs_ratio":    obs_ratio,
                    })
            else:
                # Dense Sampling: sample windows throughout the action starting from 20%
                action_len = end_act - start_act + 1
                start_frame = start_act + int(action_len * 0.2)
                for f in range(start_frame, end_act + 1, self.stride):
                    ratio = (f - start_act + 1) / action_len
                    self.samples.append({
                        "rel_path":     path,
                        "start_act":    start_act,
                        "end_act":      end_act,
                        "label":        action_label,
                        "obs_ratio":    ratio,
                    })

        print(f"[H2ODataset] split={self.split}  segments={len(self.samples)} (dense={self.dense})")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta      = self.samples[idx]
        seq_data  = self._get_sequence(meta["rel_path"])

        if seq_data is None:
            # Return a zeroed fallback rather than crashing
            T  = self.window_size
            return {
                "hand_flat":   torch.zeros(T, NUM_HANDS * NUM_JOINTS * JOINT_DIM),
                "obj_rt":      torch.zeros(T, 16),
                "target_pose": torch.zeros(NUM_HANDS * NUM_JOINTS * JOINT_DIM),
                "obs_ratio":   torch.tensor(meta["obs_ratio"]),
                "label":       torch.tensor(meta["label"], dtype=torch.long),
            }

        window = extract_window(
            seq_data,
            meta["start_act"],
            meta["end_act"],
            meta["obs_ratio"],
            self.window_size,
        )

        if window is None:
            T = self.window_size
            return {
                "hand_flat":   torch.zeros(T, NUM_HANDS * NUM_JOINTS * JOINT_DIM),
                "obj_rt":      torch.zeros(T, 16),
                "target_pose": torch.zeros(NUM_HANDS * NUM_JOINTS * JOINT_DIM),
                "obs_ratio":   torch.tensor(meta["obs_ratio"]),
                "label":       torch.tensor(meta["label"], dtype=torch.long),
            }

        return {
            "hand_flat":   torch.from_numpy(window["hand_flat"]),
            "obj_rt":      torch.from_numpy(window["obj_rt"]),
            "target_pose": torch.from_numpy(window["target_pose"]),
            "obs_ratio":   torch.tensor(window["obs_ratio"]),
            "label":       torch.tensor(meta["label"], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────

def get_dataloaders(
    root_dir: str,
    batch_size:  int   = 32,
    window_size: int   = 30,
    obs_ratios: list[float] | None = None,
    num_workers: int   = 4,
    dense:       bool  = False,
    stride:      int   = 5,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    if obs_ratios is None:
        obs_ratios = [0.2, 0.25, 0.3]

    def make_loader(split, shuffle):
        ds = H2ODataset(root_dir, split=split,
                        window_size=window_size, obs_ratios=obs_ratios,
                        dense=dense, stride=stride)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=(num_workers > 0))

    return (
        make_loader("train", True),
        make_loader("val",   False),
        make_loader("test",  False),
    )


# ─────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/h2o"
    ds   = H2ODataset(root, split="train", window_size=30)
    sample = ds[0]
    print("hand_flat :", sample["hand_flat"].shape)
    print("obj_rt    :", sample["obj_rt"].shape)
    print("obs_ratio :", sample["obs_ratio"].item())
    print("label     :", sample["label"].item())
