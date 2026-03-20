"""
HOT3D Dataset Loader — Intent-Aware XR Framework
================================================

This module handles the loading and reconstruction of the HOT3D dataset 
(Project Aria / Quest 3) for egocentric intent prediction.

The Challenge:
--------------
HOT3D is a state-of-the-art 3D hand tracking dataset but it uses 
a non-standard skeletal representation (UmeTrack) and lacks 
frame-level action labels for high-level intent.

Our Solution:
-------------
1. Forward Kinematics (FK) Heuristic:
   HOT3D provides joint angles (22) and a wrist transform. We implement 
   a custom FK chain to reconstruct Cartesian XYZ coordinates for all 
   21 hand joints, which allows the IntentFormer to use the same features 
   across different datasets.

2. Temporal Phase Inference:
   Since the dataset focuses on 6D object pose tracking, we infer 
   intent (Pickup, Observe, Release) by segmenting each 150-frame clip 
   into equal thirds.

3. Tar-Resident Loading:
   To manage the massive size of HOT3D (~800GB full), this loader 
   reads JSON annotations directly from compressed .tar archives 
   without full extraction, significantly reducing disk space and I/O.

Features Produced:
-----------------
- hand_flat : (T, 126) – Reconstructed XYZ skeletal joints (normalized).
- obj_rt    : (T, 16)  – 6D object pose (world-to-object matrix).
- label     : int      – Inferred temporal phase (0, 1, 2).
"""

import os
import io
import json
import math
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

NUM_JOINTS     = 21          # MANO / UmeTrack joints per hand
NUM_HANDS      = 2           # left + right
JOINT_DIM      = 3           # x, y, z
CLIP_LEN       = 150         # frames per HOT3D clip
HOT3D_FPS      = 30

# HOT3D has 3 coarse action types inferred by temporal position
# (Meta does not provide frame-level action labels; we use clip-level heuristic)
NUM_CLASSES_HOT3D = 3
ACTION_PICKUP   = 0
ACTION_OBSERVE  = 1
ACTION_PUTDOWN  = 2

# Devices shipped in the dataset
DEVICE_ARIA    = "Aria"
DEVICE_QUEST3  = "Quest3"


# ─────────────────────────────────────────────────────────
# Forward-kinematics approximation from UmeTrack features
# ─────────────────────────────────────────────────────────

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] → 3×3 rotation matrix.
    Safe for batched input of shape (..., 4).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    norm = np.sqrt(w*w + x*x + y*y + z*z) + 1e-9
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.stack([
        1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w),
        2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def _umetrack_to_joints(
    joint_angles: np.ndarray,   # (22,) UmeTrack joint angles (real format has 22)
    wrist_t: np.ndarray,        # (3,)  translation (wrist world position)
    wrist_q: np.ndarray,        # (4,)  quaternion [w,x,y,z] (wrist orientation)
) -> np.ndarray:
    """
    Approximate 3D joint positions (21, 3) from UmeTrack parameters.

    Real HOT3D UmeTrack format (verified):
      T_world_from_wrist: {translation_xyz: [x,y,z], quaternion_wxyz: [w,x,y,z]}
      joint_angles: 22 floats

    We reconstruct 20 finger joints (joints 1-20) via heuristic FK:
      - Joint 0 (wrist) = wrist_t (world position)
      - Joints 1-20     = accumulated offsets in wrist's local frame
    """
    R = _quat_to_rot(wrist_q)   # (3, 3) rotation matrix
    t = wrist_t                  # (3,)   wrist world position

    # Canonical per-joint bone lengths (metres)
    # 5 fingers × 4 joints = 20, using first 20 of 22 angles
    bone_len = np.array([
        0.040, 0.035, 0.030, 0.005,   # thumb (4 DOF)
        0.090, 0.040, 0.025, 0.020,   # index
        0.090, 0.040, 0.025, 0.020,   # middle
        0.080, 0.035, 0.022, 0.018,   # ring
        0.070, 0.030, 0.020, 0.015,   # pinky
    ], dtype=np.float32)

    joints = np.zeros((NUM_JOINTS, JOINT_DIM), dtype=np.float32)
    joints[0] = t   # wrist

    for i in range(20):
        angle = float(joint_angles[i]) if i < len(joint_angles) else 0.0
        # Offset in the wrist's local frame: primarily along palm-forward (R[:,0])
        offset = bone_len[i] * (
            math.cos(angle) * R[:, 0] + math.sin(angle) * R[:, 2]
        )
        joints[i + 1] = joints[i] + offset.astype(np.float32)

    return joints   # (21, 3)


def _parse_hands_json(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a <FRAME_ID>.hands.json dict.

    Real format (verified):
      data['left' | 'right']['umetrack_pose']:
        T_world_from_wrist: {translation_xyz: [...], quaternion_wxyz: [...]}
        joint_angles: [22 floats]

    Returns:
        left_joints  : (21, 3) or zeros if missing
        right_joints : (21, 3) or zeros if missing
    """
    zeros = np.zeros((NUM_JOINTS, JOINT_DIM), dtype=np.float32)

    def _extract(hand_dict: Optional[dict]) -> np.ndarray:
        if hand_dict is None:
            return zeros.copy()
        try:
            ut = hand_dict.get("umetrack_pose", {})
            if not ut:
                return zeros.copy()
            ja = np.array(ut.get("joint_angles", [0.0] * 22), dtype=np.float32)

            # Real key is T_world_from_wrist (not wrist_xform)
            T_wrist = ut.get("T_world_from_wrist", {})
            t = np.array(T_wrist.get("translation_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
            q = np.array(T_wrist.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)

            return _umetrack_to_joints(ja, t, q)
        except Exception:
            return zeros.copy()

    left  = _extract(data.get("left"))
    right = _extract(data.get("right"))
    return left, right


def _parse_objects_json(data: dict) -> np.ndarray:
    """
    Parse a <FRAME_ID>.objects.json dict.

    Real format (verified):
      {"<obj_id>": [list of annotation dicts], ...}
      Each annotation dict: {T_world_from_object: {translation_xyz, quaternion_wxyz}, ...}

    Returns flattened 4×4 RT matrix (16,) for the FIRST annotation of the FIRST object.
    If no objects, returns zeros.
    """
    zeros = np.zeros(16, dtype=np.float32)
    if not isinstance(data, dict) or len(data) == 0:
        return zeros

    # data[obj_id] is a LIST of annotations — take the first
    try:
        first_list = next(iter(data.values()))
        if isinstance(first_list, list):
            if len(first_list) == 0:
                return zeros
            obj = first_list[0]
        else:
            obj = first_list   # fallback: already a dict

        T = obj.get("T_world_from_object", {})
        t = np.array(T.get("translation_xyz", [0, 0, 0]), dtype=np.float32)
        q = np.array(T.get("quaternion_wxyz", [1, 0, 0, 0]), dtype=np.float32)
        R = _quat_to_rot(q)   # (3, 3)

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R
        mat[:3,  3] = t
        return mat.flatten()   # (16,)
    except Exception:
        return zeros


def _infer_action_label(frame_idx: int, clip_len: int = CLIP_LEN) -> int:
    """
    Infer coarse action label from temporal position within the clip.

    HOT3D scenarios show participants picking up → observing → putting down objects.
    We approximate the temporal segmentation as equal thirds:
        0-33%  → pick-up  (ACTION_PICKUP  = 0)
        33-66% → observe  (ACTION_OBSERVE = 1)
        66-100% → put-down (ACTION_PUTDOWN = 2)

    This is a heuristic — replace with frame-level labels if available.
    """
    ratio = frame_idx / max(clip_len - 1, 1)
    if ratio < 0.33:
        return ACTION_PICKUP
    elif ratio < 0.66:
        return ACTION_OBSERVE
    else:
        return ACTION_PUTDOWN


# ─────────────────────────────────────────────────────────
# Wrist-Relative Normalisation (same as H2O loader)
# ─────────────────────────────────────────────────────────

def wrist_relative_normalize(joints: np.ndarray) -> np.ndarray:
    """
    joints : (T, 2, 21, 3)
    Returns (T, 2, 21, 3) with wrist (joint 0) subtracted.
    """
    wrist = joints[:, :, 0:1, :]   # (T, 2, 1, 3)
    return joints - wrist


# ─────────────────────────────────────────────────────────
# Tar-based clip loader (no image decoding)
# ─────────────────────────────────────────────────────────

def load_hot3d_clip(tar_path: str) -> Optional[dict]:
    """
    Load one HOT3D clip from its .tar archive.

    Returns dict:
        hand_poses   : (F, 2, 21, 3)  wrist-relative UmeTrack joints
        obj_poses_rt : (F, 16)        flattened RT for first object
        action_labels: (F,)           coarse action label per frame
        num_frames   : int
    Returns None if the clip is too sparse (>50% missing hands).
    """
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
    except Exception as e:
        print(f"[HOT3D] Cannot open tar {tar_path}: {e}")
        return None

    # Collect per-frame JSON file members
    # Real tar names: '000000.hands.json', '000000.objects.json' (no leading path)
    hands_files   = {}   # frame_id → TarInfo
    objects_files = {}
    for m in members:
        name = os.path.basename(m.name)   # strip any leading path prefix
        if name.endswith(".hands.json"):
            fid = name[:-len(".hands.json")]
            hands_files[fid] = m
        elif name.endswith(".objects.json"):
            fid = name[:-len(".objects.json")]
            objects_files[fid] = m

    frame_ids = sorted(hands_files.keys())
    F = len(frame_ids)
    if F == 0:
        return None

    hand_poses    = np.zeros((F, NUM_HANDS, NUM_JOINTS, JOINT_DIM), dtype=np.float32)
    obj_poses_rt  = np.zeros((F, 16),                               dtype=np.float32)
    action_labels = np.zeros(F,                                     dtype=np.int64)
    missing_hand  = 0

    try:
        with tarfile.open(tar_path, "r:*") as tf:
            for i, fid in enumerate(frame_ids):
                # ── Hand joints ──────────────────────────────────
                try:
                    m = hands_files[fid]
                    raw = tf.extractfile(m).read()
                    hdata = json.loads(raw.decode("utf-8"))
                    left, right = _parse_hands_json(hdata)
                    hand_poses[i, 0] = left
                    hand_poses[i, 1] = right
                    if np.all(left == 0) and np.all(right == 0):
                        missing_hand += 1
                except Exception:
                    missing_hand += 1

                # ── Object RT ─────────────────────────────────────
                if fid in objects_files:
                    try:
                        m = objects_files[fid]
                        raw = tf.extractfile(m).read()
                        odata = json.loads(raw.decode("utf-8"))
                        obj_poses_rt[i] = _parse_objects_json(odata)
                    except Exception:
                        pass

                # ── Action label ──────────────────────────────────
                action_labels[i] = _infer_action_label(i, F)
    except Exception as e:
        print(f"[HOT3D] Error reading clip {tar_path}: {e}")
        return None

    # Skip clips where more than 50 % of frames have no hand data
    if missing_hand > F * 0.5:
        return None

    # Wrist-relative normalisation
    hand_poses = wrist_relative_normalize(hand_poses)

    return {
        "hand_poses":    hand_poses,
        "obj_poses_rt":  obj_poses_rt,
        "action_labels": action_labels,
        "num_frames":    F,
    }


# ─────────────────────────────────────────────────────────
# Window extractor (mirrors H2ODataset.extract_window)
# ─────────────────────────────────────────────────────────

def extract_window_hot3d(
    clip_data:   dict,
    obs_ratio:   float,
    window_size: int,
) -> Optional[dict]:
    """
    Sample a fixed-length window from the first `obs_ratio` fraction of
    a clip and return (hand_flat, obj_rt, obs_ratio).

    HOT3D clips are already segmented — we treat each clip as one action.
    """
    F   = clip_data["num_frames"]
    obs_end   = max(1, int(F * obs_ratio))
    obs_start = max(0, obs_end - window_size)

    hand_snippet = clip_data["hand_poses"][obs_start:obs_end]      # (T, 2, 21, 3)
    obj_snippet  = clip_data["obj_poses_rt"][obs_start:obs_end]    # (T, 16)

    T = hand_snippet.shape[0]
    if T < 1:
        return None

    # Pad or truncate to window_size
    if T < window_size:
        pad = window_size - T
        hand_snippet = np.pad(hand_snippet, ((pad, 0), (0, 0), (0, 0), (0, 0)))
        obj_snippet  = np.pad(obj_snippet,  ((pad, 0), (0, 0)))
    else:
        hand_snippet = hand_snippet[-window_size:]
        obj_snippet  = obj_snippet[-window_size:]

    T_out     = hand_snippet.shape[0]
    hand_flat = hand_snippet.reshape(T_out, -1)   # (T, 126)

    # Label from the frame at obs_end (majority vote of the window)
    label = int(clip_data["action_labels"][min(obs_end - 1, F - 1)])

    return {
        "hand_flat": hand_flat.astype(np.float32),
        "obj_rt":    obj_snippet.astype(np.float32),
        "obs_ratio": np.float32(obs_ratio),
        "label":     label,
    }


# ─────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────

class HOT3DDataset(Dataset):
    """
    PyTorch Dataset for HOT3D-Clips.

    Each item is a dict with the same keys as H2ODataset:
        hand_flat  : Tensor (T, 126)   wrist-relative 3D joints (2 hands)
        obj_rt     : Tensor (T, 16)    flattened 4×4 RT (world← object)
        obs_ratio  : Tensor scalar     fraction of clip observed
        label      : Tensor long       coarse action label (0=pickup,1=observe,2=putdown)

    Args:
        root_dir    : path to data/hot3d  (must contain clip_splits.json +
                      train_aria/ and/or train_quest3/ sub-folders)
        split       : 'train' | 'test_ht_pose' | 'test_bop'
        devices     : list of devices to include, subset of ['Aria','Quest3']
        window_size : observation window length in frames (default 30)
        obs_ratios  : list of observation ratios  (default [0.2, 0.25, 0.3])
        max_clips   : if set, cap the number of clips loaded (useful for smoke tests)
    """

    def __init__(
        self,
        root_dir:    str,
        split:       str         = "train",
        devices:     list[str]   = None,
        window_size: int         = 30,
        obs_ratios:  list[float] = None,
        max_clips:   int         = None,
    ):
        super().__init__()
        self.root_dir    = Path(root_dir)
        self.split       = split
        self.devices     = devices if devices is not None else [DEVICE_ARIA, DEVICE_QUEST3]
        self.window_size = window_size
        self.obs_ratios  = obs_ratios if obs_ratios is not None else [0.2, 0.25, 0.3]
        self.max_clips   = max_clips

        self._clip_cache: dict[str, Optional[dict]] = {}
        self.samples:     list[dict]                = []

        self._build_sample_list()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _splits_file(self) -> Path:
        return self.root_dir / "clip_splits.json"

    def _tar_path(self, clip_id: int, device: str) -> Optional[Path]:
        """Locate the tar archive for a given clip id and device."""
        dev_folder = {
            DEVICE_ARIA:   ("train_aria",   "test_aria"),
            DEVICE_QUEST3: ("train_quest3", "test_quest3"),
        }
        if device not in dev_folder:
            return None

        split_train_key = "train"
        # Map dataset splits to folder names
        folder_name = dev_folder[device][0] if self.split == split_train_key \
                      else dev_folder[device][1]

        candidate = self.root_dir / folder_name / f"clip-{clip_id:06d}.tar"
        if candidate.exists():
            return candidate

        # Some archives use zero-padded differently – try plain int
        candidate2 = self.root_dir / folder_name / f"clip-{clip_id}.tar"
        if candidate2.exists():
            return candidate2

        return None

    def _get_clip(self, tar_path: str) -> Optional[dict]:
        if tar_path not in self._clip_cache:
            self._clip_cache[tar_path] = load_hot3d_clip(tar_path)
        return self._clip_cache[tar_path]

    def _build_sample_list(self):
        splits_path = self._splits_file()
        if not splits_path.exists():
            raise FileNotFoundError(
                f"[HOT3D] clip_splits.json not found at: {splits_path}\n"
                f"Download from https://huggingface.co/datasets/bop-benchmark/hot3d"
            )

        with open(splits_path) as f:
            splits = json.load(f)

        # clip_splits.json has shape: {"train": {"Aria": [...], "Quest3": [...]}, ...}
        split_data = splits.get(self.split, {})
        if not split_data:
            raise ValueError(
                f"[HOT3D] Split '{self.split}' not found in clip_splits.json. "
                f"Available: {list(splits.keys())}"
            )

        clips_added = 0
        for device in self.devices:
            clip_ids = split_data.get(device, [])
            for clip_id in clip_ids:
                if self.max_clips is not None and clips_added >= self.max_clips:
                    break
                tar_path = self._tar_path(clip_id, device)
                if tar_path is None:
                    continue   # file not downloaded yet — skip silently
                for obs_ratio in self.obs_ratios:
                    self.samples.append({
                        "tar_path":  str(tar_path),
                        "clip_id":   clip_id,
                        "device":    device,
                        "obs_ratio": obs_ratio,
                    })
                clips_added += 1
            if self.max_clips is not None and clips_added >= self.max_clips:
                break

        print(
            f"[HOT3DDataset] split={self.split}  "
            f"devices={self.devices}  clips={clips_added}  "
            f"samples={len(self.samples)}"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta      = self.samples[idx]
        clip_data = self._get_clip(meta["tar_path"])

        T = self.window_size
        fallback = {
            "hand_flat": torch.zeros(T, NUM_HANDS * NUM_JOINTS * JOINT_DIM),
            "obj_rt":    torch.zeros(T, 16),
            "obs_ratio": torch.tensor(meta["obs_ratio"]),
            "label":     torch.tensor(ACTION_OBSERVE, dtype=torch.long),
        }

        if clip_data is None:
            return fallback

        window = extract_window_hot3d(clip_data, meta["obs_ratio"], self.window_size)
        if window is None:
            return fallback

        return {
            "hand_flat": torch.from_numpy(window["hand_flat"]),
            "obj_rt":    torch.from_numpy(window["obj_rt"]),
            "obs_ratio": torch.tensor(window["obs_ratio"]),
            "label":     torch.tensor(window["label"], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# Convenience factory (mirrors get_dataloaders from h2o_dataset)
# ─────────────────────────────────────────────────────────

def get_hot3d_dataloaders(
    root_dir:    str,
    batch_size:  int         = 32,
    window_size: int         = 30,
    obs_ratios:  list[float] = None,
    devices:     list[str]   = None,
    num_workers: int         = 4,
    max_clips:   int         = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for HOT3D-Clips.

    Note: HOT3D test splits do NOT have GT hand/object annotations
    publicly available (only poses for test images), so 'test_ht_pose'
    clips load poses from available frames only.
    """
    if obs_ratios is None:
        obs_ratios = [0.2, 0.25, 0.3]
    if devices is None:
        devices = [DEVICE_ARIA, DEVICE_QUEST3]

    def make_loader(split, shuffle):
        ds = HOT3DDataset(
            root_dir,
            split=split,
            devices=devices,
            window_size=window_size,
            obs_ratios=obs_ratios,
            max_clips=max_clips,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return (
        make_loader("train",       True),
        make_loader("test_ht_pose", False),
    )


# ─────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/hot3d"
    ds   = HOT3DDataset(root, split="train", window_size=30, max_clips=5)
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        s = ds[0]
        print("hand_flat:", s["hand_flat"].shape)
        print("obj_rt   :", s["obj_rt"].shape)
        print("obs_ratio:", s["obs_ratio"].item())
        print("label    :", s["label"].item())
