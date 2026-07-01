"""OakInk static grasp dataset (docs/A-veri-hazirlama.md, B1 unified frame_feat interface).

Expected processed layout:
  data/processed/oakink_canonical/
    dataset.npz       pose(N,48), shape(N,10), tsl(N,3), category(N), obj_name(N)
    stats.json        wrist_mean/std, pts_mean/std
    split.json        train/val/test indices
    obj_pts/*.npy     per-object point clouds

OakInk is single-frame (T=1) and static: rel_vel is zero-filled since there is no
temporal window, and `dist` is the nearest fingertip-independent wrist-to-surface
distance computed from the sampled object point cloud (real geometry, not a stub).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import axis_angle_to_rot6d  # noqa: E402
from model.mano_fk import axis_angle_to_matrix as _aa_to_mat  # noqa: E402
from model.model_io import DEFAULT_N_POINTS, FINGER_POSE_DIM, HOT3D_FRAME_DIM, MANO_SHAPE_DIM  # noqa: E402


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → (3,) axis-angle via scipy (no torch dependency)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)
from preprocessing.quality_labels import compute_quality_label  # noqa: E402
from utils.paths import OAKINK_CANON  # noqa: E402


class OakInkStaticDataset(Dataset):
    """Static object-conditioned grasp samples from OakInk, exposed via frame_feat(1,13)."""

    def __init__(
        self,
        root: str | Path = OAKINK_CANON,
        split: str = "train",
        n_points: int = DEFAULT_N_POINTS,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split}")
        self.root = Path(root)
        self.split = split
        self.n_points = n_points
        self.augment = augment and split == "train"
        self.rng = np.random.default_rng(seed)

        data_path = self.root / "dataset.npz"
        split_path = self.root / "split.json"
        stats_path = self.root / "stats.json"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        if not stats_path.exists():
            raise FileNotFoundError(stats_path)

        data = np.load(data_path, allow_pickle=True)
        self.pose = data["pose"].astype(np.float32)
        self.shape = data["shape"].astype(np.float32)
        self.tsl = data["tsl"].astype(np.float32)
        # obj_anno: (N, 12) = R(9, row-major) + t(3) of object-to-world transform.
        # If absent (old NPZ), fall back to identity — triggers re-preprocessing warning.
        if "obj_anno" in data:
            self.obj_anno = data["obj_anno"].astype(np.float32)  # (N, 12)
        else:
            import warnings
            warnings.warn(
                "dataset.npz has no 'obj_anno' field. obj_pts_contact will be zero. "
                "Re-run build_oakink_canonical.py to fix contact/penetration loss.",
                stacklevel=2,
            )
            self.obj_anno = None
        # fingertips_world: (N, 5, 3) GT fingertip world positions from hand_j via cam_extr^{-1}.
        # Added in the updated build_oakink_canonical.py. Falls back to None for old NPZ.
        self.fingertips_world = (
            data["fingertips_world"].astype(np.float32) if "fingertips_world" in data else None
        )
        self.category = data["category"].astype(str)
        self.obj_name = data["obj_name"].astype(str)

        with split_path.open() as f:
            split_data = json.load(f)
        self.indices = np.asarray(split_data[split], dtype=np.int64)

        with stats_path.open() as f:
            raw_stats = json.load(f)
        self.stats = {}
        for key, value in raw_stats.items():
            try:
                self.stats[key] = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                # Keep stats.json free to carry metadata such as input_feature_order.
                continue
        self.obj_dir = self.root / "obj_pts"
        self._point_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(len(self.indices))

    def _load_points(self, obj_name: str) -> np.ndarray:
        if obj_name not in self._point_cache:
            path = self.obj_dir / f"{obj_name}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            self._point_cache[obj_name] = np.load(path).astype(np.float32)
        return self._point_cache[obj_name]

    def _sample_points(self, points: np.ndarray, item_index: int) -> np.ndarray:
        replace = len(points) < self.n_points
        if self.augment:
            idx = self.rng.choice(len(points), self.n_points, replace=replace)
            sampled = points[idx].copy()
            sampled += self.rng.normal(0.0, 0.003, size=sampled.shape).astype(np.float32)
            return sampled
        rng = np.random.default_rng(item_index)
        idx = rng.choice(len(points), self.n_points, replace=replace)
        return points[idx].astype(np.float32)

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        mean = self.stats.get("pts_mean", np.zeros(3, dtype=np.float32))
        std = self.stats.get("pts_std", np.ones(3, dtype=np.float32))
        return (points - mean) / np.maximum(std, 1e-6)

    @staticmethod
    def _obj_pts_to_wrist_frame(
        raw_points: np.ndarray,    # (N, 3) object-canonical
        obj_anno_12: np.ndarray,   # (12,) = R(9, row-major) + t(3) of object-to-world
        wrist_tsl: np.ndarray,     # (3,)  wrist world translation
        global_orient_aa: np.ndarray,  # (3,) wrist global orient axis-angle
    ) -> np.ndarray:
        """Transform obj_pts from object-canonical → wrist frame (metres, no normalisation).

        Pipeline:
          canonical → world:  pts_world = pts @ R_obj.T + t_obj
          world → wrist:      pts_wrist = (pts_world - wrist_t) @ R_wrist
        """
        R_obj = obj_anno_12[:9].reshape(3, 3)    # rotation cols: obj-local → world
        t_obj = obj_anno_12[9:12]                # object origin in world
        pts_world = raw_points @ R_obj.T + t_obj  # (N, 3)

        R_wrist = _aa_to_mat(
            torch.from_numpy(global_orient_aa.astype(np.float32))
        ).numpy()  # (3, 3)  wrist-local → world

        pts_wrist = (pts_world - wrist_tsl) @ R_wrist  # (N, 3) wrist frame
        return pts_wrist.astype(np.float32)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        pose = self.pose[idx]
        global_orient = pose[:3]
        finger_pose = pose[3:]
        obj_name = self.obj_name[idx]
        raw_points = self._sample_points(self._load_points(obj_name), item)
        points = self._normalize_points(raw_points)

        tsl = self.tsl[idx].astype(np.float32)

        # nearest_dist: wrist-to-object-surface distance in world frame.
        # raw_points are object-canonical; need world transform to compute correctly.
        # Approximate with canonical distance when obj_anno absent (legacy fallback).
        if self.obj_anno is not None:
            obj_anno_12 = self.obj_anno[idx]
            R_obj = obj_anno_12[:9].reshape(3, 3)
            t_obj = obj_anno_12[9:12]
            pts_world = raw_points @ R_obj.T + t_obj
            nearest_dist = float(np.linalg.norm(pts_world - tsl[None, :], axis=-1).min())
        else:
            nearest_dist = float(np.linalg.norm(raw_points - tsl[None, :], axis=-1).min())

        R_wrist = _aa_to_mat(
            torch.from_numpy(global_orient.astype(np.float32))
        ).numpy()  # (3,3) wrist-local → world

        if self.obj_anno is not None:
            # Object-relative frame_feat — same semantics as HOT3D:
            #   rel_pos   = R_obj.T @ (tsl - t_obj)  : wrist in object frame
            #   rel_rot6d = rot6d(R_obj.T @ R_wrist)  : wrist rotation in object frame
            rel_pos = R_obj.T @ (tsl - t_obj)                          # (3,)
            rel_rot6d = axis_angle_to_rot6d(
                torch.from_numpy(
                    _matrix_to_axis_angle(R_obj.T @ R_wrist).astype(np.float32)
                )
            ).numpy()                                                    # (6,)
        else:
            # Legacy fallback when obj_anno absent — world coords, different semantics.
            rel_pos = tsl
            rel_rot6d = axis_angle_to_rot6d(
                torch.from_numpy(global_orient.astype(np.float32))
            ).numpy()

        raw_feat = np.concatenate(
            [rel_pos, rel_rot6d, np.zeros(3, dtype=np.float32), np.array([nearest_dist], dtype=np.float32)]
        ).astype(np.float32)

        # Normalize with the same input_mean/input_std as HOT3D so distributions match.
        input_mean = np.asarray(self.stats.get("input_mean", []), dtype=np.float32)
        input_std  = np.asarray(self.stats.get("input_std",  []), dtype=np.float32)
        if input_mean.shape == (HOT3D_FRAME_DIM,) and input_std.shape == (HOT3D_FRAME_DIM,):
            raw_feat = (raw_feat - input_mean) / np.maximum(input_std, 1e-6)

        frame_feat = raw_feat[None, :]  # (1, 13): OakInk is single-frame (T=1)

        finger_pose_t = torch.from_numpy(finger_pose.astype(np.float32))
        obj_pts_t = torch.from_numpy(points.astype(np.float32))  # normalised canonical, for PointNet

        # obj_pts_contact: object surface in wrist frame (metres), used by contact/penetration loss.
        if self.obj_anno is not None:
            obj_pts_contact_np = self._obj_pts_to_wrist_frame(
                raw_points, self.obj_anno[idx], tsl, global_orient
            )
        else:
            obj_pts_contact_np = np.zeros_like(raw_points)  # sentinel; loss will be ~zero
        obj_pts_contact_t = torch.from_numpy(obj_pts_contact_np)

        # Use GT fingertip world positions (from hand_j) transformed to wrist frame,
        # bypassing the simplified mano_fk.py which has an 18-20cm convention error.
        fingertips_wrist = None
        if self.fingertips_world is not None and self.obj_anno is not None:
            tips_world = self.fingertips_world[idx]   # (5,3) world frame
            from model.mano_fk import axis_angle_to_matrix
            R_wrist = axis_angle_to_matrix(
                torch.from_numpy(global_orient.astype(np.float32))
            ).numpy()
            fingertips_wrist = torch.from_numpy(
                ((tips_world - tsl) @ R_wrist).astype(np.float32)
            )  # (5,3) wrist frame

        quality_label = compute_quality_label(
            finger_pose_t, obj_pts_contact_t, torch.tensor(nearest_dist, dtype=torch.float32),
            fingertips_wrist=fingertips_wrist,
        )

        return {
            "source": "oakink",
            "frame_feat": torch.from_numpy(frame_feat),
            "obj_pts": obj_pts_t,
            "obj_pts_contact": obj_pts_contact_t,
            "target_pose": finger_pose_t,
            "quality_label": quality_label,
            "shape": torch.from_numpy(self.shape[idx].astype(np.float32)),
            "tsl": torch.from_numpy(tsl),
            "pose_global": torch.from_numpy(global_orient.astype(np.float32)),
            "obj_name": obj_name,
            "category": self.category[idx],
        }


def collate_oakink(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "frame_feat",
        "obj_pts",
        "obj_pts_contact",
        "target_pose",
        "quality_label",
        "shape",
        "tsl",
        "pose_global",
    ]
    out: dict[str, Any] = {key: torch.stack([sample[key] for sample in batch]) for key in tensor_keys}
    out["source"] = [sample["source"] for sample in batch]
    out["obj_name"] = [sample["obj_name"] for sample in batch]
    out["category"] = [sample["category"] for sample in batch]
    return out


def assert_oakink_contract(batch: dict[str, Any]) -> None:
    if batch["frame_feat"].shape[-1] != HOT3D_FRAME_DIM:
        raise AssertionError(f"frame_feat must end with {HOT3D_FRAME_DIM}, got {batch['frame_feat'].shape}")
    if batch["frame_feat"].shape[-2] != 1:
        raise AssertionError(f"OakInk frame_feat must have T=1, got {batch['frame_feat'].shape}")
    if batch["obj_pts"].shape[-2:] != (DEFAULT_N_POINTS, 3):
        raise AssertionError(f"obj_pts must be (*,{DEFAULT_N_POINTS},3), got {batch['obj_pts'].shape}")
    if batch["target_pose"].shape[-1] != FINGER_POSE_DIM:
        raise AssertionError(f"target_pose must end with {FINGER_POSE_DIM}, got {batch['target_pose'].shape}")
    if batch["shape"].shape[-1] != MANO_SHAPE_DIM:
        raise AssertionError(f"shape must end with {MANO_SHAPE_DIM}, got {batch['shape'].shape}")
