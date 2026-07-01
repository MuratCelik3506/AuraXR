"""HOT3D temporal grasp dataset from processed canonical NPZ files (B1/B8)."""
from __future__ import annotations

import csv
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

from model.model_io import DEFAULT_N_POINTS, FINGER_POSE_DIM, HOT3D_FRAME_DIM  # noqa: E402
from preprocessing.quality_labels import compute_quality_label  # noqa: E402
from utils.paths import HOT3D_CANON, OAKINK_CANON  # noqa: E402


class Hot3DTemporalDataset(Dataset):
    """Sliding-window temporal samples.

    Uses HOT3D processed fields:
      frame_feat = rel_pos(3)+rel_rot6d(6)+rel_vel(3)+dist(1)
      target_pose = finger_aa45[t] or finger_aa45[t+target_offset]
      prev_pose/prev2_pose = finger_aa45[t-1]/[t-2], used by the B8
        GT-velocity/acceleration-matching loss terms.
    """

    def __init__(
        self,
        root: str | Path = HOT3D_CANON,
        split: str = "train",
        window: int = 16,
        target_offset: int = 0,
        stride: int = 1,
        n_points: int = DEFAULT_N_POINTS,
        obj_pts_root: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be train/val/test/all, got {split}")
        if window < 1:
            raise ValueError("window must be positive")
        self.root = Path(root)
        self.split = split
        self.window = window
        self.target_offset = target_offset
        self.stride = stride
        self.n_points = n_points
        # Look in HOT3D's own obj_pts first, fall back to OakInk's
        if obj_pts_root is not None:
            self.obj_pts_roots = [Path(obj_pts_root)]
        else:
            self.obj_pts_roots = [
                Path(root) / "obj_pts",
                OAKINK_CANON / "obj_pts",
            ]
        self.stats = self._load_stats()
        self.obj_split = self._load_obj_split()   # {obj_name: "train"|"val"|"test"}
        self.seq_files = self._load_seq_files()
        self.samples = self._index_windows()
        self._npz_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._pts_cache: dict[str, np.ndarray] = {}

    def _load_stats(self) -> dict[str, Any]:
        path = self.root / "stats.json"
        if not path.exists():
            return {}
        with path.open() as f:
            return json.load(f)

    def _load_obj_split(self) -> dict[str, str]:
        """obj_split.json'dan {obj_name: split} mapping'i yükle.

        Yoksa tüm objeler 'all' kabul edilir (geriye dönük uyumluluk).
        """
        path = self.root / "obj_split.json"
        if not path.exists():
            return {}
        with path.open() as f:
            data = json.load(f)
        mapping: dict[str, str] = {}
        for sp, objs in data.items():
            for obj in objs:
                mapping[obj] = sp
        return mapping

    def _load_seq_files(self) -> list[Path]:
        # obj_split varsa tüm sequence'ları yükle; frame-level filtre _index_windows'da.
        # Yoksa manifest'teki split alanını kullan (geriye dönük uyumluluk).
        manifest = self.root / "manifest.csv"
        if not manifest.exists():
            return sorted(self.root.glob("seq_*.npz"))
        files = []
        with manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                if not self.obj_split:
                    row_split = row.get("split", "all")
                    if self.split != "all" and row_split != self.split:
                        continue
                seq_id = row["seq_id"]
                path = self.root / f"seq_{seq_id}.npz"
                if path.exists():
                    files.append(path)
        return files

    def _index_windows(self) -> list[tuple[Path, int, int]]:
        samples: list[tuple[Path, int, int]] = []
        for path in self.seq_files:
            data = np.load(path, allow_pickle=True)
            n = int(len(data["finger_aa45"]))
            if n < self.window + max(0, self.target_offset):
                continue
            seg      = data["segment_id"] if "segment_id" in data.files else np.zeros(n, dtype=np.int64)
            obj_names = data["obj_name"] if "obj_name" in data.files else None
            for end in range(self.window - 1, n - max(0, self.target_offset), self.stride):
                start  = end - self.window + 1
                target = end + self.target_offset
                if seg[start] != seg[end] or seg[end] != seg[target]:
                    continue
                # Frame-level obj split: target frame'deki obj_name'e göre filtrele.
                if self.obj_split and self.split != "all" and obj_names is not None:
                    obj = str(obj_names[target])
                    frame_split = self.obj_split.get(obj, "train")
                    if frame_split != self.split:
                        continue
                samples.append((path, start, target))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npz(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._npz_cache:
            data = np.load(path, allow_pickle=True)
            self._npz_cache[path] = {key: data[key] for key in data.files}
        return self._npz_cache[path]

    def _frame_feat(self, data: dict[str, np.ndarray]) -> np.ndarray:
        feat = np.concatenate(
            [data["rel_pos"], data["rel_rot6d"], data["rel_vel"], data["dist"]],
            axis=-1,
        ).astype(np.float32)
        mean = np.asarray(self.stats.get("input_mean", []), dtype=np.float32)
        std = np.asarray(self.stats.get("input_std", []), dtype=np.float32)
        if mean.shape == (HOT3D_FRAME_DIM,) and std.shape == (HOT3D_FRAME_DIM,):
            feat = (feat - mean) / np.maximum(std, 1e-6)
        return feat

    def _normalize_points(self, pts: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.stats.get("pts_mean", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        std  = np.asarray(self.stats.get("pts_std",  np.ones(3,  dtype=np.float32)), dtype=np.float32)
        return (pts - mean) / np.maximum(std, 1e-6)

    def _load_object_points(self, obj_name: str) -> np.ndarray:
        if obj_name not in self._pts_cache:
            pts = None
            for root in self.obj_pts_roots:
                path = root / f"{obj_name}.npy"
                if path.exists():
                    pts = np.load(path).astype(np.float32)
                    break
            if pts is None:
                pts = np.zeros((self.n_points, 3), dtype=np.float32)
            self._pts_cache[obj_name] = pts
        pts = self._pts_cache[obj_name]
        replace = len(pts) < self.n_points
        rng = np.random.default_rng(abs(hash(obj_name)) % (2**32))
        idx = rng.choice(len(pts), self.n_points, replace=replace)
        return pts[idx].astype(np.float32)

    @staticmethod
    def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
        """(4,) wxyz quaternion -> (3, 3) rotation matrix."""
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        n = w*w + x*x + y*y + z*z
        if n < 1e-10:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n
        return np.array([
            [1 - s*(y*y + z*z),     s*(x*y - w*z),     s*(x*z + w*y)],
            [    s*(x*y + w*z), 1 - s*(x*x + z*z),     s*(y*z - w*x)],
            [    s*(x*z - w*y),     s*(y*z + w*x), 1 - s*(x*x + y*y)],
        ], dtype=np.float32)

    def _obj_pts_contact(
        self,
        obj_name: str,
        obj_world_t: np.ndarray,  # (3,)
        obj_world_q: np.ndarray,  # (4,) wxyz
        wrist_world_t: np.ndarray,  # (3,)
        wrist_world_q: np.ndarray,  # (4,) wxyz
    ) -> np.ndarray:
        """Object surface points in wrist frame (metres, no normalisation).

        Pipeline:
          canonical → world:  pts_world = pts @ R_obj.T + t_obj
          world → wrist:      pts_wrist = (pts_world - wrist_t) @ R_wrist
        """
        raw_pts = self._load_object_points(obj_name)  # (N, 3) canonical
        R_obj = self._quat_wxyz_to_matrix(obj_world_q)
        pts_world = raw_pts @ R_obj.T + obj_world_t   # (N, 3) world
        R_wrist = self._quat_wxyz_to_matrix(wrist_world_q)
        pts_wrist = (pts_world - wrist_world_t) @ R_wrist  # (N, 3) wrist
        return pts_wrist.astype(np.float32)

    def _shifted_frame_feat(self, feat: np.ndarray, start: int, shift: int) -> np.ndarray:
        """Window of frame features shifted `shift` frames back. Pads by repeating first row."""
        s = max(0, start - shift)
        e = max(self.window, start + self.window - shift)
        window = feat[s : s + self.window]
        if len(window) < self.window:
            pad = np.repeat(feat[:1], self.window - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        return window.astype(np.float32)

    def __getitem__(self, item: int) -> dict[str, Any]:
        path, start, target = self.samples[item]
        data = self._load_npz(path)
        end = start + self.window
        obj_name = str(data["obj_name"][target])
        feat_all = self._frame_feat(data)
        frame_feat = feat_all[start:end]
        # Shifted windows for velocity/acceleration loss in Phase 2
        prev_frame_feat  = self._shifted_frame_feat(feat_all, start, shift=1)
        prev2_frame_feat = self._shifted_frame_feat(feat_all, start, shift=2)
        finger_hist = data["finger_aa45"][start:end].astype(np.float32)
        contact = data["contact_flag"][start:end].astype(np.float32).reshape(self.window, 1)
        target_pose = data["finger_aa45"][target].astype(np.float32)
        prev_pose = data["finger_aa45"][max(target - 1, 0)].astype(np.float32)
        prev2_pose = data["finger_aa45"][max(target - 2, 0)].astype(np.float32)
        dist_target = float(data["dist"][target][0])

        raw_obj_pts = self._load_object_points(obj_name)
        obj_pts = torch.from_numpy(self._normalize_points(raw_obj_pts))  # normalized canonical, for PointNet
        target_pose_t = torch.from_numpy(target_pose)

        # obj_pts_contact: object surface in wrist frame (metres), for contact/penetration loss.
        # HOT3D NPZ stores wrist_world_t/q and obj_world_t/q per frame — all info is available.
        obj_pts_contact_np = self._obj_pts_contact(
            obj_name,
            obj_world_t=data["obj_world_t"][target].astype(np.float32),
            obj_world_q=data["obj_world_q"][target].astype(np.float32),
            wrist_world_t=data["wrist_world_t"][target].astype(np.float32),
            wrist_world_q=data["wrist_world_q"][target].astype(np.float32),
        )
        obj_pts_contact_t = torch.from_numpy(obj_pts_contact_np)

        # Use stored MANO FK joints (world frame) instead of simplified mano_fk.py to avoid
        # the 18-20cm FK convention mismatch. Transform to wrist frame with same @ R_wrist used
        # for obj_pts_contact so both are in the identical frame.
        fingertips_wrist = None
        if "fk_joints" in data:
            fk_world = data["fk_joints"][target].astype(np.float32)   # (16,3) world
            wt = data["wrist_world_t"][target].astype(np.float32)
            wq = data["wrist_world_q"][target].astype(np.float32)
            R_wrist = self._quat_wxyz_to_matrix(wq)
            fk_wrist = (fk_world - wt) @ R_wrist                      # (16,3) wrist frame
            tip_idx = [3, 6, 9, 12, 15]                               # last joint per 3-joint chain
            fingertips_wrist = torch.from_numpy(fk_wrist[tip_idx])    # (5,3)

        quality_label = compute_quality_label(
            target_pose_t, obj_pts_contact_t, torch.tensor(dist_target, dtype=torch.float32),
            fingertips_wrist=fingertips_wrist,
        )

        return {
            "source": "hot3d",
            "frame_feat": torch.from_numpy(frame_feat),
            "prev_frame_feat": torch.from_numpy(prev_frame_feat),
            "prev2_frame_feat": torch.from_numpy(prev2_frame_feat),
            "finger_hist": torch.from_numpy(finger_hist),
            "contact_flag": torch.from_numpy(contact),
            "target_pose": target_pose_t,
            "prev_pose": torch.from_numpy(prev_pose),
            "prev2_pose": torch.from_numpy(prev2_pose),
            "quality_label": quality_label,
            "obj_pts": obj_pts,
            "obj_pts_contact": obj_pts_contact_t,
            "obj_name": obj_name,
            "seq_path": str(path),
            "target_index": target,
        }


def collate_hot3d(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "frame_feat",
        "prev_frame_feat",
        "prev2_frame_feat",
        "finger_hist",
        "contact_flag",
        "target_pose",
        "prev_pose",
        "prev2_pose",
        "quality_label",
        "obj_pts",
        "obj_pts_contact",
    ]
    out: dict[str, Any] = {key: torch.stack([sample[key] for sample in batch]) for key in tensor_keys}
    out["source"] = [sample["source"] for sample in batch]
    out["obj_name"] = [sample["obj_name"] for sample in batch]
    out["seq_path"] = [sample["seq_path"] for sample in batch]
    out["target_index"] = torch.tensor([sample["target_index"] for sample in batch], dtype=torch.long)
    return out


def assert_hot3d_contract(batch: dict[str, Any]) -> None:
    if batch["frame_feat"].shape[-1] != HOT3D_FRAME_DIM:
        raise AssertionError(f"frame_feat must end with {HOT3D_FRAME_DIM}, got {batch['frame_feat'].shape}")
    if batch["finger_hist"].shape[-1] != FINGER_POSE_DIM:
        raise AssertionError(f"finger_hist must end with {FINGER_POSE_DIM}, got {batch['finger_hist'].shape}")
    if batch["target_pose"].shape[-1] != FINGER_POSE_DIM:
        raise AssertionError(f"target_pose must end with {FINGER_POSE_DIM}, got {batch['target_pose'].shape}")
    if batch["obj_pts"].shape[-2:] != (DEFAULT_N_POINTS, 3):
        raise AssertionError(f"obj_pts must be (*,{DEFAULT_N_POINTS},3), got {batch['obj_pts'].shape}")
