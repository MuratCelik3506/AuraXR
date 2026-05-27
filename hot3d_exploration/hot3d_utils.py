"""hot3d_utils.py — Low-level HOT3D data-access helpers.

Reads Quest3 ZIP files directly — no additional SDK required.
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

# Hand key in umetrack_hand_pose_trajectory.jsonl
HAND_KEY = {"left": "0", "right": "1"}


# ---------------------------------------------------------------------------
# Quaternion helpers  (q stored as [w, x, y, z])
# ---------------------------------------------------------------------------

def quat_conjugate(q_wxyz: np.ndarray) -> np.ndarray:
    """Conjugate of a unit quaternion [w, x, y, z] → [w, -x, -y, -z]."""
    c = q_wxyz.copy()
    c[1:] *= -1
    return c


def rotate_vec(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by unit quaternion q (w, x, y, z).

    Uses the formula: v' = q ⊗ [0, v] ⊗ q*
    """
    w, x, y, z = q_wxyz.astype(np.float64)
    vx, vy, vz = v.astype(np.float64)
    # t = 2 * cross(q_xyz, v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# ZIP readers
# ---------------------------------------------------------------------------

def read_umetrack_trajectory(hand_zip_path: Path) -> dict[int, dict]:
    """Parse umetrack_hand_pose_trajectory.jsonl from a hand_data.zip.

    Returns:
        Dict mapping timestamp_ns (int) → {
            "0": {"wrist_xform": {"t_xyz": [...], "q_wxyz": [...]},
                  "joint_angles": [...22...],
                  "hand_confidence": float},
            "1": {...}   (if present)
        }
        Frames with empty hand_poses are omitted.
    """
    result = {}
    with zipfile.ZipFile(hand_zip_path, "r") as zf:
        with zf.open("umetrack_hand_pose_trajectory.jsonl") as f:
            for line in f:
                entry = json.loads(line)
                poses = entry.get("hand_poses", {})
                if poses:
                    result[entry["timestamp_ns"]] = poses
    return result


def read_dynamic_objects(gt_zip_path: Path) -> dict[int, list[dict]]:
    """Parse dynamic_objects.csv from a ground_truth.zip.

    Returns:
        Dict mapping timestamp_ns (int) → list of {
            "object_uid": str,
            "pos_world": np.ndarray([x, y, z], float32)
        }
    """
    result: dict[int, list[dict]] = {}
    with zipfile.ZipFile(gt_zip_path, "r") as zf:
        with zf.open("dynamic_objects.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                ts = int(row["timestamp[ns]"])
                pos = np.array([
                    float(row["t_wo_x[m]"]),
                    float(row["t_wo_y[m]"]),
                    float(row["t_wo_z[m]"]),
                ], dtype=np.float32)
                if ts not in result:
                    result[ts] = []
                result[ts].append({"object_uid": row["object_uid"], "pos_world": pos})
    return result


def read_metadata(gt_zip_path: Path) -> dict:
    """Parse metadata.json from a ground_truth.zip.

    Returns raw dict with keys: object_uids, object_bop_uids, object_names, etc.
    """
    with zipfile.ZipFile(gt_zip_path, "r") as zf:
        with zf.open("metadata.json") as f:
            return json.load(f)


def build_uid_to_bop(metadata: dict) -> dict[str, int]:
    """Build mapping: object_uid (str) → BOP ID (int) from metadata dict."""
    return {
        uid: int(bop)
        for uid, bop in zip(metadata["object_uids"], metadata["object_bop_uids"])
    }


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def find_sequences(data_dir: Path, split: str = "train") -> list[Path]:
    """Return sorted list of sequence directories under data_dir/split/.

    Each returned path is a directory containing:
      Hot3DQuest_v4.0.0_{seq_id}_hand_data.zip
      Hot3DQuest_v4.0.0_{seq_id}_ground_truth.zip
    """
    split_dir = data_dir / split
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def zip_paths(seq_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (hand_data_zip, ground_truth_zip) for a sequence directory.

    Returns (None, None) if either file is missing.
    """
    seq_id = seq_dir.name
    hand_zip = seq_dir / f"Hot3DQuest_v4.0.0_{seq_id}_hand_data.zip"
    gt_zip   = seq_dir / f"Hot3DQuest_v4.0.0_{seq_id}_ground_truth.zip"
    if not hand_zip.exists() or not gt_zip.exists():
        return None, None
    return hand_zip, gt_zip
