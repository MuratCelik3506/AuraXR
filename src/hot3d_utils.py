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
            "pos_world":  np.ndarray([x, y, z], float32),
            "quat_world": np.ndarray([w, x, y, z], float32)  — object orientation
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
                quat = np.array([
                    float(row["q_wo_w"]),
                    float(row["q_wo_x"]),
                    float(row["q_wo_y"]),
                    float(row["q_wo_z"]),
                ], dtype=np.float32)
                if ts not in result:
                    result[ts] = []
                result[ts].append({
                    "object_uid": row["object_uid"],
                    "pos_world":  pos,
                    "quat_world": quat,
                })
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

# ---------------------------------------------------------------------------
# Wrist rotation helpers — 6D continuous rotation representation
# ---------------------------------------------------------------------------

def quat_multiply(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1_wxyz.astype(np.float64)
    w2, x2, y2, z2 = q2_wxyz.astype(np.float64)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_to_rot_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q_wxyz.astype(np.float64)
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),     2*(x*z+y*w)    ],
        [2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)    ],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x**2+y**2)],
    ], dtype=np.float32)


def rot_mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s], dtype=np.float32)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s], dtype=np.float32)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s], dtype=np.float32)
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s], dtype=np.float32)


def look_rotation(forward: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Build quaternion [w,x,y,z] where local Z = forward.

    Matches Unity's Quaternion.LookRotation(forward, up) exactly (verified for
    canonical inputs forward=Z, forward=X, forward=Y).

    Singularity handling: when forward is nearly parallel to up, uses (0,0,1)
    as fallback up to avoid a degenerate cross product. The resulting discontinuity
    at the singularity threshold is handled at runtime by EMA smoothing in Unity
    (rotEmaAlpha), not by smooth blending here — smooth blending of up vectors
    introduces sign flips in the right-vector that are worse than the original switch.
    """
    if up is None:
        up = np.array([0., 1., 0.])
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(up, forward)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:          # forward nearly parallel to up — use secondary up
        up = np.array([0., 0., 1.])
        right = np.cross(up, forward)
        r_norm = np.linalg.norm(right) + 1e-8
    right = right / r_norm
    up_actual = np.cross(forward, right)
    R = np.column_stack([right, up_actual, forward])   # cols = [X, Y, Z] local axes
    return rot_mat_to_quat(R)


def wrist_rot_to_6d(q_wrist_hot3d_wxyz: np.ndarray, direction_hot3d: np.ndarray) -> np.ndarray:
    """Encode wrist quaternion as 6D rotation relative to approach direction.

    Both inputs are in HOT3D coordinate frame.  The 6D output can be decoded
    in Unity with no additional coordinate conversion beyond the direction
    conversion already done there (negate Z of dir_world to get Unity frame).

    Steps:
      1. Convert wrist quat and direction to Unity frame (negate Z)
      2. Canonical quat = LookRotation(dir_unity, up) — matches Unity exactly
      3. q_rel = canonical^{-1} ⊗ q_wrist_unity
      4. 6D = first two columns of rot_mat(q_rel): [col0(3), col1(3)]
    """
    # HOT3D wxyz → Unity wxyz: negate z-component (index 3 in [w,x,y,z])
    q_unity = q_wrist_hot3d_wxyz.astype(np.float32) * np.array([1., 1., 1., -1.], dtype=np.float32)
    # HOT3D xyz direction → Unity xyz direction: negate z
    dir_unity = direction_hot3d.astype(np.float32) * np.array([1., 1., -1.], dtype=np.float32)
    # Canonical rotation: local Z = dir_unity
    q_canonical = look_rotation(dir_unity)
    # Relative rotation: q_rel = canonical^{-1} * q_wrist
    q_rel = quat_multiply(quat_conjugate(q_canonical), q_unity)
    q_rel = q_rel / (np.linalg.norm(q_rel) + 1e-8)
    # 6D: first two columns of rotation matrix
    R = quat_to_rot_mat(q_rel)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)  # [col0(3), col1(3)]


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
