"""Input/output constants shared by datasets, models, and Unity export.

Mirrors docs/A-veri-hazirlama.md (canonical data contract), docs/B-grasp-model.md
(model I/O contract) and docs/D-deney-plani.md (split sizes).
"""
from __future__ import annotations


# --- Core tensor dimensions (B1, B2) ---
WRIST_DIM = 6
HOT3D_FRAME_DIM = 13  # rel_pos(3) + rel_rot6d(6) + rel_vel(3) + dist(1)
FINGER_POSE_DIM = 45  # MANO finger axis-angle, 15 joints x 3
MANO_SHAPE_DIM = 10
MANO_POSE_DIM = 48  # global_orient(3) + finger_aa45(45)
OBJ_POINT_DIM = 3
DEFAULT_N_POINTS = 1024
OBJ_EMB_DIM = 256
LATENT_DIM = 64
NUM_FINGER_JOINTS = 15
NUM_FINGERTIPS = 5

HOT3D_FEATURE_ORDER = (
    "rel_pos(3)",
    "rel_rot6d(6)",
    "rel_vel(3)",
    "dist(1)",
)

MANO_FINGER_RANGES = {
    "index_mcp": (0, 3),
    "index_pip": (3, 6),
    "index_dip": (6, 9),
    "middle_mcp": (9, 12),
    "middle_pip": (12, 15),
    "middle_dip": (15, 18),
    "ring_mcp": (18, 21),
    "ring_pip": (21, 24),
    "ring_dip": (24, 27),
    "pinky_mcp": (27, 30),
    "pinky_pip": (30, 33),
    "pinky_dip": (33, 36),
    "thumb_cmc": (36, 39),
    "thumb_mcp": (39, 42),
    "thumb_ip": (42, 45),
}

# Joint index (0..14) -> finger id (0=index,1=middle,2=ring,3=pinky,4=thumb), used by
# the B4 self-attention learned_joint_emb and by fingertip-only metrics (B10).
JOINT_TO_FINGER = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
FINGERTIP_JOINT_INDICES = [2, 5, 8, 11, 14]  # *_dip / thumb_ip per finger, last joint in chain

# --- A5: heuristic quality_label weights ---
QUALITY_W_CONTACT = 1.0
QUALITY_W_PENETRATION = 0.3
QUALITY_W_DIST = 0.2
QUALITY_MAX_PENETRATION_M = 0.01  # 10mm normalization ceiling
QUALITY_MAX_DIST_M = 0.05  # 5cm normalization ceiling
CONTACT_THRESHOLD_M = 0.030  # 30mm: matches 3cm AABB contact detection used in HOT3D build

# --- A4: HOT3D segmentation thresholds ---
APPROACH_DIST_THRESHOLD_M = 0.15
GRASP_FINGER_MCP_DEG = 20.0
GRASP_CONTACT_THRESHOLD_M = 0.005
GRASP_VELOCITY_THRESHOLD_M_S = 0.1
TRANSITION_WINDOW_FRAMES = 10

# --- A7/D2: HOT3D 4-level object split (33 objects total) ---
HOT3D_SPLIT_COUNTS = {
    "backbone_train": 22,
    "backbone_val": 4,
    "calibration": 3,
    "held_out_test": 4,
}

# --- C3: Unity physics eval defaults ---
UNITY_FORCE_ALPHA = 1.0
UNITY_DISPLACEMENT_TAU = 0.10  # fraction of bbox diagonal
UNITY_ROTATION_TAU_DEG = 15.0
UNITY_FRICTION_BASELINE = 0.6


def validate_pose45_shape(size: int) -> None:
    if size != FINGER_POSE_DIM:
        raise ValueError(f"Expected MANO finger pose dim {FINGER_POSE_DIM}, got {size}")
