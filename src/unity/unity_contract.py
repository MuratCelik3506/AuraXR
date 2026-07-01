"""Unity runtime / eval JSON contract (docs/C-unity-entegrasyonu.md, D10).

Key design: quality_score and success_prob are SEPARATE fields.
  quality_score: heuristic [0,1] from contact/penetration/distance (MSE head, A5/B7)
  success_prob:  Unity physics binary label [0,1] (BCE head, C3/B7, trained offline
                 after Unity simulation exports success_label CSV)
These MUST NOT be conflated; the docs explicitly distinguish them.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.model_io import UNITY_DISPLACEMENT_TAU, UNITY_FORCE_ALPHA, UNITY_ROTATION_TAU_DEG  # noqa: E402


@dataclass
class GraspCandidate:
    """Per-candidate inference output (B9 K-candidate sampling)."""

    candidate_index: int
    quality_score: float
    success_prob: float


@dataclass
class UnityEvalRecord:
    """Single Unity physics evaluation record (C3, D8).

    Fields:
      quality_score: heuristic confidence from the model's quality head [0,1]
      success_prob:  model's success_prob head output [0,1]
      success:       ground-truth binary outcome from Unity physics simulation
      displacement_cm: object centroid displacement after force disturbance
    """

    object_id: str
    test_direction: str
    window_size: int
    candidate_count: int
    selected_candidate_index: int
    quality_score: float
    success_prob: float
    success: bool
    contact_ratio: float
    penetration_mm: float
    displacement_cm: float
    latency_ms: float
    force_alpha: float = UNITY_FORCE_ALPHA
    displacement_tau: float = UNITY_DISPLACEMENT_TAU
    rotation_tau_deg: float = UNITY_ROTATION_TAU_DEG
    failure_category: str | None = None
    candidates: list[GraspCandidate] = field(default_factory=list)


@dataclass
class UnityInferRequest:
    """JSON payload sent from Unity C# to the Python inference server (C2)."""

    object_id: str
    obj_pts: list[list[float]]   # [[x,y,z], ...] length N_POINTS
    frame_feat: list[list[float]]  # [[13 floats], ...] length T
    contact_flag: list[float]
    k_candidates: int = 1


@dataclass
class UnityInferResponse:
    """JSON payload returned from inference server to Unity (C2)."""

    selected_pose: list[float]        # 45 floats, MANO finger axis-angle
    quality_score: float
    success_prob: float
    candidate_poses: list[list[float]] = field(default_factory=list)
    candidate_quality_scores: list[float] = field(default_factory=list)
    candidate_success_probs: list[float] = field(default_factory=list)
    selected_candidate_index: int = 0
    latency_ms: float = 0.0


def write_unity_eval_record(record: UnityEvalRecord, path: str | Path) -> None:
    d = asdict(record)
    Path(path).write_text(json.dumps(d, indent=2))


def load_unity_eval_records(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return [data]
    return data


def check_success(displacement_cm: float, rotation_deg: float, dropped: bool) -> bool:
    """C3 success criterion: displacement < tau*bbox_diagonal AND rotation < tau_deg AND not dropped."""
    disp_ok = displacement_cm < UNITY_DISPLACEMENT_TAU * 100.0
    rot_ok = rotation_deg < UNITY_ROTATION_TAU_DEG
    return disp_ok and rot_ok and not dropped


if __name__ == "__main__":
    rec = UnityEvalRecord(
        object_id="mug_001",
        test_direction="top",
        window_size=16,
        candidate_count=3,
        selected_candidate_index=0,
        quality_score=0.82,
        success_prob=0.75,
        success=True,
        contact_ratio=0.8,
        penetration_mm=1.2,
        displacement_cm=0.8,
        latency_ms=4.7,
        candidates=[
            GraspCandidate(0, 0.82, 0.75),
            GraspCandidate(1, 0.65, 0.60),
            GraspCandidate(2, 0.48, 0.41),
        ],
    )
    print(json.dumps(asdict(rec), indent=2))
