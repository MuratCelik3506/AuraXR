"""Export TemporalGeometryConditionedGraspModel to ONNX for Unity Sentis (C1).

Exports THREE output nodes: selected_pose, quality_score, success_prob.
Unity reads these to choose a grasp candidate and log dual-head metrics (B7/C3).

Usage:
  python3 src/export/export_onnx.py --checkpoint checkpoints/grasp_best.pt --out checkpoints/grasp_model.onnx
  python3 src/export/export_onnx.py --checkpoint ... --window 16 --k 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import GraspModel  # noqa: E402
from model.model_io import DEFAULT_N_POINTS, FINGER_POSE_DIM, HOT3D_FRAME_DIM  # noqa: E402
from utils.paths import CHECKPOINT_DIR  # noqa: E402


class GraspModelExportWrapper(nn.Module):
    """Strips dynamic dispatch for ONNX tracing; returns (pose, quality, success)."""

    def __init__(self, model: GraspModel, k: int = 1) -> None:
        super().__init__()
        self.model = model
        self.k = k

    def forward(
        self,
        frame_feat: torch.Tensor,
        obj_pts: torch.Tensor,
        contact_flag: torch.Tensor,
        prev_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model.infer(frame_feat, obj_pts, prev_pose=prev_pose,
                               contact_flag=contact_flag, k=self.k)
        return (
            out["selected_pose"],
            out["selected_quality_score"],
            out["selected_success_prob"],
        )


def load_model(checkpoint: str | None) -> GraspModel:
    model = GraspModel()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(CHECKPOINT_DIR / "grasp_model.onnx"))
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    wrapper = GraspModelExportWrapper(model, k=args.k)
    wrapper.eval()
    # PyTorch's MultiheadAttention fastpath lowers to aten::_native_multi_head_attention,
    # which the ONNX exporter cannot serialize in opset 17.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    dummy_frame   = torch.randn(1, args.window, HOT3D_FRAME_DIM)
    dummy_pts     = torch.randn(1, args.n_points, 3)
    dummy_contact = torch.zeros(1, args.window, 1)
    dummy_prev    = torch.zeros(1, FINGER_POSE_DIM)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_frame, dummy_pts, dummy_contact, dummy_prev),
            out_path,
            input_names=["frame_feat", "obj_pts", "contact_flag", "prev_pose"],
            output_names=["selected_pose", "quality_score", "success_prob"],
            dynamic_axes={
                "frame_feat":    {0: "batch", 1: "window"},
                "obj_pts":       {0: "batch"},
                "contact_flag":  {0: "batch", 1: "window"},
                "prev_pose":     {0: "batch"},
                "selected_pose": {0: "batch"},
                "quality_score": {0: "batch"},
                "success_prob":  {0: "batch"},
            },
            opset_version=17,
        )
    print(f"Wrote {out_path}  (frame_feat={dummy_frame.shape}, k={args.k})")


if __name__ == "__main__":
    main()
