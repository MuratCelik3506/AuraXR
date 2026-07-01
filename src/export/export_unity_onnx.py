"""Export a Unity InferenceEngine-friendly deterministic AuraXR ONNX.

The generic exporter traces model.infer(), which includes CVAE random sampling
and candidate selection. Unity's ONNX importer is stricter than ONNX Runtime, so
this exporter removes runtime randomness/dynamic candidate indexing:

  context -> joint tokens -> CVAE decode with z=zeros -> heads

Inputs are fixed to the demo contract:
  frame_feat    (1, 16, 13)
  obj_pts       (1, 1024, 3)
  contact_flag  (1, 16, 1)
  prev_pose     (1, 45)

Outputs:
  selected_pose (1, 45)
  quality_score (1, 1)
  success_prob  (1, 1)
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
from model.model_io import DEFAULT_N_POINTS, FINGER_POSE_DIM, HOT3D_FRAME_DIM, LATENT_DIM  # noqa: E402


class UnityDeterministicWrapper(nn.Module):
    def __init__(self, model: GraspModel, window: int) -> None:
        super().__init__()
        self.model = model
        self.window = window

    def _manual_temporal_encoder(self, frame_feat: torch.Tensor, contact_flag: torch.Tensor) -> torch.Tensor:
        """Equivalent to TemporalEncoder.forward, but avoids exporting ONNX GRU op.

        Unity InferenceEngine 2.6.1 does not support ONNX GRU. The demo window is
        fixed at 16 frames, so tracing this Python loop unrolls the recurrence
        into MatMul/Add/Sigmoid/Tanh ops.
        """
        temporal = self.model.temporal_encoder
        x = torch.cat([frame_feat, contact_flag], dim=-1)
        x = temporal.input_proj(x)

        gru = temporal.gru
        w_ih = gru.weight_ih_l0
        w_hh = gru.weight_hh_l0
        b_ih = gru.bias_ih_l0
        b_hh = gru.bias_hh_l0
        hidden = gru.hidden_size
        h = torch.zeros(frame_feat.shape[0], hidden, device=frame_feat.device, dtype=frame_feat.dtype)

        for t in range(self.window):
            xt = x[:, t, :]
            gi = torch.matmul(xt, w_ih.t()) + b_ih
            gh = torch.matmul(h, w_hh.t()) + b_hh
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            resetgate = torch.sigmoid(i_r + h_r)
            updategate = torch.sigmoid(i_z + h_z)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h = newgate + updategate * (h - newgate)
        return h

    def forward(
        self,
        frame_feat: torch.Tensor,
        obj_pts: torch.Tensor,
        contact_flag: torch.Tensor,
        prev_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal = self._manual_temporal_encoder(frame_feat, contact_flag)
        obj_emb = self.model.encode_object(obj_pts)
        context = self.model.context_encoder(temporal, obj_emb)
        joint_tokens = self.model.joint_tokens_from_context(context, prev_pose)
        z = torch.zeros(frame_feat.shape[0], LATENT_DIM, device=frame_feat.device, dtype=frame_feat.dtype)
        pred = self.model.cvae.decode(joint_tokens, z)
        quality, success = self.model._heads(joint_tokens, pred)
        return pred, quality, success


def load_model(checkpoint: str) -> GraspModel:
    model = GraspModel()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aura_phase2_best.pt")
    parser.add_argument("--out", type=str, default="checkpoints/grasp_model_unity.onnx")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    model = load_model(args.checkpoint)
    wrapper = UnityDeterministicWrapper(model, window=args.window).eval()

    dummy_frame = torch.zeros(1, args.window, HOT3D_FRAME_DIM)
    dummy_pts = torch.zeros(1, args.n_points, 3)
    dummy_contact = torch.zeros(1, args.window, 1)
    dummy_prev = torch.zeros(1, FINGER_POSE_DIM)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_frame, dummy_pts, dummy_contact, dummy_prev),
            out_path,
            input_names=["frame_feat", "obj_pts", "contact_flag", "prev_pose"],
            output_names=["selected_pose", "quality_score", "success_prob"],
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(f"Wrote {out_path} (fixed inputs: frame={tuple(dummy_frame.shape)}, pts={tuple(dummy_pts.shape)})")


if __name__ == "__main__":
    main()
