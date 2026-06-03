"""model_v2.py — AuraXR hand pose model, version 2.

Architecture changes from v1:
  1. Extended spatial input: 4 → 8 dims  (adds wrist quaternion w,x,y,z)
     The wrist quaternion encodes hand orientation (pronated vs. supinated, tilted),
     which strongly determines grip shape but was absent in v1.
  2. Per-finger decoder heads: single 22-dim head → 5 separate 4-dim heads.
     Each head specialises on one finger, preventing cross-finger interference
     in the gradient signal and allowing finger-level uncertainty modelling.
  3. LayerNorm after spatial encoder: stabilises training with wider input range.
  4. Slightly larger capacity: hidden=256, emb=128 (~4× more params than v1 ~54k → ~210k).
     Still fast enough for real-time inference on Quest 3 GPU/NPU via Unity Sentis.

Feature layout (15 dims):
  spatial_input (8): [dir_x, dir_y, dir_z, distance, wrist_qw, wrist_qx, wrist_qy, wrist_qz]
  object_input  (7): [grip_oh(4), bbox_x, bbox_y, bbox_z]  ← same as v1

Output (22 dims):
  22 UME joint angles, normalized to [-1, 1] via Tanh.
  Joints 20-21 are always 0 (placeholders) — excluded from loss.

Training: use train_v2.py which passes spatial_input as (B, 8) instead of (B, 4).
Export:   export_onnx_v2.py — two ONNX inputs ('spatial_input', 'object_input') same names as v1,
          but spatial_input shape is (1, 8) not (1, 4). Unity C# side assembles the 8-dim input.
"""

import torch
import torch.nn as nn


class AuraXRModelV2(nn.Module):
    # Joints 20-21 are always 0 — excluded from loss.
    ACTIVE_JOINTS = list(range(20))

    # UME joint ranges per finger (4 joints each):
    #   Thumb  [0-3]:  CMC-flex, abduction, MCP, DIP
    #   Index  [4-7]:  abduction, MCP, PIP, DIP
    #   Middle [8-11]: abduction, MCP, PIP, DIP
    #   Ring   [12-15]:abduction, MCP, PIP, DIP
    #   Pinky  [16-19]:abduction, MCP, PIP, DIP
    FINGER_RANGES = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]

    def __init__(
        self,
        spatial_input_dim: int   = 8,    # dir(3) + dist(1) + wrist_quat(4)
        object_input_dim:  int   = 7,    # grip_oh(4) + bbox(3)
        hidden_dim:        int   = 256,
        embedding_dim:     int   = 128,
        finger_hidden:     int   = 64,   # per-finger head hidden size
        dropout:           float = 0.30,
    ):
        super().__init__()
        self.spatial_input_dim = spatial_input_dim
        self.object_input_dim  = object_input_dim

        # Branch A: spatial encoder — includes wrist orientation
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

        # Branch B: object encoder — unchanged from v1
        self.object_encoder = nn.Sequential(
            nn.Linear(object_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.ReLU(),
        )

        # Shared trunk — fuses both branches
        fused_dim = embedding_dim * 2
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Per-finger decoder heads (5 fingers × 4 UME joints = 20 active joints)
        self.finger_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, finger_hidden),
                nn.ReLU(),
                nn.Linear(finger_hidden, 4),  # 4 UME joints per finger
                nn.Tanh(),
            )
            for _ in range(5)
        ])

        # Placeholder for joints 20-21 (always 0 in HOT3D UmeTrack)
        self.register_buffer("_zeros2", torch.zeros(1, 2))

    def forward(
        self,
        spatial_input: torch.Tensor,   # (B, 8)
        object_input:  torch.Tensor,   # (B, 7)
    ) -> torch.Tensor:
        """
        Returns:
            (B, 22) — normalized joint angles in [-1, 1].
            Joints 20-21 are always 0.
        """
        spatial_emb = self.spatial_encoder(spatial_input)
        object_emb  = self.object_encoder(object_input)
        fused       = self.trunk(torch.cat([spatial_emb, object_emb], dim=-1))

        # Per-finger decoding
        finger_parts = [head(fused) for head in self.finger_heads]  # 5 × (B, 4)
        joints_20    = torch.cat(finger_parts, dim=-1)              # (B, 20)

        # Append zero placeholders for joints 20-21
        batch = joints_20.shape[0]
        zeros = self._zeros2.expand(batch, -1)                      # (B, 2)
        return torch.cat([joints_20, zeros], dim=-1)                # (B, 22)

    @staticmethod
    def split_feature(feature: torch.Tensor):
        """Split (B, 15) feature vector into spatial (B, 8) and object (B, 7)."""
        return feature[:, :8], feature[:, 8:]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = AuraXRModelV2()
    print(f"AuraXRModelV2 — {model.count_params():,} parameters")

    B = 16
    sp = torch.randn(B, 8)
    ob = torch.randn(B, 7)
    out = model(sp, ob)
    print(f"Input:  spatial={tuple(sp.shape)}  object={tuple(ob.shape)}")
    print(f"Output: {tuple(out.shape)}  range=[{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (B, 22), "Output shape mismatch"
    assert out[:, 20:].abs().max() < 1e-6, "Placeholders should be zero"
    print("OK — forward pass verified.")
