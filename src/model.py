"""model.py — AuraXR hand pose model.

Feature layout (15 dims):
  spatial_input (8): [dir_world(3), dir_obj_local(3), dist(1), approach_speed(1)]
  object_input  (7): [grip_oh(4), bbox_x, bbox_y, bbox_z]

Outputs:
  joint_angles  (22): UME joint angles, normalized. Joints 20-21 = 0 placeholders.
  wrist_rot_6d   (6): Wrist palm orientation — first two columns of rotation matrix
                      of q_rel, where q_rel = canonical^{-1} ⊗ q_wrist (Unity frame).
                      Decoded in Unity via Gram-Schmidt orthogonalization.

Architecture changes from v1:
  - finger_hidden: 64 → 128  (direct capacity increase for finger heads)
  - trunk: 2 layers → 3 layers  (deeper feature fusion)
"""

import torch
import torch.nn as nn


class AuraXRModel(nn.Module):
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
        spatial_input_dim: int   = 8,    # dir_world(3) + dir_obj_local(3) + dist(1) + approach_speed(1)
        object_input_dim:  int   = 7,    # grip_oh(4) + bbox(3)
        hidden_dim:        int   = 256,
        embedding_dim:     int   = 128,
        finger_hidden:     int   = 128,  # per-finger head hidden size (raised from 64)
        dropout:           float = 0.40,
    ):
        super().__init__()
        self.spatial_input_dim = spatial_input_dim
        self.object_input_dim  = object_input_dim

        # Branch A: spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

        # Branch B: object encoder (LayerNorm added to match spatial encoder symmetry)
        self.object_encoder = nn.Sequential(
            nn.Linear(object_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.ReLU(),
        )

        # Shared trunk — fuses both branches (3 layers for richer feature combinations)
        fused_dim = embedding_dim * 2
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
        )

        # Per-finger decoder heads — linear output, no Tanh
        # Compound loss (train.py) handles valid range via penalty term.
        self.finger_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, finger_hidden),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(finger_hidden, 4),
            )
            for _ in range(5)
        ])

        # Wrist rotation head — predicts palm orientation as 6D continuous rotation
        # relative to approach direction (decoded via Gram-Schmidt in Unity)
        self.wrist_rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 6),
        )

        # Auxiliary grip classifier — training only, not exported to ONNX
        self.grip_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

        # Placeholder for joints 20-21 (always 0 in HOT3D UmeTrack)
        self.register_buffer("_zeros2", torch.zeros(1, 2))

    def _encode(self, spatial_input: torch.Tensor, object_input: torch.Tensor) -> torch.Tensor:
        spatial_emb = self.spatial_encoder(spatial_input)
        object_emb  = self.object_encoder(object_input)
        return self.trunk(torch.cat([spatial_emb, object_emb], dim=-1))

    def forward(
        self,
        spatial_input: torch.Tensor,   # (B, 8)
        object_input:  torch.Tensor,   # (B, 7)
    ):
        """ONNX-export path. Returns (joint_angles (B,22), wrist_rot_6d (B,6)).
        Joints 20-21 are always 0.
        """
        fused        = self._encode(spatial_input, object_input)
        finger_parts = [head(fused) for head in self.finger_heads]
        joints_20    = torch.cat(finger_parts, dim=-1)
        batch        = joints_20.shape[0]
        zeros        = self._zeros2.expand(batch, -1)
        joint_angles = torch.cat([joints_20, zeros], dim=-1)
        wrist_rot    = self.wrist_rotation_head(fused)
        return joint_angles, wrist_rot

    def forward_train(
        self,
        spatial_input: torch.Tensor,   # (B, 8)
        object_input:  torch.Tensor,   # (B, 7)
    ):
        """Training path. Returns (joint_angles (B,22), wrist_rot_6d (B,6), grip_logits (B,4))."""
        fused        = self._encode(spatial_input, object_input)
        finger_parts = [head(fused) for head in self.finger_heads]
        joints_20    = torch.cat(finger_parts, dim=-1)
        batch        = joints_20.shape[0]
        zeros        = self._zeros2.expand(batch, -1)
        joint_angles = torch.cat([joints_20, zeros], dim=-1)
        wrist_rot    = self.wrist_rotation_head(fused)
        grip_logits  = self.grip_classifier(fused)
        return joint_angles, wrist_rot, grip_logits

    @staticmethod
    def split_feature(feature: torch.Tensor):
        """Split (B, 15) feature vector into spatial (B, 8) and object (B, 7)."""
        return feature[:, :8], feature[:, 8:]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = AuraXRModel()
    print(f"AuraXRModel — {model.count_params():,} parameters")

    B = 16
    sp = torch.randn(B, 8)
    ob = torch.randn(B, 7)
    joint_angles, wrist_rot           = model(sp, ob)
    joints, wrist_t, logits           = model.forward_train(sp, ob)
    print(f"Input:  spatial={tuple(sp.shape)}  object={tuple(ob.shape)}")
    print(f"forward():       joint_angles={tuple(joint_angles.shape)}  wrist_rot={tuple(wrist_rot.shape)}")
    print(f"forward_train(): joints={tuple(joints.shape)}  wrist_rot={tuple(wrist_t.shape)}  logits={tuple(logits.shape)}")
    assert joint_angles.shape == (B, 22), "joint_angles shape mismatch"
    assert wrist_rot.shape    == (B,  6), "wrist_rot shape mismatch"
    assert joint_angles[:, 20:].abs().max() < 1e-6, "Placeholders should be zero"
    print("OK — forward pass verified.")
