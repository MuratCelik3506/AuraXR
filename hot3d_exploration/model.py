"""model.py — Two-branch MLP for AuraXR hand pose prediction.

Architecture (default: hidden=128, emb=64, ~54k params):
    Branch A (Spatial Encoder): [rel_pos(3) + distance(1)] = 4
        FC(4→128) → ReLU → FC(128→64) → ReLU

    Branch B (Object Encoder): [grip_onehot(4) + bbox(3)] = 7
        FC(7→128) → ReLU → FC(128→64) → ReLU

    Head: Concat(128)
        FC(128→128) → ReLU → Dropout → FC(128→128) → ReLU → Dropout → FC(128→22) → Tanh

Joints 20–21 are always 0 in HOT3D UmeTrack (placeholder joints).
They are excluded from training loss via ACTIVE_JOINTS mask.
"""

import torch
import torch.nn as nn


class AuraXRModel(nn.Module):
    # Joints 20–21 are always 0.0 in HOT3D — excluded from training loss.
    ACTIVE_JOINTS = list(range(20))

    def __init__(
        self,
        spatial_input_dim: int   = 4,
        object_input_dim:  int   = 7,
        hidden_dim:        int   = 128,
        embedding_dim:     int   = 64,
        output_dim:        int   = 22,
        dropout:           float = 0.20,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

        self.object_encoder = nn.Sequential(
            nn.Linear(object_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, spatial_input: torch.Tensor, object_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_input: (B, 4)  — [rel_pos(3), distance(1)], normalized
            object_input:  (B, 7)  — [grip_onehot(4), bbox(3)], normalized
        Returns:
            (B, 22) — normalized joint angles in [-1, 1]
        """
        return self.head(torch.cat([
            self.spatial_encoder(spatial_input),
            self.object_encoder(object_input),
        ], dim=-1))

    @staticmethod
    def split_feature(feature: torch.Tensor):
        """Split (B, 11) feature vector into spatial (B, 4) and object (B, 7)."""
        return feature[:, :4], feature[:, 4:]
