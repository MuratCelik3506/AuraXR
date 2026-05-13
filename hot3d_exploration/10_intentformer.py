"""
10_intentformer.py — IntentFormer model architecture for AuraXR hand pose prediction.

Architecture overview:
  [Feature seq T×F_IN] ──► Linear projection ──► d_model=256
                            + Learned positional encoding
                            4× Transformer Encoder layers (heads=8, ffn=512)
                                    │
                          ┌─────────┴──────────┐
                          │  Learned pose query │  (2 tokens: h0, h1)
                          │  4× Decoder layers  │  (cross-attn over encoder)
                          └─────────┬──────────┘
                                    │
                        MLP head → 78-dim output

Visual branch (optional):
  Supports pre-extracted 64-dim CNN embeddings injected as an extra token
  before the encoder. When visual_dim=0 the branch is disabled.

Output: 78 floats
  [0:15]   mano_pose_h0   — θ for hand 0
  [15:25]  mano_betas_h0  — β shape for hand 0
  [25:28]  wrist_t_h0     — wrist translation
  [28:32]  wrist_q_h0     — wrist quaternion (w,x,y,z)
  [32:35]  delta_t_h0     — controller-to-wrist offset translation
  [35:39]  delta_q_h0     — controller-to-wrist offset quaternion
  [39:78]  same for hand 1

Usage (standalone test):
  python 10_intentformer.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dimensions from 09_build_dataset.py
F_IN        = 96   # feature vector dim per frame
T           = 16   # temporal window
TARGET_DIM  = 78   # full output dim
MANO_POSE_DIM = 15
MANO_BETA_DIM = 10
WRIST_DIM     = 7   # 3+4
DELTA_DIM     = 7   # 3+4
PER_HAND_DIM  = MANO_POSE_DIM + MANO_BETA_DIM + WRIST_DIM + DELTA_DIM  # 39
assert PER_HAND_DIM * 2 == TARGET_DIM


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class LearnedPE(nn.Module):
    """Learned absolute positional encoding for a fixed max sequence length."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        return x + self.pe[:x.size(1)]


# ---------------------------------------------------------------------------
# Object Category Embedding
# ---------------------------------------------------------------------------

class ObjectCategoryEmbedding(nn.Module):
    """
    Embed the raw object category ID (0–33, 0=unknown) into a learned 16-dim vector.
    The embedding is added to the object-context portion of the feature vector.
    """
    def __init__(self, num_categories: int = 34, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        self.proj  = nn.Linear(embed_dim, embed_dim)

    def forward(self, cat_ids: torch.Tensor) -> torch.Tensor:
        """cat_ids: [B, 2] int — category IDs for hand0 and hand1."""
        return self.proj(self.embed(cat_ids))  # [B, 2, 16]


# ---------------------------------------------------------------------------
# Input Projection
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """
    Projects each time-step feature vector to d_model.
    Also handles the learned category embedding injection.
    """
    def __init__(self, f_in: int, d_model: int, cat_embed_dim: int = 16):
        super().__init__()
        # The raw feature has category as a float scalar (1 dim per hand, at positions 18 and 24).
        # We subtract 1 float per hand and add cat_embed_dim per hand to get the projected dim.
        f_numeric = f_in - 2 + cat_embed_dim * 2   # replace 2 cat scalars with 2×16 embeddings
        self.cat_emb = ObjectCategoryEmbedding(34, cat_embed_dim)
        self.linear  = nn.Linear(f_numeric, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F_IN]
        Returns: [B, T, d_model]
        """
        B, L, _ = x.shape

        # Extract category IDs (positions 18 and 24 in feature vector)
        # Feature layout: ctrl0(9) + ctrl1(9) + obj_ctx0(3+3+1=7) + obj_ctx1(7) + visual(64)
        # Category ID positions: offset 18 (hand0) and offset 25 (hand1) — 0-indexed
        cat_h0 = x[:, :, 18].long().clamp(0, 33)   # [B, T]
        cat_h1 = x[:, :, 25].long().clamp(0, 33)   # [B, T]

        # Get category embeddings [B, T, 16]
        cat_ids_flat = torch.stack([cat_h0, cat_h1], dim=-1).reshape(B * L, 2)
        cat_embeds   = self.cat_emb(cat_ids_flat).reshape(B, L, 32)  # 2×16=32

        # Remove the 2 raw category scalars and append embeddings
        # Feature: [0:18] ctrl+obj_centroid_bbox | [18] cat_h0 | [19:25] obj1 | [25] cat_h1 | [26:96] visual
        numeric = torch.cat([
            x[:, :, :18],    # controller + obj centroid/bbox for h0
            x[:, :, 19:25],  # obj centroid/bbox for h1
            x[:, :, 26:],    # visual (64 dims)
            cat_embeds,       # learned category embeddings (32 dims)
        ], dim=-1)  # [B, T, F_IN - 2 + 32]

        return self.linear(numeric)  # [B, T, d_model]


# ---------------------------------------------------------------------------
# IntentFormer
# ---------------------------------------------------------------------------

class IntentFormer(nn.Module):
    """
    Temporal Transformer for bimanual hand pose prediction.

    Args:
        f_in        : feature dim per timestep
        t           : temporal window length
        d_model     : transformer hidden dim
        nhead       : number of attention heads
        num_enc_layers : encoder depth
        num_dec_layers : decoder depth
        ffn_dim     : feedforward network hidden dim
        dropout     : dropout probability
        target_dim  : output dimension (78)
    """

    def __init__(
        self,
        f_in:           int   = F_IN,
        t:              int   = T,
        d_model:        int   = 256,
        nhead:          int   = 8,
        num_enc_layers: int   = 4,
        num_dec_layers: int   = 4,
        ffn_dim:        int   = 512,
        dropout:        float = 0.1,
        target_dim:     int   = TARGET_DIM,
    ):
        super().__init__()
        self.d_model    = d_model
        self.target_dim = target_dim

        # Input projection + positional encoding
        self.input_proj = InputProjection(f_in, d_model)
        self.pos_enc    = LearnedPE(t + 1, d_model)  # +1 for optional visual token

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers,
                                             enable_nested_tensor=False)

        # Learned decoder query tokens: 2 tokens (one per hand)
        self.query_tokens = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # MLP output head: 2 query tokens → PER_HAND_DIM each
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, PER_HAND_DIM),
        )

        # Quaternion normalisation layers (ensures unit-norm outputs)
        # Applied to wrist_q and delta_q slices post-head
        self._wrist_q_slice  = slice(MANO_POSE_DIM + MANO_BETA_DIM + 3,
                                      MANO_POSE_DIM + MANO_BETA_DIM + 7)   # [28:32]
        self._delta_q_slice  = slice(MANO_POSE_DIM + MANO_BETA_DIM + 7 + 3,
                                      PER_HAND_DIM)                          # [35:39]

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Bias zero
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, F_IN]
        Returns: [B, TARGET_DIM]
        """
        B = x.size(0)

        # Project + add positional encoding
        enc_in = self.pos_enc(self.input_proj(x))  # [B, T, d_model]

        # Encode
        memory = self.encoder(enc_in)  # [B, T, d_model]

        # Decode: expand query tokens to batch size
        queries = self.query_tokens.expand(B, -1, -1)  # [B, 2, d_model]
        dec_out = self.decoder(queries, memory)         # [B, 2, d_model]

        # Head: [B, 2, PER_HAND_DIM]
        per_hand = self.head(dec_out)

        # Quaternions are normalised inside geodesic_quat_loss during training
        # and via .normalized in Unity at inference — no in-place ops needed here.

        # Flatten to [B, TARGET_DIM]
        return per_hand.reshape(B, self.target_dim)

    def predict_hand_params(self, output: torch.Tensor) -> dict:
        """Split a [B, TARGET_DIM] output into named parameter dicts."""
        return _predict_hand_params(output)


# ---------------------------------------------------------------------------
# Baselines (for ablation comparison)
# ---------------------------------------------------------------------------

def _predict_hand_params(output: torch.Tensor) -> dict:
    """Split a [B, TARGET_DIM] output into named hand parameter dicts."""
    h0 = output[:, :PER_HAND_DIM]
    h1 = output[:, PER_HAND_DIM:]
    return {
        "mano_pose_h0":  h0[:, 0:15],
        "mano_betas_h0": h0[:, 15:25],
        "wrist_t_h0":    h0[:, 25:28],
        "wrist_q_h0":    h0[:, 28:32],
        "delta_t_h0":    h0[:, 32:35],
        "delta_q_h0":    h0[:, 35:39],
        "mano_pose_h1":  h1[:, 0:15],
        "mano_betas_h1": h1[:, 15:25],
        "wrist_t_h1":    h1[:, 25:28],
        "wrist_q_h1":    h1[:, 28:32],
        "delta_t_h1":    h1[:, 32:35],
        "delta_q_h1":    h1[:, 35:39],
    }


class SingleFrameMLP(nn.Module):
    """
    Baseline: flat MLP on a single frame (no temporal context).
    Takes the last frame of the window only.
    """
    def __init__(self, f_in: int = F_IN, hidden: int = 512, target_dim: int = TARGET_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F_IN] — uses last frame only."""
        return self.net(x[:, -1, :])

    def predict_hand_params(self, output: torch.Tensor) -> dict:
        return _predict_hand_params(output)


class GRUBaseline(nn.Module):
    """Baseline: bidirectional GRU on the temporal sequence."""
    def __init__(self, f_in: int = F_IN, hidden: int = 256,
                 num_layers: int = 2, target_dim: int = TARGET_DIM):
        super().__init__()
        self.gru  = nn.GRU(f_in, hidden, num_layers=num_layers,
                           batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F_IN]"""
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])  # last time step

    def predict_hand_params(self, output: torch.Tensor) -> dict:
        return _predict_hand_params(output)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = IntentFormer()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"IntentFormer parameters : {n_params:,}")

    dummy = torch.randn(4, T, F_IN)
    out   = model(dummy)
    print(f"Input  shape : {list(dummy.shape)}")
    print(f"Output shape : {list(out.shape)}")
    assert out.shape == (4, TARGET_DIM), f"Unexpected output shape: {out.shape}"

    params = model.predict_hand_params(out)
    for k, v in params.items():
        print(f"  {k:<18} {list(v.shape)}")

    # Baselines
    mlp = SingleFrameMLP()
    gru = GRUBaseline()
    print(f"\nSingleFrameMLP params : {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"GRUBaseline params    : {sum(p.numel() for p in gru.parameters()):,}")
    print(f"\nAll shapes OK.")
