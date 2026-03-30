"""
IntentFormer: Transformer-based Early Action Prediction
=======================================================

This module implements the core architecture for predicting user intent 
from hand kinematics and object poses in XR environments.

Design Rationale:
-----------------
To achieve "Early Prediction," the model must capture temporal 
dependencies in the motion's initial phase. We use a Transformer 
architecture because its self-attention mechanism can weigh different 
frames of the trajectory independently of their distance, unlike RNNs.

Architecture Details:
---------------------
1. Joint Fusion:
   Concatenates hand joints (126) and object RT (16) into a 142-dim 
   feature vector per frame.
   
2. Linear Projection:
   Projects the 142 features into a higher-dimensional embedding (d_model).

3. Observation-Ratio Encoding:
   Unlike standard Transformers, we inject the 'obs_ratio' (0.2-1.0) 
   into the positional encoding. This informs the model how much of the 
   action has already occurred, helping it calibrate its confidence.

4. Transformer Encoder:
   N layers of self-attention (default N=4) process the sequence. 
   We use a [CLS] token at the start to aggregate global intent information.

5. Classification Head:
   A simple linear layer maps the [CLS] embedding to target action classes.

Performance:
-----------
- Lightweight: <1M parameters.
- Fast: Optimized for Apple Silicon (M-series) and CoreML/ANE deployment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal PE + an optional scalar 'obs_ratio' embedding
    projected and added to every position.  This lets the model know
    'how far into the motion' the window was sampled.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                        # (1, max_len, d_model)
        self.register_buffer("pe", pe)

        # Observation-ratio conditioning: scalar → d_model
        self.obs_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor, obs_ratio: torch.Tensor) -> torch.Tensor:
        """
        x         : (B, T, d_model)
        obs_ratio : (B,)  values in [0, 1]
        """
        T   = x.size(1)
        pos = self.pe[:, :T, :]                     # (1, T, d_model)
        obs = self.obs_proj(obs_ratio.unsqueeze(-1)) # (B, d_model)
        obs = obs.unsqueeze(1)                       # (B, 1, d_model)
        return self.dropout(x + pos + obs)


# ─────────────────────────────────────────────────────────
# Early Prediction Loss Penalty
# ─────────────────────────────────────────────────────────

class EarlyPredictionLoss(nn.Module):
    """
    Weighted cross-entropy that rewards correct predictions made at lower
    observation ratios.  Penalty weight = (1 - obs_ratio) × alpha.

    If the model is wrong at low obs_ratio → large loss (learning signal).
    If the model is right at low obs_ratio → small loss (bonus).
    """

    def __init__(self, alpha: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits:    torch.Tensor,    # (B, C)
        labels:    torch.Tensor,    # (B,)
        obs_ratio: torch.Tensor,    # (B,) in [0, 1]
    ) -> torch.Tensor:
        base_loss = F.cross_entropy(
            logits, labels,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # Weight: observations made earlier get a higher multiplier
        weight = 1.0 + self.alpha * (1.0 - obs_ratio)
        return (weight * base_loss).mean()


# ─────────────────────────────────────────────────────────
# IntentFormer
# ─────────────────────────────────────────────────────────

class IntentFormer(nn.Module):
    """
    Intent-prediction Transformer.

    Args:
        input_dim    : raw feature dimension per time-step (default 142 = 126+16)
        d_model      : Transformer embedding dimension (default 128)
        nhead        : number of attention heads (default 4)
        num_layers   : number of encoder layers (default 4)
        dim_feedforward: FFN inner dim (default 512)
        num_classes  : number of action categories (default 36)
        window_size  : temporal window T (default 30)
        dropout      : dropout rate (default 0.1)
    """

    def __init__(
        self,
        input_dim:       int = 378 + 16,   # (126*3) hand + 16 obj_rt
        d_model:         int = 128,
        nhead:           int = 4,
        num_layers:      int = 4,
        dim_feedforward: int = 512,
        num_classes:     int = 36,
        window_size:     int = 30,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ── 1. Input projection ──────────────────────────────
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
        )

        # ── 2. CLS token ─────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── 3. Positional + obs-ratio encoding ───────────────
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=window_size + 1, dropout=dropout
        )

        # ── 4. Transformer Encoder ───────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,       # (B, T, d_model)
            activation="gelu",
            norm_first=True,        # Pre-LN (more stable, lower latency)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── 5. Classification head ────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # ── 6. Pose Regression head (Next Frame) ──────────────
        # Predicts 126 coordinates (21 joints * 2 hands * 3) or as per input
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 126),           # always predict 126 positions
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hand_flat:  torch.Tensor,   # (B, T, 126)
        obj_rt:     torch.Tensor,   # (B, T, 16)
        obs_ratio:  torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, pred_pose).
           logits: (B, num_classes)
           pred_pose: (B, 126) - predicted joints for the NEXT frame
        """
        B, T, _ = hand_flat.shape

        # ── Fuse inputs ──────────────────────────────────────
        x = torch.cat([hand_flat, obj_rt], dim=-1)     # (B, T, 142)
        x = self.input_proj(x)                         # (B, T, d_model)

        # ── Prepend CLS token ─────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)               # (B, T+1, d_model)

        # ── Positional + obs-ratio encoding ──────────────────
        x = self.pos_enc(x, obs_ratio)                 # (B, T+1, d_model)

        # ── Transformer Encoder ───────────────────────────────
        x = self.encoder(x)                            # (B, T+1, d_model)

        # ── Heads from CLS token ────────────────────────────
        cls_out = x[:, 0, :]                           # (B, d_model)
        
        logits    = self.head(cls_out)                 # (B, num_classes)
        pred_pose = self.pose_head(cls_out)            # (B, 126)
        
        return logits, pred_pose

    # ── Convenience method for inference confidence ───────────
    @torch.no_grad()
    def predict_proba(
        self,
        hand_flat:  torch.Tensor,
        obj_rt:     torch.Tensor,
        obs_ratio:  torch.Tensor,
    ) -> torch.Tensor:
        """Returns softmax probabilities (B, num_classes)."""
        logits, _ = self(hand_flat, obj_rt, obs_ratio)
        return F.softmax(logits, dim=-1)

    # ── Number of parameters ──────────────────────────────────
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────
# Quick model summary
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = IntentFormer()
    print(f"IntentFormer — parameters: {model.num_parameters():,}")
    B, T = 4, 30
    logits, pred_pose = model(
        torch.randn(B, T, 126),
        torch.randn(B, T, 16),
        torch.rand(B),
    )
    print(f"Output logits: {logits.shape}")  # (4, 36)
    print(f"Output pose  : {pred_pose.shape}")  # (4, 126)
