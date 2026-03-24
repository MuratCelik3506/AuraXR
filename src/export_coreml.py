"""
CoreML Deployment Pipeline — Intent-Aware XR Framework
======================================================

This module converts a trained PyTorch (.pt) model into an Apple CoreML 
(.mlpackage) format, specifically optimized for the Apple Neural Engine (ANE).

The Goal:
---------
Running the intent prediction model on the ANE ensures that the GPU remains 
fully available for high-frame-rate Unity rendering in VR/AR, while the 
prediction runs in the background with zero lag.

The Problem with Transformers & CoreML:
--------------------------------------
Standard PyTorch Transformer layers use fused C++ kernels (like 
`scaled_dot_product_attention`) which are not lowerable to the ANE as of 
coremltools 7.x. 

The Solution (Unfused Transformer):
----------------------------------
This script performs a "weight transplant":
1. It defines a custom `UnfusedMHA` and `UnfusedEncoderLayer` using 
   only primitive tensor ops (matrix multiplication, softmax, etc.).
2. It rebuilds the model's architecture using these unfused ops.
3. It copies the weights from the trained checkpoint into this new structure.
4. It traces the result for CoreML conversion.

Usage:
------
    python -m src.export_coreml \\
        --checkpoint checkpoints/best_model.pt \\
        --output     build/IntentFormer.mlpackage \\
        --window_size 30
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Unfused Multi-Head Attention
#   – avoids `_transformer_encoder_layer_fwd` / `scaled_dot_product_attention`
#     which coremltools cannot lower as of coremltools 7.x
# ─────────────────────────────────────────────────────────

class UnfusedMHA(nn.Module):
    """Explicit Q/K/V projection + manual scaled dot-product attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        Q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, H, Dh).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class UnfusedEncoderLayer(nn.Module):
    """Pre-LN encoder layer using only primitive ops."""

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = UnfusedMHA(d_model, nhead, dropout=dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class UnfusedTransformerEncoder(nn.Module):
    def __init__(self, layers, norm):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm   = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────
# Weight transplant
# ─────────────────────────────────────────────────────────

def rebuild_encoder_unfused(model, d_model, nhead, num_layers,
                             dim_feedforward, dropout=0.0):
    """
    Replace model.encoder with UnfusedTransformerEncoder.
    Copies all weights from the original nn.TransformerEncoder layers.
    """
    unfused_layers = []
    for orig in model.encoder.layers:
        ufl = UnfusedEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        # in_proj_weight layout: [Q | K | V] each (D, D)
        W = orig.self_attn.in_proj_weight   # (3D, D)
        b = orig.self_attn.in_proj_bias     # (3D,)
        ufl.attn.q_proj.weight.data.copy_(W[: d_model])
        ufl.attn.k_proj.weight.data.copy_(W[d_model : 2 * d_model])
        ufl.attn.v_proj.weight.data.copy_(W[2 * d_model :])
        ufl.attn.q_proj.bias.data.copy_(b[: d_model])
        ufl.attn.k_proj.bias.data.copy_(b[d_model : 2 * d_model])
        ufl.attn.v_proj.bias.data.copy_(b[2 * d_model :])
        ufl.attn.out_proj.weight.data.copy_(orig.self_attn.out_proj.weight)
        ufl.attn.out_proj.bias.data.copy_(orig.self_attn.out_proj.bias)

        ufl.ff[0].weight.data.copy_(orig.linear1.weight)
        ufl.ff[0].bias.data.copy_(orig.linear1.bias)
        ufl.ff[3].weight.data.copy_(orig.linear2.weight)
        ufl.ff[3].bias.data.copy_(orig.linear2.bias)

        ufl.norm1.weight.data.copy_(orig.norm1.weight)
        ufl.norm1.bias.data.copy_(orig.norm1.bias)
        ufl.norm2.weight.data.copy_(orig.norm2.weight)
        ufl.norm2.bias.data.copy_(orig.norm2.bias)

        unfused_layers.append(ufl)

    model.encoder = UnfusedTransformerEncoder(unfused_layers, model.encoder.norm)
    return model


# ─────────────────────────────────────────────────────────
# Export wrapper
# ─────────────────────────────────────────────────────────

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hand_flat: torch.Tensor,
                obj_rt: torch.Tensor,
                obs_ratio: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(hand_flat, obj_rt, obs_ratio)
        return logits


# ─────────────────────────────────────────────────────────
# Main export function
# ─────────────────────────────────────────────────────────

def export_coreml(args):
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("Run: pip install coremltools")

    from src.models.intent_former import IntentFormer
    from src.data.h2o_dataset import NUM_CLASSES

    T        = args.window_size
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────
    print(f"[export] Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    
    # Try to extract num_classes from checkpoint, fallback to H2O if missing
    num_classes = ckpt.get("num_classes", 36)
    dataset_name = ckpt.get("dataset", "Unknown")
    
    model = IntentFormer(
        input_dim=126 + 16, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        num_classes=num_classes, window_size=T, dropout=0.0,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Rebuild encoder ───────────────────────────────────
    print(f"[export] Dataset={dataset_name}, Classes={num_classes}")
    print("[export] Rebuilding encoder with unfused (CoreML-compatible) ops...")
    model = rebuild_encoder_unfused(
        model, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
    )
    model.eval()

    example_hand = torch.zeros(1, T, 126)
    example_obj  = torch.zeros(1, T, 16)
    example_obs  = torch.tensor([0.25])

    wrapper = ExportWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        test_out = wrapper(example_hand, example_obj, example_obs)
    print(f"[export] Forward check: {test_out.shape}  (expected (1,{num_classes}))  ✓")

    # ── Trace ─────────────────────────────────────────────
    print("[export] TorchScript tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_hand, example_obj, example_obs))

    # ── Convert ───────────────────────────────────────────
    print("[export] Converting to CoreML (may take ~60s)...")
    ml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hand_flat", shape=(1, T, 126)),
            ct.TensorType(name="obj_rt",    shape=(1, T, 16)),
            ct.TensorType(name="obs_ratio", shape=(1,)),
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    # ── Metadata ──────────────────────────────────────────
    ml_model.short_description = (
        f"IntentFormer: Early Action Prediction ({dataset_name}, {num_classes} classes)"
    )
    ml_model.input_description["hand_flat"] = (
        "Wrist-relative 3-D joints (both hands). Shape (1, T, 126)."
    )
    ml_model.input_description["obj_rt"]    = (
        "Flattened 4x4 RT matrix of target object. Shape (1, T, 16)."
    )
    ml_model.input_description["obs_ratio"] = (
        "Fraction of action observed, in [0,1]. Shape (1,)."
    )
    ml_model.output_description["logits"]   = (
        f"Raw per-class logits ({num_classes}). Apply softmax for confidence."
    )

    ml_model.save(str(out_path))
    print(f"\n[export] ✓  Saved: {out_path}")
    print(f"[export]    Compute units : ALL (CPU + GPU + ANE)")
    print(f"[export]    Window size   : {T} frames")
    print(f"[export]    Classes       : {num_classes}")

    # ── Validate ──────────────────────────────────────────
    print("[export] Running CoreML prediction for validation...")
    pred = ml_model.predict({
        "hand_flat": example_hand.numpy(),
        "obj_rt":    example_obj.numpy(),
        "obs_ratio": example_obs.numpy(),
    })
    print(f"[export] Output shape: {pred['logits'].shape}  ← expected (1,{num_classes})  ✓")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Export IntentFormer to CoreML")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output",      default="build/IntentFormer.mlpackage")
    p.add_argument("--window_size", type=int, default=30)
    p.add_argument("--d_model",     type=int, default=128)
    p.add_argument("--nhead",       type=int, default=4)
    p.add_argument("--num_layers",  type=int, default=4)
    p.add_argument("--dim_ff",      type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    export_coreml(parse_args())
