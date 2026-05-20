"""
15_export_onnx_unity.py — GRU-ONLY export (ablation baseline).

WARNING: This script exports GRUBaseline, NOT IntentFormer.
         To export IntentFormer for Unity, use 13_export_onnx.py instead.

This script exists because Unity AI Inference does not support the ONNX GRU
operator. It re-exports a GRU checkpoint with the GRU cell unrolled into
Linear + sigmoid + tanh primitives that Unity accepts.

Usage:
    cd hot3d_exploration
    python 15_export_onnx_unity.py
    python 15_export_onnx_unity.py --checkpoint ../data/checkpoints/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx

F_IN       = 96
T_STEPS    = 16
TARGET_DIM = 78
HIDDEN     = 256
NUM_LAYERS = 2

OUT_ONNX = Path("../data/intentformer.onnx")


# ---------------------------------------------------------------------------
# Unrolled GRU cell — identical math to nn.GRU, no GRU ONNX op
# ---------------------------------------------------------------------------

class GRUCellUnrolled(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.H = hidden_size
        # Fused: [W_r; W_z; W_n] applied to input, then split
        self.lin_ih = nn.Linear(input_size, 3 * hidden_size)
        self.lin_hh = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        H  = self.H
        gi = self.lin_ih(x)   # [B, 3H]
        gh = self.lin_hh(h)   # [B, 3H]
        r  = torch.sigmoid(gi[:, :H]     + gh[:, :H])
        z  = torch.sigmoid(gi[:, H:2*H]  + gh[:, H:2*H])
        n  = torch.tanh(   gi[:, 2*H:]   + r * gh[:, 2*H:])
        return (1.0 - z) * n + z * h


class GRUBaselineUnrolled(nn.Module):
    """
    Mirrors GRUBaseline (bidirectional, 2-layer) using only primitives.
    T_STEPS is fixed at 16 so the loop fully unrolls in the ONNX graph.
    """
    def __init__(self, f_in=F_IN, hidden=HIDDEN, target_dim=TARGET_DIM):
        super().__init__()
        self.H = hidden
        # Layer 0 (input size = f_in)
        self.cell_fwd0 = GRUCellUnrolled(f_in, hidden)
        self.cell_bwd0 = GRUCellUnrolled(f_in, hidden)
        # Layer 1 (input size = 2*hidden — bidirectional layer 0 output)
        self.cell_fwd1 = GRUCellUnrolled(hidden * 2, hidden)
        self.cell_bwd1 = GRUCellUnrolled(hidden * 2, hidden)
        self.head = nn.Linear(hidden * 2, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F_IN]"""
        B  = x.size(0)
        H  = self.H
        h0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)

        # ---- Layer 0 forward ----
        h = h0
        fwd0 = []
        for t in range(T_STEPS):
            h = self.cell_fwd0(x[:, t, :], h)
            fwd0.append(h)

        # ---- Layer 0 backward ----
        h = h0
        bwd0 = [h0] * T_STEPS
        for t in range(T_STEPS - 1, -1, -1):
            h = self.cell_bwd0(x[:, t, :], h)
            bwd0[t] = h

        # Layer 0 output: [B, T, 2H]
        l0 = torch.stack(
            [torch.cat([fwd0[t], bwd0[t]], dim=-1) for t in range(T_STEPS)],
            dim=1,
        )

        # ---- Layer 1 forward ----
        h = h0
        fwd1 = []
        for t in range(T_STEPS):
            h = self.cell_fwd1(l0[:, t, :], h)
            fwd1.append(h)

        # ---- Layer 1 backward ----
        h = h0
        bwd1 = [h0] * T_STEPS
        for t in range(T_STEPS - 1, -1, -1):
            h = self.cell_bwd1(l0[:, t, :], h)
            bwd1[t] = h

        # Mirrors nn.GRU: out[:, -1, :] = cat(fwd_last, bwd_at_T-1)
        last = torch.cat([fwd1[-1], bwd1[-1]], dim=-1)  # [B, 2H]
        return self.head(last)


# ---------------------------------------------------------------------------
# Weight mapping from nn.GRU state_dict
# ---------------------------------------------------------------------------

def load_weights_from_gru(unrolled: GRUBaselineUnrolled, sd: dict):
    def copy_cell(cell: GRUCellUnrolled, ih_key: str, hh_key: str,
                  bih_key: str, bhh_key: str):
        cell.lin_ih.weight.data.copy_(sd[ih_key])
        cell.lin_ih.bias.data.copy_(sd[bih_key])
        cell.lin_hh.weight.data.copy_(sd[hh_key])
        cell.lin_hh.bias.data.copy_(sd[bhh_key])

    copy_cell(unrolled.cell_fwd0,
              "gru.weight_ih_l0",         "gru.weight_hh_l0",
              "gru.bias_ih_l0",           "gru.bias_hh_l0")
    copy_cell(unrolled.cell_bwd0,
              "gru.weight_ih_l0_reverse", "gru.weight_hh_l0_reverse",
              "gru.bias_ih_l0_reverse",   "gru.bias_hh_l0_reverse")
    copy_cell(unrolled.cell_fwd1,
              "gru.weight_ih_l1",         "gru.weight_hh_l1",
              "gru.bias_ih_l1",           "gru.bias_hh_l1")
    copy_cell(unrolled.cell_bwd1,
              "gru.weight_ih_l1_reverse", "gru.weight_hh_l1_reverse",
              "gru.bias_ih_l1_reverse",   "gru.bias_hh_l1_reverse")

    unrolled.head.weight.data.copy_(sd["head.weight"])
    unrolled.head.bias.data.copy_(sd["head.bias"])


# ---------------------------------------------------------------------------
# Numerical verification against original nn.GRU
# ---------------------------------------------------------------------------

def verify(unrolled: GRUBaselineUnrolled, sd: dict, tol: float = 1e-4) -> float:
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "intentformer_mod", Path(__file__).parent / "10_intentformer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["intentformer_mod"] = mod
    spec.loader.exec_module(mod)

    ref = mod.GRUBaseline().eval()
    ref.load_state_dict(sd)

    dummy = torch.randn(4, T_STEPS, F_IN)
    with torch.no_grad():
        out_ref     = ref(dummy).numpy()
        out_unrolled = unrolled(dummy).numpy()

    max_diff = float(np.abs(out_ref - out_unrolled).max())
    status   = "OK" if max_diff < tol else "LARGE — check weight mapping"
    print(f"  Max diff vs nn.GRU reference: {max_diff:.2e}  [{status}]")
    return max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../data/checkpoints/best.pt")
    ap.add_argument("--out",        default=str(OUT_ONNX))
    ap.add_argument("--no-verify",  action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.out)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["model"]

    print("Building unrolled model...")
    model = GRUBaselineUnrolled().eval()
    load_weights_from_gru(model, sd)

    if not args.no_verify:
        print("Verifying against original nn.GRU...")
        diff = verify(model, sd)
        if diff > 1e-4:
            raise RuntimeError("Numerical mismatch too large — aborting export.")

    dummy = torch.zeros(1, T_STEPS, F_IN)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=15,
            input_names=["features"],
            output_names=["pose"],
            dynamic_axes={"features": {0: "batch_size"}, "pose": {0: "batch_size"}},
            do_constant_folding=True,
            dynamo=False,
        )

    # Verify no GRU nodes remain
    m = onnx.load(str(out_path))
    gru_nodes = [n for n in m.graph.node if n.op_type == "GRU"]
    if gru_nodes:
        raise RuntimeError(f"GRU nodes still present after export: {gru_nodes}")

    ops = sorted(set(n.op_type for n in m.graph.node))
    print(f"Operators in exported model: {ops}")
    print(f"GRU nodes remaining: 0  [OK]")
    print(f"Model size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
