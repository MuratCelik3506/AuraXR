"""
13_export_onnx.py — Export trained IntentFormer to ONNX for Unity Sentis.

Produces:
  data/intentformer.onnx       ONNX opset 17 (Unity Sentis compatible)
  data/intentformer_meta.json  Normalisation stats + dim info for the Unity runtime

The ONNX model:
  Input  name: "features"   shape: [1, 16, 96]   dtype: float32
  Output name: "pose"       shape: [1, 78]        dtype: float32

Unity Sentis integration:
  1. Import intentformer.onnx into Unity via Assets window
  2. Load intentformer_meta.json at runtime to normalise inputs
  3. Feed T=16 × 96 feature tensor
  4. Read 78-dim output and decode using predict_hand_params layout:
       [0:15]   mano_pose_h0   — MANO θ (15 joint angles)
       [15:25]  mano_betas_h0  — MANO β (10 shape params)
       [25:28]  wrist_t_h0     — wrist world position (metres)
       [28:32]  wrist_q_h0     — wrist quaternion (w,x,y,z)
       [32:35]  delta_t_h0     — controller→wrist offset (metres)
       [35:39]  delta_q_h0     — controller→wrist quaternion
       [39:78]  same for hand 1

Usage:
    python 13_export_onnx.py
    python 13_export_onnx.py --checkpoint data/checkpoints/best.pt
    python 13_export_onnx.py --validate    # run inference check after export
"""

import argparse
import json
from pathlib import Path

import importlib.util
import sys
from pathlib import Path as _Path

import numpy as np
import torch

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_im          = _load_module("intentformer_mod3", _Path(__file__).parent / "10_intentformer.py")
IntentFormer  = _im.IntentFormer
GRUBaseline   = _im.GRUBaseline
SingleFrameMLP = _im.SingleFrameMLP
F_IN          = _im.F_IN
T             = _im.T
TARGET_DIM    = _im.TARGET_DIM

DATA_FILE = Path("../data/hot3d_training.h5")
CKPT_DIR  = Path("../data/checkpoints")
OUT_ONNX  = Path("../data/intentformer.onnx")
OUT_META  = Path("../data/intentformer_meta.json")


def load_norm_stats(hf_path: Path) -> dict:
    """Read normalisation stats from the HDF5 training file."""
    try:
        import h5py
        with h5py.File(str(hf_path), "r") as hf:
            return json.loads(hf.attrs["meta"])
    except Exception:
        return {}


def _build_model(name: str):
    if name == "gru":
        return GRUBaseline()
    elif name == "mlp":
        return SingleFrameMLP()
    else:
        return IntentFormer()


def export(args):
    device = torch.device("cpu")  # ONNX export always on CPU

    ckpt_path  = Path(args.checkpoint)
    model_name = args.model
    ckpt = None

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if model_name == "auto":
            model_name = ckpt.get("args", {}).get("model", "intentformer")
            print(f"[Auto-detect] model={model_name}")
    elif model_name == "auto":
        model_name = "intentformer"

    model = _build_model(model_name).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded: {ckpt_path}  (epoch={ckpt.get('epoch','?')}, "
              f"best_mpjpe={ckpt.get('best_mpjpe', float('nan')):.2f} mm)")
    else:
        print(f"[WARN] {ckpt_path} not found — exporting random-weight model.")

    model.eval()

    # Dummy input: batch=1, T=16, F_IN=96
    dummy = torch.zeros(1, T, F_IN)

    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)

    # PyTorch 2.9+ dispatches transformer layers through fused C++ kernels
    # (_transformer_encoder_layer_fwd, _native_multi_head_attention) that the
    # legacy TorchScript exporter (dynamo=False) cannot decompose into ONNX ops.
    # The dynamo exporter uses torch.export which fully decomposes all ops.
    # At ~21 MB the model is well under the 2 GB external-data threshold,
    # so the output is a single self-contained .onnx file.
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(OUT_ONNX),
            opset_version=17,
            input_names=["features"],
            output_names=["pose"],
            dynamo=True,
        )

    # dynamo exporter always writes weights to a .onnx.data sidecar.
    # Unity AI Inference requires a single self-contained file — merge here.
    sidecar = Path(str(OUT_ONNX) + ".data")
    if sidecar.exists():
        import onnx as _onnx
        merged = _onnx.load(str(OUT_ONNX), load_external_data=True)
        _onnx.save(merged, str(OUT_ONNX), save_as_external_data=False)
        sidecar.unlink()
    print(f"Exported: {OUT_ONNX}")

    # Companion metadata for Unity runtime
    norm = load_norm_stats(DATA_FILE)
    meta = {
        "model_name":  "IntentFormer",
        "input_name":  "features",
        "output_name": "pose",
        "T":           T,
        "feature_dim": F_IN,
        "target_dim":  TARGET_DIM,
        "feature_mean": norm.get("feature_mean", [0.0] * F_IN),
        "feature_std":  norm.get("feature_std",  [1.0] * F_IN),
        "target_mean":  norm.get("target_mean",  [0.0] * TARGET_DIM),
        "target_std":   norm.get("target_std",   [1.0] * TARGET_DIM),
        "output_layout": {
            "mano_pose_h0":  [0,  15],
            "mano_betas_h0": [15, 25],
            "wrist_t_h0":    [25, 28],
            "wrist_q_h0":    [28, 32],
            "delta_t_h0":    [32, 35],
            "delta_q_h0":    [35, 39],
            "mano_pose_h1":  [39, 54],
            "mano_betas_h1": [54, 64],
            "wrist_t_h1":    [64, 67],
            "wrist_q_h1":    [67, 71],
            "delta_t_h1":    [71, 74],
            "delta_q_h1":    [74, 78],
        },
        "notes": (
            "delta_t is in metres, controller-to-wrist offset. "
            "delta_q is quaternion (w,x,y,z). "
            "Apply target_mean/target_std de-normalisation after inference. "
            "Quaternions require re-normalisation (L2 norm) after de-normalisation."
        ),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {OUT_META}")

    # File size
    size_mb = OUT_ONNX.stat().st_size / 1e6
    print(f"Model size: {size_mb:.1f} MB  (target < 50 MB for Quest 3)")

    # Validation
    if args.validate:
        print("\nValidating ONNX export...")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(OUT_ONNX))
            dummy_np = dummy.numpy()
            outputs  = sess.run(None, {"features": dummy_np})
            pose_out = outputs[0]
            assert pose_out.shape == (1, TARGET_DIM), \
                f"Unexpected shape: {pose_out.shape}"
            print(f"  ONNX Runtime output shape: {list(pose_out.shape)}  ✓")
            print(f"  Output range: [{pose_out.min():.3f}, {pose_out.max():.3f}]")
        except ImportError:
            print("  [INFO] onnxruntime not installed — skipping runtime validation.")
            print("  Install with: pip install onnxruntime")

        # PyTorch vs ONNX consistency check
        with torch.no_grad():
            torch_out = model(dummy).numpy()
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(OUT_ONNX))
            onnx_out = sess.run(None, {"features": dummy.numpy()})[0]
            max_diff = float(np.abs(torch_out - onnx_out).max())
            print(f"  Max PyTorch vs ONNX diff: {max_diff:.2e}  "
                  f"({'✓ OK' if max_diff < 1e-4 else '✗ LARGE — check model'})")
        except Exception:
            pass

    print(f"\n[DONE] {OUT_ONNX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--model",      type=str, default="auto",
                    choices=["auto", "intentformer", "gru", "mlp"],
                    help="Model architecture (default: auto-detect from checkpoint)")
    ap.add_argument("--validate",   action="store_true",
                    help="Run ONNX Runtime consistency check after export")
    args = ap.parse_args()
    export(args)


if __name__ == "__main__":
    main()
