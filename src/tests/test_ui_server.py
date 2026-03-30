"""
Interactive Single-Sample Test Server
=======================================
Runs a local web server that lets you define inputs manually
(obs_ratio, hand joints, object pose) and run a single forward
pass through IntentFormer to see predictions, confidence scores,
and ghosting trigger status.

Usage:
    python -m src.tests.test_ui_server
    # or with a pre-trained checkpoint:
    python -m src.tests.test_ui_server --checkpoint checkpoints/best_model.pt

Then open: http://localhost:5001
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # Phase1/
sys.path.insert(0, str(ROOT))

from src.models.intent_former import IntentFormer
from src.data.h2o_dataset import NUM_CLASSES, H2ODataset
from src.evaluate import compute_ttc
from src.sliding_window_eval import run_sliding_window

# ── H2O action-class names (36 verb-object combinations) ────────────────────
ACTION_NAMES = [
    "grab bottle",        "grab cup",           "grab knife",
    "grab liquid soap",   "grab milk",          "grab plate",
    "pour bottle",        "pour cup",           "pour liquid soap",
    "pour milk",          "put bottle",         "put cup",
    "put knife",          "put plate",          "drink bottle",
    "drink cup",          "drink milk",         "use knife",
    "use liquid soap",    "cut plate",          "stir cup",
    "squeeze liquid soap","open bottle",        "open liquid soap",
    "open milk",          "apply liquid soap",  "read bottle",
    "spray liquid soap",  "lay bottle",         "lay cup",
    "lay knife",          "lay plate",          "peel plate",
    "drink milk alt",     "grab milk alt",      "use plate",
]

GHOSTING_THRESHOLD = 0.65   # mirrors instruction.md §5

# (Removed synthetic generators: _make_hand_flat, _make_obj_rt)


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

_model:  IntentFormer | None = None
_device: torch.device         = torch.device("cpu")


def get_model(ckpt_params: dict = None) -> IntentFormer:
    global _model
    if _model is None:
        if ckpt_params:
            # Dynamically use checkpoint parameters
            _model = IntentFormer(
                input_dim       = ckpt_params.get("input_dim", 142),
                num_classes     = ckpt_params.get("num_classes", NUM_CLASSES),
                d_model         = ckpt_params.get("d_model", 128),
                nhead           = ckpt_params.get("nhead", 4),
                num_layers      = ckpt_params.get("num_layers", 4),
                dim_feedforward = ckpt_params.get("dim_ff", 512),
                window_size     = ckpt_params.get("window_size", 30)
            )
        else:
            _model = IntentFormer(input_dim=142, num_classes=NUM_CLASSES)
        
        _model.eval()
        _model.to(_device)
    return _model


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/action_names", methods=["GET"])
def api_action_names():
    return jsonify(ACTION_NAMES)

def discover_extra_samples(root_dir):
    """
    Scans the annotations directory to find all available clips that have hand_pose data.
    Ensures we aren't limited by small placeholder split files.
    """
    extra = []
    root = Path(root_dir)
    anno_root = root / "annotations"
    if not anno_root.exists():
        return []

    # subjectN / RIG / TAKE / cam4
    for cam4_path in anno_root.glob("subject*/h*/[0-9]*/cam4"):
        rel_path = str(cam4_path.parent.relative_to(anno_root))
        
        # Check if hand_pose exists
        hp_dir = cam4_path / "hand_pose"
        if not hp_dir.exists():
            continue
            
        # Try to infer label from action_label
        label = 0
        al_dir = cam4_path / "action_label"
        if al_dir.exists():
            al_files = sorted(list(al_dir.glob("[0-9][0-9][0-9][0-9].txt")))
            if not al_files:
                al_files = sorted(list(al_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].txt")))
            
            if al_files:
                try:
                    label = int(open(al_files[0]).read().strip()) - 1
                    label = max(0, min(label, NUM_CLASSES - 1))
                except:
                    label = 0
        
        extra.append({
            "rel_path":     rel_path,
            "start_act":    10,  # Default range for discovery
            "end_act":      100, # Will be clipped by loader if too long
            "label":        label,
            "obs_ratio":    0.25,
        })
    return extra


_h2o_ds = None

def get_h2o_dataset():
    global _h2o_ds
    if _h2o_ds is None:
        try:
            # Combine all samples from train, val, and test splits for the UI
            all_samples = []
            base_ds = None
            for split in ["train", "val", "test"]:
                try:
                    ds = H2ODataset("data/h2o", split=split, window_size=30)
                    if ds and len(ds.samples) > 0:
                        all_samples.extend(ds.samples)
                        if base_ds is None:
                            base_ds = ds
                except Exception:
                    continue
            
            # Fallback: discover everything in annotations if split files are small
            if len(all_samples) < 50:
                print(f"[server] Split files limited ({len(all_samples)}). Discovering all clips...")
                extra = discover_extra_samples("data/h2o")
                # Merge and avoid exact duplicates
                seen = set((s["rel_path"], s["start_act"], s["end_act"]) for s in all_samples)
                for s in extra:
                    key = (s["rel_path"], s["start_act"], s["end_act"])
                    if key not in seen:
                        all_samples.append(s)
                        seen.add(key)
            
            if base_ds and all_samples:
                _h2o_ds = base_ds
                _h2o_ds.samples = all_samples
                print(f"[server] Dataset loaded: {len(all_samples)} total samples.")
            elif not base_ds:
                # Absolute fallback if no split file exists at all
                print("[server] Critical: No split files found. Using discovery only.")
                # We need a dummy H2ODataset instance to use its methods
                # Creating one with split='train' just to get the instance, it might fail __init__ if split_file missing
                # So we manually create a minimal ds-like object if needed, but H2ODataset is robust enough if split_file exists.
                # Try creating a dummy one
                try:
                    _h2o_ds = H2ODataset("data/h2o", split="train", window_size=30)
                    _h2o_ds.samples = discover_extra_samples("data/h2o")
                except:
                    pass
            else:
                print("[server] Warning: No dataset samples found in split files.")
        except Exception as e:
            print(f"[server] Dataset aggregate error: {e}")
            _h2o_ds = None
    return _h2o_ds

@app.route("/api/real_data", methods=["GET"])
def api_real_data():
    ds = get_h2o_dataset()
    if ds is None or len(ds.samples) == 0:
        return jsonify([])
    
    seen = set()
    out = []
    for i, s in enumerate(ds.samples):
        key = (s["rel_path"], s["start_act"], s["end_act"])
        if key not in seen:
            seen.add(key)
            out.append({
                "idx": i,
                "path": s["rel_path"],
                "start_act": s["start_act"],
                "end_act": s["end_act"],
                "label": int(s["label"]),
                "action_name": ACTION_NAMES[int(s["label"])],
            })
            if len(out) >= 100:
                break
    return jsonify(out)

@app.route("/api/real_data/<int:idx>", methods=["GET"])
def api_get_real_data(idx):
    ds = get_h2o_dataset()
    if ds is None or idx < 0 or idx >= len(ds.samples):
        return jsonify({"error": "Invalid index"}), 404
        
    meta = ds.samples[idx]
    seq_data = ds._get_sequence(meta["rel_path"])
    if seq_data is None:
         return jsonify({"error": "Failed to load sequence"}), 404
         
    s = meta["start_act"]
    e = meta["end_act"]
    num_frames = seq_data["num_frames"]
    # Optional: return full sequence instead of just the action segment
    is_full = request.args.get("full", "0") == "1"
    
    if is_full:
        s = 0
        e = num_frames - 1
    else:
        s = max(0, min(s, num_frames - 1))
        e = max(0, min(e, num_frames - 1))
        if e < s: e = s
    
    hand_poses = seq_data["hand_poses"][s:e+1]
    raw_hand_poses = seq_data.get("raw_hand_poses", hand_poses)[s:e+1]
    obj_poses = seq_data["obj_poses_rt"][s:e+1]
    
    T = hand_poses.shape[0]
    hand_flat = hand_poses.reshape(T, 126).tolist()
    raw_hand_flat = raw_hand_poses.reshape(T, 126).tolist()
    obj_flat = obj_poses.tolist()
    
    return jsonify({
        "hand_flat": hand_flat,
        "raw_hand_flat": raw_hand_flat,
        "obj_rt": obj_flat,
        "true_label": int(meta["label"]),
        "action_name": ACTION_NAMES[int(meta["label"])],
        "T": T,
        "start_act": s,
        "end_act": e
    })

@app.route("/api/sliding_window/<int:idx>", methods=["GET"])
def api_sliding_window(idx):
    ds = get_h2o_dataset()
    if ds is None or idx < 0 or idx >= len(ds.samples):
        return jsonify({"error": "Invalid index"}), 404
        
    meta = ds.samples[idx]
    seq_data = ds._get_sequence(meta["rel_path"])
    if seq_data is None:
         return jsonify({"error": "Failed to load sequence"}), 404

    model = get_model()
    
    # Read parameters from UI if provided, otherwise default to full sequence from start
    buffer_frames = int(request.args.get("buffer", 0))
    sim_frames    = int(request.args.get("sim", 9999))
    obs_limit     = float(request.args.get("obs_limit", 1.0))
    
    res = run_sliding_window(
        model, 
        seq_data, 
        _device, 
        window_size=30,
        buffer_frames=buffer_frames,
        sim_frames=sim_frames,
        obs_limit=obs_limit
    )
    
    return jsonify({
        "frame_wise_accuracy": round(res["frame_wise_accuracy"], 4),
        "stability":           round(res["stability"], 4),
        "convergence_point":   round(res["convergence_point"], 2),
        "predictions":         res["predictions"],
        "ground_truth":        res["ground_truth"],
        "max_probs":           res["max_probs"],
        "pose_errors":         [round(float(e), 5) for e in res["pose_errors"]],
        "predicted_poses":     res["predicted_poses"],
        "gt_poses":            res["gt_poses"],
        "gt_obj_poses":        res["gt_obj_poses"],
        "mean_pose_error":     round(res["mean_pose_error"], 5),
        "path":                meta["rel_path"]
    })

# /api/infer has been removed (was used by manual/synthetic input tabs)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Interactive IntentFormer test UI")
    p.add_argument("--checkpoint", default="",
                   help="Path to .pt checkpoint (optional — uses random weights if empty)")
    p.add_argument("--port",  type=int, default=5001)
    p.add_argument("--host",  default="127.0.0.1")
    return p.parse_args()


def main():
    global _device
    args = parse_args()

    # Device
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
        print("[server] Device: Apple MPS")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
        print("[server] Device: CUDA")
    else:
        _device = torch.device("cpu")
        print("[server] Device: CPU")

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=_device)
        state = ckpt.get("model", ckpt)
        
        # Detect input_dim from checkpoint weight shape
        lnorm_key = "input_proj.0.weight"   # LayerNorm weight => shape (input_dim,)
        linear_key = "input_proj.1.weight"  # Linear weight => shape (d_model, input_dim)
        if lnorm_key in state:
            detected_input_dim = state[lnorm_key].shape[0]
        elif linear_key in state:
            detected_input_dim = state[linear_key].shape[1]
        else:
            detected_input_dim = 142  # safe fallback
        
        # Detect d_model from linear bias or weight
        if linear_key in state:
            detected_d_model = state[linear_key].shape[0]
        else:
            detected_d_model = 128

        # Determine model params from checkpoint
        # Ultimate run: d_model=512, nhead=16, num_layers=8, dim_ff=2048
        # Optimized v3: d_model=256, nhead=8, num_layers=6, dim_ff=1024
        is_ultimate = "v3_ultimate" in args.checkpoint
        is_v3 = "optimized_v3" in args.checkpoint or "combined_best" in args.checkpoint

        ckpt_params = {
            "input_dim":   detected_input_dim,
            "num_classes": ckpt.get("num_classes", NUM_CLASSES),
            "d_model":     detected_d_model,
            "nhead":       16 if is_ultimate else (8 if is_v3 else 4),
            "num_layers":  8 if is_ultimate else (6 if is_v3 else 4),
            "dim_ff":      2048 if is_ultimate else (1024 if is_v3 else 512),
        }
        
        m = get_model(ckpt_params)
        m.load_state_dict(state, strict=False)
        print(f"[server] Loaded checkpoint: {args.checkpoint}")
        print(f"[server] Model Config: input_dim={detected_input_dim} d_model={ckpt_params['d_model']} classes={ckpt_params['num_classes']}")
    else:
        print("[server] Using random (untrained) model weights.")
        get_model() # Init default model
        print("[server] Pass --checkpoint checkpoints/best_model.pt for trained predictions.")

# Terminal output simplification
    print(f"\n[server] ✓ AuraXR Dashboard ready at: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
