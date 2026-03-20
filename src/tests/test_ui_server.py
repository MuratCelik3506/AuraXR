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
from src.data.h2o_dataset import NUM_CLASSES
from src.evaluate import compute_ttc

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

# ── presets: (description, obs_ratio, gesture_hint) ────────────────────────
PRESETS = {
    "random":   {"label": "🎲 Random Noise",         "obs_ratio": 0.25},
    "grasp":    {"label": "✋ Grasping Motion",       "obs_ratio": 0.20},
    "pour":     {"label": "🫗 Pouring Motion",        "obs_ratio": 0.30},
    "rest":     {"label": "🖐 Hand at Rest",          "obs_ratio": 0.20},
    "reach":    {"label": "👆 Reaching Motion",       "obs_ratio": 0.25},
    "pinch":    {"label": "🤌 Pinch Grasp",           "obs_ratio": 0.20},
}

GHOSTING_THRESHOLD = 0.65   # mirrors instruction.md §5

# ──────────────────────────────────────────────────────────────────────────────
# Pose generators for each preset
# ──────────────────────────────────────────────────────────────────────────────

def _make_hand_flat(preset: str, window_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return (window_size, 126) float32 hand features for the given preset."""
    T, D = window_size, 126

    if preset == "random":
        return rng.standard_normal((T, D)).astype(np.float32) * 0.1

    elif preset == "grasp":
        # Fingers curl progressively toward the palm
        flat = np.zeros((T, D), dtype=np.float32)
        for t in range(T):
            progress = t / max(T - 1, 1)
            for hand in range(2):
                base = hand * 63
                # Wrist stays at 0 (wrist-relative)
                for j in range(1, 21):
                    # Finger tips move toward palm
                    flat[t, base + j * 3]     =  0.05 * j * (1 - progress)
                    flat[t, base + j * 3 + 1] = -0.02 * j * progress
                    flat[t, base + j * 3 + 2] =  0.01 * j
        return flat + rng.standard_normal((T, D)).astype(np.float32) * 0.005

    elif preset == "pour":
        flat = np.zeros((T, D), dtype=np.float32)
        for t in range(T):
            progress = t / max(T - 1, 1)
            # Wrist rotates (simulate tilt)
            tilt = progress * 0.3
            for j in range(21):
                flat[t, j * 3 + 1] = tilt * j * 0.01
                flat[t, j * 3 + 2] = progress * 0.05
        return flat + rng.standard_normal((T, D)).astype(np.float32) * 0.005

    elif preset == "rest":
        # Fingers slightly spread, no motion
        flat = np.zeros((T, D), dtype=np.float32)
        for j in range(1, 21):
            spread = (j % 5) * 0.02
            flat[:, j * 3]     = spread
            flat[:, j * 3 + 1] = 0.05
            flat[:, j * 3 + 2] = 0.0
        return flat + rng.standard_normal((T, D)).astype(np.float32) * 0.002

    elif preset == "reach":
        flat = np.zeros((T, D), dtype=np.float32)
        for t in range(T):
            progress = t / max(T - 1, 1)
            # Hand extends forward
            flat[t, 2] = progress * 0.3         # z-axis extension
            for j in range(1, 21):
                flat[t, j * 3 + 2] = progress * 0.02 * j
        return flat + rng.standard_normal((T, D)).astype(np.float32) * 0.005

    elif preset == "pinch":
        flat = np.zeros((T, D), dtype=np.float32)
        for t in range(T):
            progress = t / max(T - 1, 1)
            # Thumb (joints 1-4) and index (joints 5-8) converge
            for j in range(1, 5):    # thumb
                flat[t, j * 3]     = 0.05 * (1 - progress)
                flat[t, j * 3 + 1] = -0.02 * progress
            for j in range(5, 9):   # index
                flat[t, j * 3]     = -0.05 * (1 - progress)
                flat[t, j * 3 + 1] = -0.02 * progress
        return flat + rng.standard_normal((T, D)).astype(np.float32) * 0.003

    return rng.standard_normal((T, D)).astype(np.float32) * 0.1


def _make_obj_rt(window_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return (window_size, 16) float32 object RT matrix."""
    base = np.eye(4, dtype=np.float32).flatten()
    obj  = np.tile(base, (window_size, 1))
    obj += rng.standard_normal((window_size, 16)).astype(np.float32) * 0.01
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

_model:  IntentFormer | None = None
_device: torch.device         = torch.device("cpu")


def get_model() -> IntentFormer:
    global _model
    if _model is None:
        _model = IntentFormer(num_classes=NUM_CLASSES)
        _model.eval()
        _model.to(_device)
    return _model


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/presets", methods=["GET"])
def api_presets():
    return jsonify(PRESETS)


@app.route("/api/action_names", methods=["GET"])
def api_action_names():
    return jsonify(ACTION_NAMES)


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """
    Body (JSON):
      preset      : str            one of 'random'|'grasp'|'pour'|'rest'|'reach'|'pinch'
      obs_ratio   : float          0.10 – 0.50
      window_size : int            10 – 60
      seed        : int            random seed (reproducibility)
      start_act   : int            action start frame (for TTC)
      end_act     : int            action end frame   (for TTC)
      hand_flat   : list[float]    optional: flatten(window_size×126) custom values
      obj_rt      : list[float]    optional: flatten(window_size×16)  custom values
    """
    body = request.get_json(force=True)

    preset      = body.get("preset", "random")
    obs_ratio   = float(body.get("obs_ratio", 0.25))
    window_size = int(body.get("window_size", 30))
    seed        = int(body.get("seed", 42))
    start_act   = int(body.get("start_act", 20))
    end_act     = int(body.get("end_act", 80))

    rng = np.random.default_rng(seed)

    # ── Build tensors ────────────────────────────────────────────────────────
    if "hand_flat" in body and body["hand_flat"]:
        hf = np.array(body["hand_flat"], dtype=np.float32)
        hf = hf.reshape(window_size, 126)
    else:
        hf = _make_hand_flat(preset, window_size, rng)

    if "obj_rt" in body and body["obj_rt"]:
        obj = np.array(body["obj_rt"], dtype=np.float32)
        obj = obj.reshape(window_size, 16)
    else:
        obj = _make_obj_rt(window_size, rng)

    model = get_model()

    hand_t = torch.from_numpy(hf).unsqueeze(0).to(_device)     # (1, T, 126)
    obj_t  = torch.from_numpy(obj).unsqueeze(0).to(_device)    # (1, T,  16)
    obs_t  = torch.tensor([obs_ratio], dtype=torch.float32).to(_device)

    with torch.no_grad():
        logits = model(hand_t, obj_t, obs_t)           # (1, 36)
        proba  = F.softmax(logits, dim=-1)[0]          # (36,)

    proba_np        = proba.cpu().numpy().tolist()
    pred_class      = int(np.argmax(proba_np))
    max_conf        = float(proba_np[pred_class])
    ghost_triggered = max_conf >= GHOSTING_THRESHOLD

    # Top-5
    top5_idx = np.argsort(proba_np)[::-1][:5].tolist()
    top5 = [
        {
            "rank":       i + 1,
            "class_idx":  idx,
            "class_name": ACTION_NAMES[idx],
            "prob":       round(proba_np[idx], 4),
        }
        for i, idx in enumerate(top5_idx)
    ]

    # TTC
    ttc_seconds = compute_ttc(start_act, end_act, obs_ratio, fps=30)

    return jsonify({
        "pred_class":      pred_class,
        "pred_name":       ACTION_NAMES[pred_class],
        "confidence":      round(max_conf, 4),
        "ghost_triggered": ghost_triggered,
        "top5":            top5,
        "ttc_seconds":     round(ttc_seconds, 3),
        "obs_ratio":       obs_ratio,
        "window_size":     window_size,
        "all_probs":       [round(p, 4) for p in proba_np],
    })


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
        m = get_model()
        m.load_state_dict(state)
        print(f"[server] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[server] Using random (untrained) model weights.")
        print("[server] Pass --checkpoint checkpoints/best_model.pt for trained predictions.")

    print(f"\n[server] ✓  Open your browser:  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
