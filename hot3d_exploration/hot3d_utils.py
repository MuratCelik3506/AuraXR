"""hot3d_utils.py — Shared utilities for HOT3D exploration scripts."""

import json
from pathlib import Path

import numpy as np

CLIP_REPO = "bop-benchmark/hot3d"

TRAINING_FPS = 30
INFERENCE_HZ = 72
T_CANDIDATES = [4, 8, 16, 24, 32, 38, 48, 64]
MAX_PLOT_OBJECTS = 20

# MANO key name variants (try in order until one is found)
BETA_KEYS   = ("betas", "shape", "beta")
TRANSL_KEYS = ("transl", "translation", "wrist_pos", "global_position")
ORIENT_KEYS = ("global_orient", "wrist_orient", "orientation", "global_rotation")


def decode_json(raw):
    if isinstance(raw, (bytes, bytearray)):
        return json.loads(raw.decode())
    return raw


def load_hot3d(split: str = "train", streaming: bool = True):
    try:
        from datasets import load_dataset
        return load_dataset(CLIP_REPO, split=split, streaming=streaming)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("[HINT] huggingface-cli login")
        raise


def ensure_output_dir(path: str = "output") -> Path:
    p = Path(path)
    p.mkdir(exist_ok=True)
    return p


def first_value(d: dict, keys: tuple):
    """Return the first non-None value from d matching any key in keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None
