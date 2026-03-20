"""
Performance Benchmarking — Intent-Aware XR Framework
====================================================

This script profiles the IntentFormer model on Apple Silicon (M-series) 
hardware using the Metal Performance Shaders (MPS) backend.

The Goal:
---------
To ensure that the model meets real-time latency requirements for VR/AR. 
For a 90Hz or 120Hz display, the inference latency must be significantly 
lower than the frame budget (11.1ms or 8.3ms).

What it Measure:
---------------
- Latency: p50 (median), p95 (worst-case), and p99 (extreme tail).
- Throughput: Samples processed per second (fps).
- Scaling: Performance across different batch sizes (1 to 32).

Usage:
------
    python -m src.benchmark_mps \\
        --checkpoint checkpoints/best_model.pt \\
        --window_size 30 \\
        --batch_sizes 1,4,8,16,32
"""

import argparse
import time
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from src.models.intent_former import IntentFormer
from src.data.h2o_dataset import NUM_CLASSES


def benchmark(args):
    if not torch.backends.mps.is_available():
        print("[benchmark] MPS not available — falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print(f"[benchmark] Using Apple MPS (M-series GPU)")

    # ── Load model ────────────────────────────────────────
    num_classes = 36 # default
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        num_classes = ckpt.get("num_classes", 36)
        print(f"[benchmark] Loaded checkpoint: {args.checkpoint} (classes={num_classes})")
    else:
        print(f"[benchmark] No checkpoint — using random weights (classes={num_classes})")
    
    model = IntentFormer(
        input_dim=126 + 16, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        num_classes=num_classes, window_size=args.window_size, dropout=0.0,
    ).to(device).eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(ckpt["model"])

    T = args.window_size
    header = f"{'BatchSize':>10} {'Throughput(fps)':>18} {'p50(ms)':>10} " \
             f"{'p95(ms)':>10} {'p99(ms)':>10} {'LatMean(ms)':>13}"
    print("\n" + header)
    print("─" * len(header))

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    for B in batch_sizes:
        hand = torch.randn(B, T, 126).to(device)
        obj  = torch.randn(B, T,  16).to(device)
        obs  = torch.rand(B).to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(hand, obj, obs)
        if device.type == "mps":
            torch.mps.synchronize()

        # Timed runs
        latencies_ms = []
        with torch.no_grad():
            for _ in range(args.runs):
                t0 = time.perf_counter()
                _  = model(hand, obj, obs)
                if device.type == "mps":
                    torch.mps.synchronize()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        p50  = statistics.median(latencies_ms)
        p95  = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        p99  = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        mean = statistics.mean(latencies_ms)
        fps  = (B * 1000.0) / mean    # samples per second

        print(f"{B:>10} {fps:>18.1f} {p50:>10.2f} {p95:>10.2f} "
              f"{p99:>10.2f} {mean:>13.2f}")

    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="")
    p.add_argument("--window_size", type=int, default=30)
    p.add_argument("--batch_sizes", default="1,4,8,16,32")
    p.add_argument("--warmup",      type=int, default=50)
    p.add_argument("--runs",        type=int, default=200)
    p.add_argument("--d_model",     type=int, default=128)
    p.add_argument("--nhead",       type=int, default=4)
    p.add_argument("--num_layers",  type=int, default=4)
    p.add_argument("--dim_ff",      type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
