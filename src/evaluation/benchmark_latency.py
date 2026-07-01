"""C3: Model inference latency benchmark.

Measures end-to-end Python inference time including:
  - Point cloud input assembly
  - Model forward pass (PointNet + GRU + self-attention + CVAE)
  - Candidate selection (K candidates)

Does NOT include: Unity C# overhead, Air Link transmission, or XR render pipeline.
Unity-side latency is logged separately via AuraXRModelRuntime.latencyMs.

Usage:
  python3 src/evaluation/benchmark_latency.py --checkpoint checkpoints/full_phase2_best.pt
  python3 src/evaluation/benchmark_latency.py --checkpoint checkpoints/full_phase2_best.pt --k 5 --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.grasp_model import GraspModel  # noqa: E402
from model.model_io import DEFAULT_N_POINTS, HOT3D_FRAME_DIM  # noqa: E402
from utils.paths import RESULTS_DIR  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    model: GraspModel,
    device: torch.device,
    k: int = 1,
    n_points: int = DEFAULT_N_POINTS,
    window: int = 16,
    batch_size: int = 1,
    warmup: int = 50,
    n_runs: int = 500,
) -> dict[str, float]:
    model.eval()

    frame_feat = torch.randn(batch_size, window, HOT3D_FRAME_DIM, device=device)
    obj_pts = torch.randn(batch_size, n_points, 3, device=device)
    prev_pose = torch.zeros(batch_size, 45, device=device)
    contact_flag = torch.zeros(batch_size, window, 1, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.infer(frame_feat, obj_pts, prev_pose=prev_pose, contact_flag=contact_flag, k=k)
        _sync(device)

    # Measurement
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.infer(frame_feat, obj_pts, prev_pose=prev_pose, contact_flag=contact_flag, k=k)
            _sync(device)
            times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    return {
        "median_ms": float(np.median(times_ms)),
        "mean_ms": float(mean(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "n_runs": n_runs,
        "warmup": warmup,
        "k": k,
        "window": window,
        "n_points": n_points,
        "batch_size": batch_size,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu / mps / cuda (auto-detects if omitted)")
    parser.add_argument("--k", type=int, default=1, help="Number of CVAE candidates")
    parser.add_argument("--window", type=int, default=16, help="Temporal window length")
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS, dest="n_points")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--out", type=str, default=None, help="Save JSON results to this path")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_args = ckpt.get("args", {})

    model = GraspModel(
        hidden=saved_args.get("hidden", 256),
        z_dim=saved_args.get("z_dim", 64),
        encoder_type=saved_args.get("encoder_type", "gru"),
        obj_encoder_type=saved_args.get("obj_encoder_type", "pointnet"),
        use_attention=saved_args.get("use_attention", True),
        use_film=saved_args.get("use_film", True),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    print(f"Benchmarking on {device}  k={args.k}  window={args.window}  n_points={args.n_points}")
    print(f"Measuring: point cloud input + model forward + candidate selection (NOT Unity overhead)")

    results = benchmark(
        model, device,
        k=args.k,
        n_points=args.n_points,
        window=args.window,
        warmup=args.warmup,
        n_runs=args.runs,
    )

    print(f"\nLatency results (n={results['n_runs']} runs, warmup={results['warmup']}):")
    print(f"  Median : {results['median_ms']:.2f} ms")
    print(f"  Mean   : {results['mean_ms']:.2f} ms")
    print(f"  P95    : {results['p95_ms']:.2f} ms")
    print(f"  P99    : {results['p99_ms']:.2f} ms")
    print(f"  Min    : {results['min_ms']:.2f} ms")
    print(f"  Max    : {results['max_ms']:.2f} ms")
    print(f"\n90 FPS target: <11.1ms per frame")
    ok = results["p95_ms"] < 11.1
    print(f"  P95 {'PASS' if ok else 'FAIL'} (<11.1ms threshold)")

    out_path = args.out or str(RESULTS_DIR / f"latency_k{args.k}_{device.type}.json")
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
