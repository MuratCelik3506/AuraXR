"""
04_fps_temporal_window.py — Analyse HOT3D frame rate and temporal window implications.

This script answers Q-C from questions.md:
  "Dataset is 30 FPS but Quest 3 runs at 72 Hz — what T is needed?"

It also:
  - Verifies actual frame rate from clip timestamps
  - Shows what different T values mean in wall-clock time at 30 FPS vs 72 Hz
  - Recommends a T value and inference strategy

Usage:
  python 04_fps_temporal_window.py
  python 04_fps_temporal_window.py --n_clips 10
"""

import argparse

import numpy as np

from hot3d_utils import decode_json, load_hot3d, TRAINING_FPS, INFERENCE_HZ, T_CANDIDATES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_clips", type=int, default=5)
    return p.parse_args()


def measure_fps_from_clip(sample: dict) -> dict:
    result = {"fps_measured": None, "n_frames": 0, "duration_s": None}

    key = next((k for k in sample if "mano" in k.lower()), None)
    if key is None:
        key = next((k for k in sample if "hand_crops" in k.lower()), None)
    if key is None:
        return result

    data = decode_json(sample[key])
    if not isinstance(data, dict) or len(data) < 2:
        return result

    timestamps = np.fromiter(data.keys(), dtype=np.int64)
    timestamps.sort()

    intervals_ns = np.diff(timestamps)
    median_ns = float(np.median(intervals_ns))

    result["n_frames"]         = len(timestamps)
    result["fps_measured"]     = 1e9 / median_ns
    result["duration_s"]       = float(timestamps[-1] - timestamps[0]) / 1e9
    result["interval_ms_std"]  = float(intervals_ns.std()) / 1e6
    return result


def print_temporal_analysis(fps_results: list):
    fpss = [r["fps_measured"] for r in fps_results if r["fps_measured"] is not None]
    if not fpss:
        print("[WARN] Could not measure FPS — no timestamped data found.")
        return

    measured_fps = float(np.mean(fpss))
    t16_ms_training   = 16 * 1000 / TRAINING_FPS
    t16_ms_inference  = 16 * 1000 / INFERENCE_HZ
    t_parity          = round(t16_ms_training * INFERENCE_HZ / 1000)

    print(f"\n{'='*60}\n  FRAME RATE ANALYSIS\n{'='*60}")
    print(f"\n  Clips measured        : {len(fpss)}")
    print(f"  Measured FPS (mean)   : {measured_fps:.2f} FPS")
    print(f"  Measured FPS (std)    : {np.std(fpss):.2f} FPS")
    print(f"  Frame interval (ms)   : {1000/measured_fps:.2f} ms")

    print(f"\n{'='*60}\n  TEMPORAL WINDOW ANALYSIS\n{'='*60}")
    print(f"\n  Training rate   : {TRAINING_FPS} FPS (HOT3D)")
    print(f"  Inference rate  : {INFERENCE_HZ} Hz (Quest 3 native)")

    print(f"\n  {'T':>6} | {'@ %d FPS' % TRAINING_FPS:>12} | {'@ %d Hz' % INFERENCE_HZ:>12} | Note")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*32}")

    for T in T_CANDIDATES:
        ms_train = T * 1000 / TRAINING_FPS
        ms_infer = T * 1000 / INFERENCE_HZ
        note = ""
        if T == 16:
            note = "← plan default"
        elif T == t_parity:
            note = f"← {INFERENCE_HZ}Hz parity with T=16@{TRAINING_FPS}FPS"
        print(f"  {T:>6} | {ms_train:>10.0f}ms | {ms_infer:>10.0f}ms | {note}")

    print(f"\n  KEY INSIGHT:")
    print(f"  T=16 @ {TRAINING_FPS} FPS = {t16_ms_training:.0f} ms context")
    print(f"  T=16 @ {INFERENCE_HZ} Hz  = {t16_ms_inference:.0f} ms context  ← {t16_ms_training/t16_ms_inference:.1f}x less at inference!")
    print(f"  T={t_parity} @ {INFERENCE_HZ} Hz  = {t_parity*1000/INFERENCE_HZ:.0f} ms  ← matches T=16@{TRAINING_FPS}FPS")
    print(f"\n  OPTIONS:")
    print(f"  A) Cap inference at {TRAINING_FPS} FPS — no mismatch, wastes display budget.")
    print(f"  B) Time-relative positional encoding (ms-based) — train {TRAINING_FPS}FPS, infer {INFERENCE_HZ}Hz.")
    print(f"  C) Upsample training data to {INFERENCE_HZ}Hz via interpolation — consistent T=16.")
    print(f"\n  → Resolve in Q-C of questions.md before designing the Transformer PE.")


def main():
    args = parse_args()
    dataset = load_hot3d("train")

    fps_results = []
    for i, sample in enumerate(dataset):
        if i >= args.n_clips:
            break
        r = measure_fps_from_clip(sample)
        fps_results.append(r)
        if r["fps_measured"]:
            print(f"  Clip {i}: {r['fps_measured']:.2f} FPS, {r['n_frames']} frames, {r['duration_s']:.1f}s")

    print_temporal_analysis(fps_results)


if __name__ == "__main__":
    main()
