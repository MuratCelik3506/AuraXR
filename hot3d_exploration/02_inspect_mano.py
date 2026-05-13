"""
02_inspect_mano.py — Extract and inspect MANO hand annotations from HOT3D clips.

What this script answers:
  - What is the exact schema of MANO annotations (θ, β, wrist transform)?
  - Are both hands present in each frame? What fraction of frames are bimanual?
  - What is the range of β (shape) values across participants?
  - What is the wrist position coordinate frame (world? camera?)?
  - Can we derive a synthetic controller proxy from the wrist transform?

This script is critical for answering Q-A (controller proxy) in questions.md.

Usage:
  python 02_inspect_mano.py
  python 02_inspect_mano.py --n_clips 10 --plot
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hot3d_utils import (
    decode_json, load_hot3d, ensure_output_dir,
    BETA_KEYS, TRANSL_KEYS, first_value,
)

SIDES = ("left", "right")
MAX_BETA_SAMPLES  = 30
MAX_WRIST_SAMPLES = 50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_clips", type=int, default=5, help="Number of clips to analyse")
    p.add_argument("--split", default="train")
    p.add_argument("--plot", action="store_true", help="Save diagnostic plots to ./output/")
    return p.parse_args()


def _extract_schema(hand_data: dict) -> dict:
    return {k: (np.array(v).shape if isinstance(v, list) else type(v).__name__)
            for k, v in hand_data.items()}


def analyse_mano_in_clip(sample: dict) -> dict:
    result = {
        "participant_id": None,
        "device": None,
        "sequence_id": None,
        "n_frames": 0,
        "n_bimanual_frames": 0,
        "n_left_only": 0,
        "n_right_only": 0,
        "n_no_hands": 0,
        "mano_schema_left": None,
        "mano_schema_right": None,
        "beta_left_samples": [],
        "beta_right_samples": [],
        "wrist_positions_left": [],
        "wrist_positions_right": [],
        "mano_key": None,
    }

    if "info.json" in sample:
        info = decode_json(sample["info.json"])
        result["participant_id"] = info.get("participant_id")
        result["device"] = info.get("device")
        result["sequence_id"] = info.get("sequence_id")

    mano_key = next((k for k in sample if "mano" in k.lower()), None)
    if mano_key is None:
        print("[WARN] No MANO key found in this clip.")
        print(f"       Available keys: {list(sample.keys())}")
        return result

    result["mano_key"] = mano_key
    mano_data = decode_json(sample[mano_key])
    result["n_frames"] = len(mano_data)

    for frame_data in mano_data.values():
        has = {s: s in frame_data and frame_data[s] is not None for s in SIDES}

        if has["left"] and has["right"]:
            result["n_bimanual_frames"] += 1
        elif has["left"]:
            result["n_left_only"] += 1
        elif has["right"]:
            result["n_right_only"] += 1
        else:
            result["n_no_hands"] += 1

        for side in SIDES:
            if not has[side]:
                continue
            hand = frame_data[side]
            schema_key = f"mano_schema_{side}"
            if result[schema_key] is None:
                result[schema_key] = _extract_schema(hand)

            beta_key = f"beta_{side}_samples"
            if len(result[beta_key]) < MAX_BETA_SAMPLES:
                beta = first_value(hand, BETA_KEYS)
                if beta is not None:
                    result[beta_key].append(np.array(beta))

            wrist_key = f"wrist_positions_{side}"
            if len(result[wrist_key]) < MAX_WRIST_SAMPLES:
                trans = first_value(hand, TRANSL_KEYS)
                if trans is not None:
                    result[wrist_key].append(np.array(trans).flatten()[:3])

    return result


def _build_betas(results: list) -> np.ndarray | None:
    all_betas = []
    for r in results:
        all_betas.extend(r["beta_left_samples"])
        all_betas.extend(r["beta_right_samples"])
    return np.stack(all_betas) if all_betas else None


def print_summary(results: list) -> np.ndarray | None:
    print(f"\n{'='*60}")
    print(f"  MANO ANNOTATION SUMMARY ({len(results)} clips)")
    print(f"{'='*60}")

    total_frames    = sum(r["n_frames"] for r in results)
    total_bimanual  = sum(r["n_bimanual_frames"] for r in results)
    total_left_only = sum(r["n_left_only"] for r in results)
    total_right_only= sum(r["n_right_only"] for r in results)
    total_no_hands  = sum(r["n_no_hands"] for r in results)

    denom = max(total_frames, 1)
    print(f"\n  Total frames analysed : {total_frames:,}")
    print(f"  Bimanual frames       : {total_bimanual:,} ({100*total_bimanual/denom:.1f}%)")
    print(f"  Left-only frames      : {total_left_only:,} ({100*total_left_only/denom:.1f}%)")
    print(f"  Right-only frames     : {total_right_only:,} ({100*total_right_only/denom:.1f}%)")
    print(f"  No-hand frames        : {total_no_hands:,} ({100*total_no_hands/denom:.1f}%)")

    for side in SIDES:
        schema_key = f"mano_schema_{side}"
        for r in results:
            if r[schema_key]:
                print(f"\n  MANO {side.upper()} schema (key: {r['mano_key']}):")
                for field, shape in r[schema_key].items():
                    print(f"    {field:<25} shape={shape}")
                break

    betas = _build_betas(results)
    if betas is not None:
        print(f"\n  β (shape) statistics across {len(betas)} samples:")
        print(f"    β dim  : {betas.shape[1]}")
        print(f"    β mean : {betas.mean(axis=0).round(3)}")
        print(f"    β std  : {betas.std(axis=0).round(3)}")
        print(f"    β min  : {betas.min(axis=0).round(3)}")
        print(f"    β max  : {betas.max(axis=0).round(3)}")
        print(f"\n  [INSIGHT] High β std → model must generalise across hand shapes → supports per-session calibration (Q-E).")

    all_wrists = []
    for r in results:
        all_wrists.extend(r["wrist_positions_left"])
        all_wrists.extend(r["wrist_positions_right"])

    if all_wrists:
        wrists = np.stack(all_wrists)
        mean_norm = np.linalg.norm(wrists.mean(axis=0))
        print(f"\n  Wrist position statistics ({len(wrists)} samples):")
        print(f"    XYZ mean (m)   : {wrists.mean(axis=0).round(3)}")
        print(f"    XYZ std  (m)   : {wrists.std(axis=0).round(3)}")
        print(f"    XYZ range (m)  : {wrists.min(axis=0).round(3)} → {wrists.max(axis=0).round(3)}")
        coord_frame = "WORLD space" if mean_norm > 0.5 else "camera-local or normalised space"
        print(f"    [INSIGHT] mean distance from origin {mean_norm:.3f}m → wrist likely in {coord_frame}.")

    return betas


def save_plots(betas: np.ndarray):
    ensure_output_dir()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("MANO β (shape) distribution across sampled frames")
    for i, ax in enumerate(axes.flat):
        if i < betas.shape[1]:
            ax.hist(betas[:, i], bins=20, color="steelblue", edgecolor="white")
            ax.set_title(f"β[{i}]")
            ax.set_xlabel("value")
    plt.tight_layout()
    plt.savefig("output/beta_distribution.png", dpi=150)
    print(f"\n  [SAVED] output/beta_distribution.png")
    plt.close()


def main():
    args = parse_args()
    dataset = load_hot3d(args.split)

    results = []
    for i, sample in enumerate(dataset):
        if i >= args.n_clips:
            break
        print(f"[INFO] Processing clip {i+1}/{args.n_clips}...")
        results.append(analyse_mano_in_clip(sample))

    betas = print_summary(results)

    if args.plot and betas is not None:
        save_plots(betas)


if __name__ == "__main__":
    main()
