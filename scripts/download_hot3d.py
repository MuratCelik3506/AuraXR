"""
HOT3D Dataset Acquisition CLI — Intent-Aware XR Framework
=========================================================

This utility script automates the retrieval of the HOT3D (Hands-Object 
Tracking in 3D) dataset from the Hugging Face Hub.

Dataset Context:
----------------
HOT3D is a large-scale (~800GB) dataset focused on 6D object pose and 
egocentric hand tracking. For this project, we focus on the "Clips" 
subset, which contains synchronized streams from Meta Quest 3 and 
Project Aria.

Key Functions:
--------------
1. Metadata Retrieval: Downloads `clip_splits.json` and 
   `clip_definitions.json` to understand the dataset structure.
   
2. Selective Downloading: Allows downloading specific splits 
   (`train`, `test`) and devices (`Aria`, `Quest3`).
   
3. Rate Limiting: Support for `--max_clips` to download a small 
   representative subset for rapid experimentation and debugging.

Prerequisites:
--------------
- A Hugging Face account with access to the `bop-benchmark/hot3d` dataset.
- `huggingface-cli login` must be executed before running this script.

Usage:
------
    python scripts/download_hot3d.py --max_clips 10 --device Aria
"""


import argparse
import os
from pathlib import Path


def download_metadata(output_dir: Path, repo_id: str = "bop-benchmark/hot3d"):
    """Download clip_splits.json and clip_definitions.json."""
    from huggingface_hub import hf_hub_download

    for fname in ["clip_splits.json", "clip_definitions.json"]:
        print(f"[download] Downloading {fname}...")
        local_path = hf_hub_download(
            repo_id   = repo_id,
            filename  = fname,
            repo_type = "dataset",
            local_dir = str(output_dir),
        )
        print(f"  → {local_path}")


def download_clips(
    output_dir: Path,
    split:      str,       # "train" or "test"
    device:     str,       # "Aria" or "Quest3"
    max_clips:  int | None = None,
    repo_id:    str        = "bop-benchmark/hot3d",
):
    """
    Download clip tar archives for a given split and device.

    HOT3D folder structure on HF:
        train_aria/clip-<ID>.tar
        train_quest3/clip-<ID>.tar
        test_aria/clip-<ID>.tar
        test_quest3/clip-<ID>.tar
    """
    import json
    from huggingface_hub import hf_hub_download, list_repo_files

    # Map to folder name
    folder = f"{split}_{device.lower()}"
    local_folder = output_dir / folder
    local_folder.mkdir(parents=True, exist_ok=True)

    # Load clip_splits.json to find which clip IDs to download
    splits_path = output_dir / "clip_splits.json"
    if not splits_path.exists():
        print("[download] clip_splits.json not found — downloading metadata first...")
        download_metadata(output_dir, repo_id)

    with open(splits_path) as f:
        splits = json.load(f)

    split_key = "train" if split == "train" else "test_ht_pose"
    clip_ids  = splits.get(split_key, {}).get(device, [])

    if not clip_ids:
        print(f"[download] No clips found for split={split_key}, device={device}")
        return

    if max_clips is not None:
        clip_ids = clip_ids[:max_clips]

    print(f"[download] Downloading {len(clip_ids)} clips → {local_folder}")

    for i, clip_id in enumerate(clip_ids):
        # Try zero-padded filename (6 digits)
        filename = f"{folder}/clip-{clip_id:06d}.tar"
        local_tar = local_folder / f"clip-{clip_id:06d}.tar"

        if local_tar.exists():
            print(f"  [{i+1}/{len(clip_ids)}] clip-{clip_id:06d}.tar already exists — skip")
            continue

        try:
            path = hf_hub_download(
                repo_id   = repo_id,
                filename  = filename,
                repo_type = "dataset",
                local_dir = str(output_dir),
            )
            print(f"  [{i+1}/{len(clip_ids)}] ✓ {filename}")
        except Exception as e:
            print(f"  [{i+1}/{len(clip_ids)}] ✗ {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download HOT3D-Clips from Hugging Face")
    parser.add_argument("--output_dir",    default="data/hot3d",
                        help="Local directory to save clips")
    parser.add_argument("--split",         nargs="+", default=["train"],
                        choices=["train", "test"],
                        help="Which splits to download")
    parser.add_argument("--device",        nargs="+", default=["Aria"],
                        choices=["Aria", "Quest3"],
                        help="Which device streams to download")
    parser.add_argument("--max_clips",     type=int, default=None,
                        help="Limit number of clips per split/device (for testing)")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Download only JSON metadata files (no clip archives)")
    parser.add_argument("--repo_id",       default="bop-benchmark/hot3d",
                        help="Hugging Face repo ID")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always download metadata
    print("─" * 60)
    print("[download] Step 1: Metadata files")
    download_metadata(output_dir, args.repo_id)

    if args.metadata_only:
        print("[download] --metadata_only set, skipping clip download.")
        return

    # Download requested splits × devices
    for split in args.split:
        for device in args.device:
            print("─" * 60)
            print(f"[download] Step 2: Clips  split={split}  device={device}")
            download_clips(
                output_dir = output_dir,
                split      = split,
                device     = device,
                max_clips  = args.max_clips,
                repo_id    = args.repo_id,
            )

    print("─" * 60)
    print(f"[download] Done.  Data saved to: {output_dir.resolve()}")
    print(f"[download] Next step:")
    print(f"  python -m src.train --dataset combined \\")
    print(f"      --data_root data/h2o \\")
    print(f"      --hot3d_root {output_dir} \\")
    print(f"      --fusion shared_head")


if __name__ == "__main__":
    main()
