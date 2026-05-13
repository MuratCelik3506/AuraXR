"""
01_explore_clips.py — Load a sample HOT3D clip from HuggingFace and inspect its structure.

What this script answers:
  - What keys/fields exist in a clip sample?
  - What does the cameras.json look like?
  - What does the info.json look like?
  - What images are available and what are their properties?
  - Are both hands present? Is MANO data present?

Usage:
  python 01_explore_clips.py
  python 01_explore_clips.py --device quest3   # filter to Quest 3 clips only
  python 01_explore_clips.py --device aria      # filter to Aria clips only
  python 01_explore_clips.py --n 5              # inspect 5 clips instead of 1
"""

import argparse
import io
import pprint
from pathlib import Path

from hot3d_utils import decode_json, load_hot3d, ensure_output_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["aria", "quest3", "all"], default="all")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n", type=int, default=1, help="Number of clips to inspect")
    p.add_argument("--save_images", action="store_true", help="Save first frame images to ./output/")
    return p.parse_args()


def inspect_sample(sample: dict, idx: int, save_images: bool = False):
    print(f"\n{'='*60}")
    print(f"  CLIP {idx}")
    print(f"{'='*60}")

    print(f"\n[Keys in sample]")
    for k, v in sample.items():
        if isinstance(v, (bytes, bytearray)):
            print(f"  {k:<30} bytes ({len(v):,} bytes)")
        elif isinstance(v, dict):
            print(f"  {k:<30} dict  ({len(v)} keys)")
        elif isinstance(v, list):
            print(f"  {k:<30} list  (len={len(v)})")
        else:
            print(f"  {k:<30} {type(v).__name__}  = {str(v)[:80]}")

    if "info.json" in sample:
        print(f"\n[info.json]")
        pprint.pprint(decode_json(sample["info.json"]), indent=2)

    if "cameras.json" in sample:
        cams = decode_json(sample["cameras.json"])
        print(f"\n[cameras.json] — {len(cams)} camera entries")
        for cam_id, cam_data in list(cams.items())[:3]:
            print(f"  Camera ID: {cam_id}")
            pprint.pprint(cam_data, indent=4)

    if "hand_crops.json" in sample:
        crops = decode_json(sample["hand_crops.json"])
        print(f"\n[hand_crops.json] — {len(crops)} entries")
        if crops:
            first_key = list(crops.keys())[0]
            print(f"  First entry ({first_key}):")
            pprint.pprint(crops[first_key], indent=4)

    image_keys = [k for k in sample if k.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"\n[Images] — {len(image_keys)} image files")
    for img_key in sorted(image_keys)[:6]:
        raw = sample[img_key]
        if isinstance(raw, (bytes, bytearray)):
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            print(f"  {img_key:<40} {img.mode} {img.width}×{img.height}")
            if save_images:
                out_dir = ensure_output_dir() / f"clip_{idx:04d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                img.save(out_dir / img_key.replace("/", "_"))
        else:
            print(f"  {img_key:<40} {type(raw).__name__}")
    if len(image_keys) > 6:
        print(f"  ... and {len(image_keys) - 6} more")

    pose_keys = [k for k in sample if any(t in k.lower() for t in ("pose", "mano", "hand"))]
    if pose_keys:
        print(f"\n[Pose/MANO keys]")
        for k in pose_keys:
            v = sample[k]
            if isinstance(v, (bytes, bytearray)):
                try:
                    parsed = decode_json(v)
                    print(f"  {k}: (JSON, {len(parsed)} entries)")
                    if isinstance(parsed, dict):
                        first_key = list(parsed.keys())[0]
                        print(f"    First entry ({first_key}):")
                        pprint.pprint(parsed[first_key], indent=6)
                except Exception:
                    print(f"  {k}: (binary, {len(v)} bytes)")
            else:
                print(f"  {k}: {type(v).__name__} = {str(v)[:80]}")


def main():
    args = parse_args()
    print(f"[INFO] Loading HOT3D ({args.split} split) from HuggingFace — streaming...")
    dataset = load_hot3d(args.split)

    seen = 0
    for sample in dataset:
        if args.device != "all":
            info = decode_json(sample.get("info.json", {}))
            device = info.get("device", "").lower()
            if args.device == "quest3" and "quest" not in device:
                continue
            if args.device == "aria" and "aria" not in device:
                continue

        inspect_sample(sample, seen, save_images=args.save_images)
        seen += 1
        if seen >= args.n:
            break

    if seen == 0:
        print(f"[WARNING] No clips found matching device={args.device} in split={args.split}")


if __name__ == "__main__":
    main()
