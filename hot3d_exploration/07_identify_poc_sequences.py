"""
07_identify_poc_sequences.py — Parse downloaded ground_truth to find POC object sequences.

After 06_download_annotations.py completes, this script:
  1. Reads every ground_truth.zip already downloaded
  2. Finds which sequences contain the POC objects (mug, can_soup, bottle, dvd_remote)
  3. Outputs a JSON list of VRS download URLs for only those sequences
  4. Prints total VRS download size for the POC subset

The output JSON feeds directly into 08_download_poc_vrs.py.

POC objects (priority 1 from HOT3D_OBJECTS):
  obj_000008  mug_patterned     handle_grasp
  obj_000009  mug_white         handle_grasp
  obj_000010  can_soup          cylindrical_power
  obj_000013  bottle_mustard    bottle_grasp
  obj_000033  dvd_remote        precision

Usage:
  python 07_identify_poc_sequences.py
  python 07_identify_poc_sequences.py --min_frames 30
"""

import argparse
import json
import zipfile
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

DATA_DIR  = Path("../data")
JSON_DIR  = Path("../JSON")
OUT_FILE  = Path("../data/poc_sequences.json")

# Object IDs for POC subset (1-indexed, matching obj_000XXX filenames)
POC_OBJECT_IDS = {8, 9, 10, 13, 33}

POC_NAMES = {
    8:  "mug_patterned",
    9:  "mug_white",
    10: "can_soup",
    13: "bottle_mustard",
    33: "dvd_remote",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min_frames", type=int, default=30,
                   help="Minimum frames a POC object must appear to include the sequence")
    return p.parse_args()


def parse_ground_truth_zip(zip_path: Path) -> dict:
    """
    Extract object presence per frame from a ground_truth.zip.
    Returns {obj_id_int: frame_count}.
    """
    obj_counts = defaultdict(int)
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.endswith(".json") or "object" not in name.lower():
                    continue
                with z.open(name) as f:
                    data = json.load(f)
                # HOT3D ground_truth format: {timestamp: {obj_id_str: {pose...}}}
                # obj_id_str is like "1" or "obj_000001"
                if isinstance(data, dict):
                    for frame_data in data.values():
                        if not isinstance(frame_data, dict):
                            continue
                        for obj_key in frame_data:
                            # normalise to int
                            raw = obj_key.replace("obj_", "").lstrip("0") or "0"
                            try:
                                obj_id = int(raw)
                                obj_counts[obj_id] += 1
                            except ValueError:
                                pass
    except Exception as e:
        pass  # corrupted or unexpected format — skip
    return dict(obj_counts)


def collect_gt_zips(device: str) -> list[Path]:
    base = DATA_DIR / device
    return sorted(base.rglob("*ground_truth*.zip"))


def main():
    args = parse_args()

    # Load URL manifests for VRS files
    with open(JSON_DIR / "Hot3DQuest_download_urls.json") as f:
        quest_manifest = json.load(f)["sequences"]
    with open(JSON_DIR / "Hot3DAria_download_urls.json") as f:
        aria_manifest = json.load(f)["sequences"]

    poc_sequences = []
    total_vrs_gb = 0.0

    for device, manifest in [("quest3", quest_manifest), ("aria", aria_manifest)]:
        gt_zips = collect_gt_zips(device)
        if not gt_zips:
            print(f"[WARN] No ground_truth.zip found for {device}. Run 06_download_annotations.py first.")
            continue

        print(f"\n[{device}] Scanning {len(gt_zips)} ground_truth archives...")
        for zip_path in tqdm(gt_zips):
            seq_id = zip_path.parent.name
            obj_counts = parse_ground_truth_zip(zip_path)

            poc_found = {oid: obj_counts[oid] for oid in POC_OBJECT_IDS
                         if obj_counts.get(oid, 0) >= args.min_frames}
            if not poc_found:
                continue

            # Get VRS download info
            if seq_id not in manifest:
                continue
            seq_info = manifest[seq_id]
            if "main_vrs" not in seq_info:
                continue

            vrs_info = seq_info["main_vrs"]
            vrs_gb   = vrs_info["file_size_bytes"] / 1e9
            total_vrs_gb += vrs_gb

            poc_sequences.append({
                "seq_id":    seq_id,
                "device":    device,
                "poc_objects": {POC_NAMES[oid]: cnt for oid, cnt in poc_found.items()},
                "vrs_filename": vrs_info["filename"],
                "vrs_url":      vrs_info["download_url"],
                "vrs_sha1":     vrs_info.get("sha1sum"),
                "vrs_size_gb":  vrs_gb,
            })

    # Report
    print(f"\n{'='*60}")
    print(f"  POC sequences found: {len(poc_sequences)}")
    print(f"  VRS download size  : {total_vrs_gb:.1f} GB")
    print(f"{'='*60}")

    obj_counts_across = defaultdict(int)
    for s in poc_sequences:
        for obj_name in s["poc_objects"]:
            obj_counts_across[obj_name] += 1

    print(f"\n  Object coverage:")
    for obj_name, count in sorted(obj_counts_across.items()):
        print(f"    {obj_name:<20} {count} sequences")

    # Device split
    devices = defaultdict(int)
    for s in poc_sequences:
        devices[s["device"]] += 1
    print(f"\n  Device split: {dict(devices)}")

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"poc_sequences": poc_sequences, "total_vrs_gb": total_vrs_gb}, f, indent=2)

    print(f"\n  [SAVED] {OUT_FILE}")
    print(f"  Next: run 08_download_poc_vrs.py to download {total_vrs_gb:.1f} GB of VRS files.")


if __name__ == "__main__":
    main()
