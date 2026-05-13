"""
06_download_annotations.py — Download HOT3D annotation files needed for AuraXR training.

Downloads (Phase 1 — annotations only, ~2.7 GB total):
  - hand_data.zip  per sequence: MANO θ, β, wrist transform  ← training target + controller proxy
  - ground_truth.zip per sequence: object 6DoF poses          ← affordance features
  - Hot3DAssets_assets_assets.zip: 3D object meshes           ← surface normals, SDF queries

Does NOT download:
  - main_vrs  (715 GB total) — deferred to Phase 2 (POC subset only)
  - video_main_rgb  (preview videos, not needed for training)
  - Aria MPS/SLAM files  (not used in AuraXR)

After this script completes, run 07_identify_poc_sequences.py to find
which sequences contain the POC objects — then download only those VRS files.

Usage:
  python 06_download_annotations.py
  python 06_download_annotations.py --device quest3   # Quest3 only
  python 06_download_annotations.py --device aria     # Aria only
  python 06_download_annotations.py --dry_run         # show what would be downloaded
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm

# Participant IDs reserved for test set (from clip_splits.json)
TEST_PIDS = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}

JSON_DIR  = Path("../JSON")
DATA_DIR  = Path("../data")

QUEST_JSON = JSON_DIR / "Hot3DQuest_download_urls.json"
ARIA_JSON  = JSON_DIR / "Hot3DAria_download_urls.json"
ASSETS_JSON= JSON_DIR / "Hot3DAssets_download_urls.json"

# Files to download per sequence (all others skipped)
QUEST_KEYS = ["hand_data", "ground_truth"]
ARIA_KEYS  = ["hand_data", "ground_truth"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["quest3", "aria", "both"], default="both")
    p.add_argument("--split",  choices=["train", "test", "all"], default="all")
    p.add_argument("--dry_run", action="store_true", help="Print what would be downloaded, don't download")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_sha1: str | None = None,
                  expected_bytes: int | None = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_bytes and dest.stat().st_size == expected_bytes:
            return True  # already complete
        dest.unlink()

    tmp = dest.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            bar = tqdm(total=total, unit="B", unit_scale=True,
                       desc=dest.name[-40:], leave=False)
            while chunk := resp.read(1 << 16):
                out.write(chunk)
                bar.update(len(chunk))
            bar.close()

        if expected_sha1 and sha1_file(tmp) != expected_sha1:
            tmp.unlink()
            print(f"    [SHA1 MISMATCH] {dest.name} — deleting, retry manually")
            return False

        tmp.rename(dest)
        return True

    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"    [ERROR] {dest.name}: {e}")
        return False


def build_task_list(sequences: dict, keys: list, device_dir: Path,
                    split: str) -> list[dict]:
    tasks = []
    for seq_id, seq_data in sequences.items():
        pid = seq_id.split("_")[0]
        is_test = pid in TEST_PIDS
        if split == "train" and is_test:
            continue
        if split == "test" and not is_test:
            continue
        split_label = "test" if is_test else "train"
        for key in keys:
            if key not in seq_data:
                continue
            info = seq_data[key]
            dest = device_dir / split_label / seq_id / info["filename"]
            tasks.append({
                "seq_id":   seq_id,
                "key":      key,
                "url":      info["download_url"],
                "dest":     dest,
                "sha1":     info.get("sha1sum"),
                "size_mb":  info["file_size_bytes"] / 1e6,
                "size_bytes": info["file_size_bytes"],
            })
    return tasks


def print_plan(tasks: list, label: str):
    total_gb = sum(t["size_bytes"] for t in tasks) / 1e9
    already  = sum(1 for t in tasks if t["dest"].exists()
                   and t["dest"].stat().st_size == t["size_bytes"])
    print(f"\n  {label}: {len(tasks)} files, {total_gb:.2f} GB "
          f"({already} already complete)")
    for t in tasks[:5]:
        status = "✓" if (t["dest"].exists() and t["dest"].stat().st_size == t["size_bytes"]) else " "
        print(f"    [{status}] {t['seq_id']}/{t['key']}  {t['size_mb']:.1f} MB")
    if len(tasks) > 5:
        print(f"    ... and {len(tasks)-5} more")


def run_downloads(tasks: list, label: str):
    pending = [t for t in tasks
               if not (t["dest"].exists() and t["dest"].stat().st_size == t["size_bytes"])]
    total_gb = sum(t["size_bytes"] for t in pending) / 1e9
    print(f"\n[{label}] Downloading {len(pending)} files ({total_gb:.2f} GB remaining)...")

    ok = fail = 0
    for i, t in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {t['seq_id']}/{t['key']}  {t['size_mb']:.1f} MB")
        if download_file(t["url"], t["dest"], t["sha1"], t["size_bytes"]):
            ok += 1
        else:
            fail += 1

    print(f"\n  Done: {ok} ok, {fail} failed")


def main():
    args = parse_args()

    all_tasks = []

    if args.device in ("quest3", "both"):
        with open(QUEST_JSON) as f:
            quest = json.load(f)
        tasks = build_task_list(quest["sequences"], QUEST_KEYS,
                                DATA_DIR / "quest3", args.split)
        all_tasks.extend(tasks)
        print_plan(tasks, "Quest3 annotations")

    if args.device in ("aria", "both"):
        with open(ARIA_JSON) as f:
            aria = json.load(f)
        tasks = build_task_list(aria["sequences"], ARIA_KEYS,
                                DATA_DIR / "aria", args.split)
        all_tasks.extend(tasks)
        print_plan(tasks, "Aria annotations")

    # Assets (3D object models — always included)
    with open(ASSETS_JSON) as f:
        assets_data = json.load(f)
    asset_info = assets_data["sequences"]["assets"]["assets"]
    asset_task = {
        "seq_id":    "assets",
        "key":       "assets",
        "url":       asset_info["download_url"],
        "dest":      DATA_DIR / "assets" / asset_info["filename"],
        "sha1":      asset_info.get("sha1sum"),
        "size_mb":   asset_info["file_size_bytes"] / 1e6,
        "size_bytes":asset_info["file_size_bytes"],
    }
    print_plan([asset_task], "3D object models (assets)")
    all_tasks.append(asset_task)

    total_gb = sum(t["size_bytes"] for t in all_tasks) / 1e9
    pending  = sum(1 for t in all_tasks
                   if not (t["dest"].exists() and t["dest"].stat().st_size == t["size_bytes"]))
    pending_gb = sum(t["size_bytes"] for t in all_tasks
                     if not (t["dest"].exists() and t["dest"].stat().st_size == t["size_bytes"])) / 1e9

    print(f"\n{'='*60}")
    print(f"  Total: {len(all_tasks)} files, {total_gb:.2f} GB")
    print(f"  Remaining: {pending} files, {pending_gb:.2f} GB to download")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] No files downloaded. Remove --dry_run to start.")
        return

    if pending == 0:
        print("\nAll files already downloaded. Nothing to do.")
        return

    # Quest3
    if args.device in ("quest3", "both"):
        quest_tasks = [t for t in all_tasks if "quest3" in str(t["dest"])]
        run_downloads(quest_tasks, "Quest3")

    # Aria
    if args.device in ("aria", "both"):
        aria_tasks = [t for t in all_tasks if "/aria/" in str(t["dest"])]
        run_downloads(aria_tasks, "Aria")

    # Assets
    run_downloads([asset_task], "Assets")

    print("\n[DONE] Phase 1 complete. Run 07_identify_poc_sequences.py next.")


if __name__ == "__main__":
    main()
