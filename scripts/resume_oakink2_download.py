"""Resume missing OakInk2 data tar downloads from Hugging Face."""

from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "kelvin34501/OakInk-v2"
LOCAL_DIR = Path("data/raw/oakink2")


def main() -> None:
    files = [
        f
        for f in list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".tar")
    ]
    missing = [f for f in files if not (LOCAL_DIR / f).exists()]
    print(f"[oakink2] remote data tar: {len(files)}", flush=True)
    print(f"[oakink2] already present: {len(files) - len(missing)}", flush=True)
    print(f"[oakink2] missing: {len(missing)}", flush=True)

    for i, filename in enumerate(missing, 1):
        started = time.time()
        print(f"[oakink2] {i}/{len(missing)} downloading {filename}", flush=True)
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(LOCAL_DIR),
        )
        size_gb = Path(path).stat().st_size / 1e9
        elapsed = time.time() - started
        print(
            f"[oakink2] {i}/{len(missing)} done {filename} "
            f"{size_gb:.2f} GB in {elapsed:.1f}s",
            flush=True,
        )

    print("[oakink2] complete", flush=True)


if __name__ == "__main__":
    main()
