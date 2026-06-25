#!/usr/bin/env python3
"""Download the ARCTIC raw MANO sequence package needed by AuraXR.

This intentionally downloads only raw_seqs.zip, not the full image dataset.
Required environment variables:
  ARCTIC_USERNAME
  ARCTIC_PASSWORD
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


RAW_SEQS_URL = (
    "https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&"
    "sfile=arctic_release/"
    "c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/"
    "v1_0/data/raw_seqs.zip"
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing {name}. Export it before running this script.")
    return value


def download(url: str, out_path: Path, username: str, password: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.post(
        url,
        data={"username": username, "password": password},
        stream=True,
        verify=False,
        allow_redirects=True,
        timeout=60,
    ) as response:
        if response.status_code == 401:
            raise SystemExit("ARCTIC authentication failed. Check username/password.")
        response.raise_for_status()

        total = int(response.headers.get("content-length", "0"))
        with out_path.open("wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=Path, default=Path("data/raw/arctic/downloads"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/raw/arctic"))
    parser.add_argument("--keep_zip", action="store_true")
    args = parser.parse_args()

    username = _require_env("ARCTIC_USERNAME")
    password = _require_env("ARCTIC_PASSWORD")

    zip_path = args.download_dir / "raw_seqs.zip"
    print(f"Downloading ARCTIC raw sequences to {zip_path}")
    download(RAW_SEQS_URL, zip_path, username, password)

    print(f"Extracting {zip_path} to {args.out_dir}")
    unzip(zip_path, args.out_dir)

    raw_seqs = args.out_dir / "raw_seqs"
    if not raw_seqs.exists():
        raise SystemExit(f"Download extracted, but {raw_seqs} was not found.")

    if not args.keep_zip:
        zip_path.unlink()

    print(f"Ready: {raw_seqs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
