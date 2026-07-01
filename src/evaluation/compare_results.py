"""Morning-after comparison table across all experiment eval results.

Reads results/eval_*.json and results/master_summary.json and prints a
ranked Markdown table by geodesic_err_deg (primary metric, B10.1).

Usage:
  python3 src/evaluation/compare_results.py
  python3 src/evaluation/compare_results.py --metric mpjpe_mm
  python3 src/evaluation/compare_results.py --source hot3d --csv

Multi-seed mode (C4): aggregates eval_<name>_seed<N>.json files into mean±std rows.
  python3 src/evaluation/compare_results.py --multi-seed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.paths import RESULTS_DIR  # noqa: E402

METRICS = [
    "geodesic_err_deg",
    "mpjpe_mm",
    "fingertip_err_mm",
    "contact_ratio",
    "penetration_depth_mm",
    "joint_limit_violation_rate",
    "diversity_score",
    "quality_auc",
    "quality_spearman",
]

LOWER_IS_BETTER = {
    "geodesic_err_deg", "mpjpe_mm", "fingertip_err_mm",
    "penetration_depth_mm", "joint_limit_violation_rate",
}


def load_results(source: str) -> list[dict]:
    rows = []
    master = RESULTS_DIR / "master_summary.json"
    if master.exists():
        summaries = json.loads(master.read_text())
        for s in summaries:
            eval_data = s.get("eval", {}).get(source, {})
            if eval_data:
                rows.append({"name": s["name"], "description": s.get("description", ""), **eval_data})

    # pick up per-experiment eval.json files: results/<name>/eval.json
    seen_names = {r["name"] for r in rows}
    for path in sorted(RESULTS_DIR.glob("*/eval.json")):
        exp_name = path.parent.name
        if exp_name in seen_names:
            continue
        data = json.loads(path.read_text())
        src_data = data.get(source, data)
        if isinstance(src_data, dict) and "geodesic_err_deg" in src_data:
            rows.append({"name": exp_name, "description": "", **src_data})
    return rows


def fmt(val: float | None, std: Optional[float] = None) -> str:
    if val is None:
        return "—"
    if std is not None:
        return f"{val:.4f}±{std:.4f}"
    return f"{val:.4f}"


def load_multi_seed_results(source: str) -> list[dict]:
    """C4: Aggregate results/<name>_seed<N>/eval.json files into mean±std rows."""
    import numpy as np
    seed_data: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*_seed*/eval.json")):
        folder = path.parent.name  # e.g. "full_seed42"
        parts = folder.rsplit("_seed", 1)
        if len(parts) == 2:
            base_name = parts[0]
            data = json.loads(path.read_text())
            src_data = data.get(source, data)
            if isinstance(src_data, dict) and "geodesic_err_deg" in src_data:
                seed_data[base_name].append(src_data)
    rows = []
    for name, runs in seed_data.items():
        row: dict = {"name": name, "description": f"n_seeds={len(runs)}"}
        for m in METRICS:
            vals = [r[m] for r in runs if m in r and r[m] is not None]
            if vals:
                arr = np.array(vals, dtype=float)
                row[m] = float(arr.mean())
                row[f"{m}_std"] = float(arr.std())
        rows.append(row)
    return rows


def print_table(rows: list[dict], sort_by: str) -> None:
    if not rows:
        print("No results found.")
        return

    reverse = sort_by not in LOWER_IS_BETTER
    rows_sorted = sorted(rows, key=lambda r: r.get(sort_by, float("inf")), reverse=reverse)

    cols = ["rank", "name"] + [m for m in METRICS if any(m in r for r in rows_sorted)]
    header = " | ".join(f"{c:>22}" if c not in ("rank", "name") else f"{c:<5}" for c in cols)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i, row in enumerate(rows_sorted, 1):
        vals = [str(i), row["name"]] + [fmt(row.get(m)) for m in METRICS if any(m in r for r in rows_sorted)]
        line = " | ".join(f"{v:>22}" if j > 1 else f"{v:<5}" for j, v in enumerate(vals))
        print(line)
    print(sep)
    print(f"\nPrimary sort: {sort_by}  ({'lower' if sort_by in LOWER_IS_BETTER else 'higher'} is better)")


def print_csv(rows: list[dict], sort_by: str) -> None:
    if not rows:
        return
    fieldnames = ["name", "description"] + METRICS
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: r.get(sort_by, float("inf"))):
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def print_mean_std_table(rows: list[dict], sort_by: str) -> None:
    """C4: Print mean±std table for multi-seed results."""
    if not rows:
        print("No multi-seed results found (expected eval_<name>_seed<N>.json files).")
        return
    reverse = sort_by not in LOWER_IS_BETTER
    rows_sorted = sorted(rows, key=lambda r: r.get(sort_by, float("inf")), reverse=reverse)
    active_metrics = [m for m in METRICS if any(m in r for r in rows_sorted)]
    header_parts = ["name"] + [m[:18] for m in active_metrics]
    print(" | ".join(f"{h:<22}" for h in header_parts))
    print("-" * (25 * len(header_parts)))
    for row in rows_sorted:
        vals = [row["name"]] + [
            fmt(row.get(m), row.get(f"{m}_std")) for m in active_metrics
        ]
        print(" | ".join(f"{v:<22}" for v in vals))
    print(f"\nPrimary sort: {sort_by}  (mean±std across seeds)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hot3d", "oakink"], default="hot3d")
    parser.add_argument("--metric", default="geodesic_err_deg", help="primary sort metric")
    parser.add_argument("--csv", action="store_true", help="output CSV instead of table")
    parser.add_argument("--multi-seed", action="store_true", dest="multi_seed",
                        help="C4: aggregate eval_<name>_seed<N>.json files into mean±std table")
    args = parser.parse_args()

    if args.multi_seed:
        rows = load_multi_seed_results(args.source)
        print(f"\nMulti-seed results — source={args.source}  n_experiments={len(rows)}\n")
        print_mean_std_table(rows, args.metric)
    else:
        rows = load_results(args.source)
        if args.csv:
            print_csv(rows, args.metric)
        else:
            print(f"\nResults — source={args.source}  n_experiments={len(rows)}\n")
            print_table(rows, args.metric)


if __name__ == "__main__":
    main()
