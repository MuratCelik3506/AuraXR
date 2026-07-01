"""Morning-after comparison table across all experiment eval results.

Reads results/eval_*.json and results/master_summary.json and prints a
ranked Markdown table by geodesic_err_deg (primary metric, B10.1).

Usage:
  python3 src/evaluation/compare_results.py
  python3 src/evaluation/compare_results.py --metric mpjpe_mm
  python3 src/evaluation/compare_results.py --source hot3d --csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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

    # also pick up standalone eval_*.json files not yet in master
    for path in sorted(RESULTS_DIR.glob("eval_*.json")):
        data = json.loads(path.read_text())
        exp_name = path.stem.removeprefix("eval_")
        if any(r["name"] == exp_name for r in rows):
            continue
        src_data = data.get(source, data)
        if isinstance(src_data, dict) and "geodesic_err_deg" in src_data:
            rows.append({"name": exp_name, "description": "", **src_data})
    return rows


def fmt(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.4f}"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hot3d", "oakink"], default="hot3d")
    parser.add_argument("--metric", default="geodesic_err_deg", help="primary sort metric")
    parser.add_argument("--csv", action="store_true", help="output CSV instead of table")
    args = parser.parse_args()

    rows = load_results(args.source)
    if args.csv:
        print_csv(rows, args.metric)
    else:
        print(f"\nResults — source={args.source}  n_experiments={len(rows)}\n")
        print_table(rows, args.metric)


if __name__ == "__main__":
    main()
