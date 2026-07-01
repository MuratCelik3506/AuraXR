"""Overnight experiment orchestrator.

Reads experiments.json and for each variant:
  1. Chains Phase 1 → 2 → 3 (or a subset per config)
  2. Passes the previous phase checkpoint into the next
  3. Runs offline evaluation after training completes
  4. Writes a per-experiment summary to results/

Usage:
  python3 src/training/orchestrate.py                          # all experiments
  python3 src/training/orchestrate.py --only full              # one experiment
  python3 src/training/orchestrate.py --skip ablation_static_only
  python3 src/training/orchestrate.py --dry-run                # print commands only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.paths import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402

EXPERIMENTS_FILE = _ROOT / "experiments.json"
TRAIN_SCRIPT = _SRC / "training" / "train_grasp.py"
EVAL_SCRIPT = _SRC / "evaluation" / "evaluate.py"


def run(cmd: list[str], dry_run: bool) -> int:
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def ckpt_path(name: str, phase: int, kind: str = "best") -> Path:
    return CHECKPOINT_DIR / name / f"phase{phase}_{kind}.pt"


def load_epoch(path: Path) -> int:
    """Return last completed epoch stored in a checkpoint, or -1."""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return int(ckpt.get("epoch", -1))
    except Exception:
        return -1


def phase_status(name: str, phase: int, target_epochs: int) -> tuple[str, Path | None, int]:
    """Returns (status, resume_ckpt, remaining_epochs).

    status: 'done' | 'partial' | 'fresh'
    """
    best = ckpt_path(name, phase, "best")
    latest = ckpt_path(name, phase, "latest")
    if best.exists():
        return "done", best, 0
    if latest.exists():
        last_epoch = load_epoch(latest)
        remaining = target_epochs - (last_epoch + 1)
        if remaining <= 0:
            return "done", latest, 0
        return "partial", latest, remaining
    return "fresh", None, target_epochs


def build_train_cmd(exp: dict, phase: int, resume_ckpt: Path | None, epochs: int, extra_args: dict) -> list[str]:
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--phase", str(phase),
        "--epochs", str(epochs),
        "--checkpoint-prefix", exp["name"],
    ]
    for key, value in exp.get("args", {}).items():
        cmd += [f"--{key}", str(value)]
    for key, value in extra_args.items():
        flag = f"--{key}"
        if flag not in cmd:
            cmd += [flag, str(value)]
    if resume_ckpt and resume_ckpt.exists():
        cmd += ["--checkpoint", str(resume_ckpt)]
    return cmd


def build_eval_cmd(exp_name: str, ckpt: Path) -> list[str]:
    return [
        sys.executable, str(EVAL_SCRIPT),
        "--checkpoint", str(ckpt),
        "--source", "both",
        "--split", "test",
        "--k", "5",
        "--out", str(RESULTS_DIR / exp_name / "eval.json"),
    ]


def run_experiment(exp: dict, dry_run: bool, extra_args: dict, args_max_phase: int | None = None) -> dict:
    name = exp["name"]
    phases = exp.get("phases", [1, 2, 3])
    phase_epochs = exp.get("phase_epochs", {})

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp.get('description', '')}")
    print(f"  phases: {phases}")
    print(f"{'='*60}", flush=True)

    t_start = time.time()
    phase_results = []
    last_good_ckpt: Path | None = None

    for phase in phases:
        if args_max_phase and phase > args_max_phase:
            print(f"  [phase {phase}] SKIP — max_phase={args_max_phase}", flush=True)
            continue
        target_epochs = int(phase_epochs.get(str(phase), 30))
        status, resume_ckpt, remaining = phase_status(name, phase, target_epochs)

        if status == "done":
            print(f"  [phase {phase}] SKIP — already complete ({ckpt_path(name, phase, 'best')})", flush=True)
            last_good_ckpt = ckpt_path(name, phase, "best")
            phase_results.append({"phase": phase, "status": "skipped"})
            continue

        label = "RESUME" if status == "partial" else "START"
        print(f"  [phase {phase}] {label} — {remaining}/{target_epochs} epochs remaining", flush=True)

        t_phase = time.time()
        cmd = build_train_cmd(exp, phase, resume_ckpt, remaining, extra_args)
        rc = run(cmd, dry_run)
        elapsed = round(time.time() - t_phase, 1)
        phase_results.append({"phase": phase, "status": "ok" if rc == 0 else "failed",
                               "elapsed_s": elapsed, "exit_code": rc})

        if rc != 0:
            print(f"  [phase {phase}] FAILED (exit {rc}) — will resume here next run.", flush=True)
            break  # stop this experiment; next orchestrate.py call resumes from partial checkpoint
        last_good_ckpt = ckpt_path(name, phase, "best")

    run_dir = RESULTS_DIR / name

    # eval — skip if already done
    eval_path = run_dir / "eval.json"
    eval_result: dict = {}
    if last_good_ckpt and last_good_ckpt.exists():
        if eval_path.exists():
            print(f"  [eval] SKIP — {eval_path} already exists", flush=True)
            eval_result = json.loads(eval_path.read_text())
        else:
            rc = run(build_eval_cmd(name, last_good_ckpt), dry_run)
            if rc == 0 and not dry_run and eval_path.exists():
                eval_result = json.loads(eval_path.read_text())

    summary = {
        "name": name,
        "description": exp.get("description", ""),
        "phases_run": phase_results,
        "total_elapsed_s": round(time.time() - t_start, 1),
        "final_checkpoint": str(last_good_ckpt) if last_good_ckpt else None,
        "eval": eval_result,
        "timestamp": datetime.now().isoformat(),
    }
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  [summary] → results/{name}/summary.json", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_FILE)
    parser.add_argument("--only", type=str, default=None, help="run only this experiment name")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="skip these experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--max-phase", type=int, default=None, dest="max_phase",
                        help="stop after this phase (e.g. --max-phase 1 runs only phase 1)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="C4 multi-seed: run each experiment multiple times with these seeds "
                             "(e.g. --seeds 42 123 456). Eval outputs named eval_<name>_seed<N>.json")
    args = parser.parse_args()

    experiments = json.loads(args.experiments.read_text())
    extra_args: dict = {}
    if args.batch:
        extra_args["batch"] = args.batch
    if args.workers:
        extra_args["workers"] = args.workers
    extra_args["window"] = args.window
    extra_args["n_points"] = args.n_points

    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            print(f"No experiment named '{args.only}'")
            sys.exit(1)
    if args.skip:
        experiments = [e for e in experiments if e["name"] not in args.skip]

    all_summaries = []
    t_total = time.time()
    seeds = args.seeds if args.seeds else [None]
    for exp in experiments:
        for seed in seeds:
            seed_extra = dict(extra_args)
            if seed is not None:
                seed_extra["seed"] = seed
                # rename experiment so checkpoints/evals don't overwrite each other
                seed_exp = dict(exp)
                seed_exp["name"] = f"{exp['name']}_seed{seed}"
                summary = run_experiment(seed_exp, args.dry_run, seed_extra, args.max_phase)
            else:
                summary = run_experiment(exp, args.dry_run, extra_args, args.max_phase)
            all_summaries.append(summary)

    master_path = RESULTS_DIR / "master_summary.json"
    if not args.dry_run:
        RESULTS_DIR.mkdir(exist_ok=True)
        master_path.write_text(json.dumps(all_summaries, indent=2))

    elapsed_total = round(time.time() - t_total, 1)
    print(f"\n{'='*60}")
    print(f"ALL DONE  total={elapsed_total}s  experiments={len(all_summaries)}")
    if not args.dry_run:
        print(f"Master summary → {master_path}")
    print("="*60)


if __name__ == "__main__":
    main()
