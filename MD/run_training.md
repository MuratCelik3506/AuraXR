# AuraXR — How to Run Full Training
**Last updated:** 2026-05-07  
**Status:** Training was stopped manually. No full epoch has completed on the full dataset yet.  
The log entries in `intentformer_training_log.jsonl` (epochs 1–3, MPJPE ~835→761 mm) are from **benchmark smoke tests** — not real training runs.

---

## When to run this

Run when you have ~7 hours available (leave overnight or over a long break).  
Your Mac stays on, lid can be closed (disable sleep in System Settings → Battery first).

---

## Prerequisites

- Make sure no other training process is already running:
  ```bash
  ps aux | grep 11_train | grep -v grep
  ```
  If you see a result, kill it first: `kill <PID>`

---

## Command — Full Training (ready to copy-paste)

```bash
cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration
source .venv/bin/activate
python3 11_train.py --epochs 100 --batch 256
```

That's it. Everything is already configured:
- Augmentation ON (pos noise + beta perturb + mirror flip)
- workers=6 (parallel data loading)
- Dataset auto-caches 8.8 GB into RAM on startup (~60 sec one-time cost)
- Saves `data/checkpoints/best.pt` whenever val MPJPE improves
- Appends one JSON line per epoch to `data/logs/intentformer_training_log.jsonl`

---

## How to monitor while running

Open a second terminal tab and run:
```bash
tail -f /Users/muratcelik/Desktop/Thesis/Workspace/V3/data/logs/intentformer_training_log.jsonl
```

Each line that appears = one completed epoch. Watch `val_mpjpe` — it should drop from ~800 mm toward the target of **< 50 mm**.

---

## Expected timeline

| Phase | Time |
|-------|------|
| Data cache load (startup) | ~60–90 sec (one time) |
| Epoch 1 (MPS GPU warmup) | ~5 min |
| Epochs 2–100 | ~4 min each |
| **Total** | **~7 hours** |

---

## If you want to resume a stopped run

If `best.pt` exists from a previous run, add `--resume`:
```bash
python3 11_train.py --epochs 100 --batch 256 --resume ../data/checkpoints/best.pt
```
Training will pick up from where it left off.

---

## After training finishes

Run evaluation on the validation split:
```bash
python3 12_evaluate.py
```

Then export a new ONNX for Unity:
```bash
python3 15_export_onnx_unity.py
```

Drop the new `intentformer.onnx` into the Unity project to replace the old one.

---

## Run baselines for comparison (Week 10 task)

Run these separately — each takes ~2–3 hours:
```bash
# GRU baseline (no augmentation for fair comparison)
python3 11_train.py --epochs 100 --batch 256 --model gru --no_aug --resume ""

# Single-frame MLP baseline
python3 11_train.py --epochs 100 --batch 256 --model mlp --no_aug --resume ""
```

Then run `12_evaluate.py` after each to collect MPJPE numbers for the thesis comparison table.

---

## Target metrics (from plan.md)

| Metric | Target | Where measured |
|--------|--------|----------------|
| MPJPE | < 50 mm | `12_evaluate.py` on val split |
| PA-MPJPE | < 25 mm | `12_evaluate.py` |
| Inference latency | < 5 ms | Unity Profiler on Quest 3 |
