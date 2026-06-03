# 05 — Training & Evaluation

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source files:**
- `hot3d_exploration/train.py`
- `hot3d_exploration/evaluate.py`
- `hot3d_exploration/evaluate_onnx.py`
- `results/eval_left.json`, `results/eval_right.json`

---

## Training: `train.py`

### Run command
```bash
python train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/
python train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/
```

### Hardware Auto-Detection
The script auto-selects the best available device:
```python
if torch.cuda.is_available():  → CUDA (GPU)
elif torch.backends.mps.is_available():  → MPS (Apple Silicon)
else:  → CPU
```
On an M2 Max, training 500 epochs takes approximately **3 minutes**. The entire dataset is preloaded onto the device (GPU/MPS) so each epoch is just matrix operations — no data loading overhead.

### Hyperparameters (defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 500 | Full training run |
| Batch size | 4096 | Large batch, GPU-friendly |
| Learning rate | 1e-3 | Adam optimizer |
| Weight decay | 1e-4 | L2 regularization |
| Grip weight | 6.0 | Loss multiplier for grip frames (< 10cm) |
| Approach weight | 2.0 | Loss multiplier for synthetic approach frames (> 45cm) |
| Warmup epochs | 20 | Linear LR warmup |
| Dropout | 0.40 | Applied in head layers |

### Loss Function: Weighted Huber

```python
def weighted_huber(pred, target, weights, active, beta=0.5):
    diff = |pred[:, active] - target[:, active]|
    huber = where(diff < beta, 0.5 * diff² / beta, diff - 0.5 * beta)
    return (weights * huber.mean(dim=-1)).mean()
```

Three components:
1. **Huber loss** (smooth L1): Less sensitive to outlier poses than MSE. `beta=0.5` means linear for errors > 0.5 rad (≈ 28°), quadratic below.
2. **Active joints mask**: Joints 20–21 (always 0) excluded from loss.
3. **Sample weights**: Grip frames (only ~2% of data) get 6× weight so the model learns the final grasp well. Synthetic approach frames (> 45cm) get 2× weight to reinforce the open-hand-at-distance behavior.

### Learning Rate Schedule
```
Epochs 1–20:   Linear warmup from 0 → 1e-3
Epochs 21–500: Cosine decay from 1e-3 → ~0
```
This prevents large gradient steps at the start (when weights are random) and allows fine convergence at the end.

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents exploding gradients, important for stable training with weighted loss.

### Checkpointing
The script saves `best_model.pt` whenever validation loss improves. It also saves:
- `training_log.json` — per-epoch train/val loss and learning rate
- `model_meta.json` — normalization statistics + architecture config

---

## Evaluation: `evaluate.py`

Loads `best_model.pt` and the validation split, then computes:

### Per-Joint MAE
Mean Absolute Error in degrees for each of the 20 active joints.
```
MAE_joint[i] = mean(|pred[i] - target[i]|) × (180/π)
```

### Per-Phase Breakdown
Validation set is split by distance:
- **Pre-shape:** 10–40cm — how well does the hand open for approaching objects?
- **Grip:** < 10cm — how accurate is the final grasp?

Grip accuracy is more important perceptually (this is what the user sees during contact).

### Per-Grip-Category Breakdown
MAE split by object grip type (Power / Precision / Palmar / Pinch). Useful for identifying which object categories the model struggles with.

---

## ONNX Validation: `evaluate_onnx.py`

After export, validates that the ONNX model produces bit-identical outputs to PyTorch:
```python
max_diff = abs(onnx_output - pytorch_output).max()
assert max_diff < 1e-5   # numerical tolerance
```
This ensures the export didn't introduce any numerical errors.

---

## Checkpoints & Variants

| Directory | Hand | Notes |
|-----------|------|-------|
| `checkpoints/left/` | Left | V1 baseline |
| `checkpoints/right/` | Right | V1 baseline |
| `checkpoints/left_v5/` | Left | V5 hyperparameter variant |
| `checkpoints/right_v5/` | Right | V5 hyperparameter variant |
| `checkpoints/left_v6/` | Left | V6 (approach-aware, latest) |
| `checkpoints/right_v6/` | Right | V6 (approach-aware, latest) |

V6 is the currently deployed model in Unity.

---

## Results Files

`results/eval_left.json` and `results/eval_right.json` contain the full per-joint MAE breakdown.
`results/onnx_eval_left.json` and `results/onnx_eval_right.json` confirm ONNX numerical match.

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] What are the actual MAE numbers from `results/eval_right.json`? Fill them in here.
- [ ] Is the grip_weight=6.0 justified? (Grip frames are ~2% of data — 6× brings effective weight to ~12%)
- [ ] Is Huber loss the right choice vs. MAE? (MAE is interpretable in degrees; Huber is more robust to outliers)
- [ ] Are V5 vs V6 difference documented somewhere? What changed between them?
- [ ] Should we add a learning curve plot from `training_log.json`?
