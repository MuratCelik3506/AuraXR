# 05 — Training & Evaluation

**Last updated:** 2026-06-06

**Source files:**
- `src/train.py`
- `src/evaluate.py`
- `src/evaluate_onnx.py`

---

## Training: `train.py`

### Run command

Training is executed **sequentially** (right then left) to give each job 100% GPU access. Two parallel MPS jobs serialize on the GPU — sequential is equally fast per-total but each job converges faster:

```bash
# Run sequentially so each job gets 100% GPU (two parallel MPS jobs serialize)
python3.11 train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/ --resume && \
python3.11 train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/  --resume
```

All defaults are used unless overridden. Key defaults:
- `--epochs 50000` (early stopping at patience=4000 handles convergence)
- `--batch_size 131072`
- `--lr 5e-3`
- `--hidden_dim 512`, `--embedding_dim 256`, `--dropout 0.25`

### Hardware Auto-Detection
```python
if torch.cuda.is_available():           → CUDA
elif torch.backends.mps.is_available(): → MPS (Apple Silicon)
else:                                   → CPU
```
The entire dataset is preloaded onto the device so each epoch is pure matrix operations — no I/O overhead.

**M2 Max hardware profile:** 12 CPU cores (8P+4E), 30 GPU cores, 32 GB unified memory.  
Two parallel MPS jobs serialize on the GPU (one blocks the other). Sequential execution gives each job 100% GPU throughput at no total-time cost.

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Epochs | 50000 | Upper bound; early stopping (patience=2000) terminates long before this in practice |
| Batch size | 131072 | The entire training split fits in device memory (~200 MB), so one epoch = one gradient step with no I/O overhead |
| Learning rate | 5e-3 | |
| LR schedule | Warmup(200) + CosineAnnealingWarmRestarts(T₀=4000, Tmult=2) | T₀=4000 means first cosine cycle spans epochs 200–4200; patience=4000 allows at least one full cosine cycle before early stopping |
| Weight decay | 3e-4 | |
| Dropout | 0.25 | Trunk dropout (reduced from 0.40 — large dataset with small model risked underfitting at 0.40); spatial encoder uses 0.20 |
| Hidden dim | 512 | ~1.37M params total (increased from 256 — 14° MAE indicated capacity bottleneck) |
| Embedding dim | 256 | Object encoder width (increased from 128 to match hidden_dim scaling) |
| AMP (bfloat16) | ❌ disabled on MPS | Apple's MPS backend does not gain speed from bfloat16 at this matrix size — the precision-reduction overhead cancels the compute gain. Additionally, when the training set size is not divisible by batch size, the trailing batch is smaller and bfloat16's reduced exponent range causes NaN overflow in that batch. CUDA has dedicated bfloat16 tensor cores that avoid both issues. |
| torch.compile | ✅ enabled | `aot_eager` backend — works on MPS without Triton |

### Loss Function: Compound Loss

```python
total_loss = compound_loss + 0.3 * wrist_rot_loss   # wrist weight reduced 0.5→0.3

compound_loss = angle_loss + 0.3 * range_penalty + 0.2 * coupling + 0.15 * cls_loss
wrist_rot_loss = MSE(pred_rot_6d_normalized, tgt_rot_6d_normalized)
```

1. **Angle loss** — per-joint weighted Huber (β=0.5) over active joints 0–19, with 2× weight for grip frames (dist < 10cm). Huber is used instead of MSE because outlier frames (e.g., tracking glitches) produce large errors that would dominate MSE; Huber caps their influence. β=0.5 is tighter than the default (β=1) so moderate errors are still penalized sharply. The 2× grip multiplier exists *on top of* 10× oversampling — oversampling fixes the data imbalance so the model *sees* grip frames equally, while the 2× weight signals that grip errors are twice as costly per sample (closer contact = more visually obvious mistakes).
   > *In plain language: the main penalty — "how wrong are the predicted finger angles?" Each finger joint is weighted separately, with grip-phase frames (hand actually touching something) penalized twice as hard, because a wrong pose at contact is immediately obvious to the eye.*

2. **Range penalty** — penalizes flexion angles outside [0, 2.0 rad]. Human finger flexion is physically bounded; without this, the unconstrained linear output can predict negative angles or hyperextension that look wrong in Unity.
   > *In plain language: fingers can't bend backwards or curl past a fist. This penalty tells the model "if you predict an anatomically impossible angle, you get punished for it" — without hard-wiring limits into the output layer.*

3. **DIP-PIP coupling** — anatomical ratio `DIP ≈ 0.67 × PIP`. In real hands, the distal joint (DIP) flexes roughly two-thirds as much as the proximal joint (PIP) due to tendon anatomy; adding this as a soft constraint prevents unrealistic independent DIP/PIP predictions.
   > *In plain language: in a real finger, the tip joint always bends about two-thirds as much as the middle joint — they're linked by the same tendon. Without this, the model might predict the tip fully curled while the middle is straight, which looks anatomically wrong.*

4. **Grip classifier loss** — auxiliary cross-entropy for grip category prediction (training only). The grip classifier is not used at runtime — it acts as a regularizer that forces the trunk representation to remain grip-category-aware, preventing the spatial encoder from ignoring object type when distance dominates the signal.
   > *In plain language: during training the model is also quizzed "is this a power grip or a pinch grip?" as a side task — not because we need that answer at runtime, but because it stops the model from lazily ignoring what type of object it is when the hand gets very close.*

5. **Wrist rotation loss** — MSE over the normalized 6D palm orientation representation (weight **0.3**, reduced from 0.5). Joint angles drive the visible finger shape; palm orientation is secondary. A higher weight caused the model to over-optimize rotation at the expense of finger MAE.
   > *In plain language: the model also predicts which way the palm should be facing. This penalty ensures that prediction is reasonable, but it's kept intentionally weak (0.3×) so it doesn't distract from the more important task of getting the finger angles right.*

### Data Augmentation (in run_epoch)

```python
f[:, :3]   += 0.02 * randn          # dir_world noise
f[:, 3:6]  += 0.02 * randn          # dir_obj_local noise
f[:, 6]    *= (1 ± 0.10 uniform)    # distance ±10%
f[:, 7]    += 0.05 * randn          # approach_speed additive noise (added)
f[:, 12:15] *= (1 ± 0.05 uniform)   # bbox ±5% (added)
```

### Class Balance
Grip frames (dist < 10cm) are oversampled 10× in `build_dataset.py`. Without oversampling, grip frames make up only ~5–8% of total frames — the model would train almost entirely on pre-shape poses and fail to learn contact-phase finger curling. The 10× multiplier brings grip frames to roughly equal representation with pre-shape frames. Normalization statistics are computed *before* oversampling to avoid bias (oversampled frames are copies, not new data).

### Checkpointing
Saves `best_model.pt` when validation loss improves, plus:
- `training_log.json` — per-epoch train/val loss and learning rate
- `model_meta.json` — normalization stats + architecture config

`--resume` flag loads `best_model.pt` before training starts, allowing continuation from a prior checkpoint with a fresh optimizer (useful for LR schedule changes without losing learned weights).

---

## Evaluation: `evaluate.py`

```bash
python evaluate.py --checkpoint ../checkpoints/right/ --data_dir ../data/right/
python evaluate.py --checkpoint ../checkpoints/left/  --data_dir ../data/left/
```

Computes per-joint MAE (degrees), per-phase MAE (pre-shape / grip), and per-grip-category MAE.

---

## ONNX Evaluation: `evaluate_onnx.py`

```bash
python evaluate_onnx.py --hand right left
```

Runs the exported ONNX on the validation split and verifies numerical match with PyTorch (tolerance < 1e-5). Reports per-finger, per-phase, and per-grip-category MAE.

---

## Live Simulation: `test_onnx_live.py`

```bash
python test_onnx_live.py
```

Simulates approach trajectories (40cm → 2cm) for each grip category. Checks that PIP joints increase monotonically as distance decreases.

