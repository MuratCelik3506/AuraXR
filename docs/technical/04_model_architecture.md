# 04 — Model Architecture

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source files:**
- `hot3d_exploration/model.py` (V1)
- `hot3d_exploration/model_v2.py` (V2)

---

## Design Goal

The model must be:
1. **Accurate enough** to produce realistic, phase-aware pre-shapes
2. **Tiny enough** to run in real-time on Quest 3 (Snapdragon XR2 Gen 2) at 30+ fps
3. **Structured** so spatial and object information are encoded separately before fusion

---

## V1 Architecture (`model.py`, ~54,000 params)

```
spatial_input (B, 4)          object_input (B, 7)
[dir_x, dir_y, dir_z, dist]   [grip_oh(4), bbox(3)]
        │                              │
  FC(4→128) + ReLU             FC(7→128) + ReLU
  FC(128→64) + ReLU            FC(128→64) + ReLU
        │                              │
        └──────── cat(128) ────────────┘
                     │
               FC(128→128) + ReLU + Dropout(0.2)
               FC(128→128) + ReLU + Dropout(0.2)
               FC(128→22)  + Tanh
                     │
             joint_angles (B, 22)  ← normalized [-1, 1]
```

**Key design decisions:**
- Two separate encoders prevent the object type from "leaking" into spatial processing
- Tanh output bounds predictions to [-1, 1] (easy to denormalize)
- Joints 20–21 are trained to 0 (they are always 0 in HOT3D, excluded from loss)

---

## V2 Architecture (`model_v2.py`, ~210,000 params)

```
spatial_input (B, 8)                    object_input (B, 7)
[dir(3), dist(1), wrist_quat(4)]        [grip_oh(4), bbox(3)]
        │                                       │
  FC(8→256) + LayerNorm + ReLU           FC(7→128) + ReLU
  Dropout(0.15)                          FC(128→128) + ReLU
  FC(256→128) + ReLU                             │
        │                                       │
        └──────────── cat(256) ─────────────────┘
                          │
                    FC(256→256) + ReLU + Dropout(0.3)
                    FC(256→256) + ReLU + Dropout(0.15)
                          │
              ┌───────────┼───────────┐
              │           │           │  (5 finger heads)
         Thumb Head   Index Head   ... Pinky Head
         FC(256→64)   FC(256→64)      FC(256→64)
         ReLU         ReLU             ReLU
         FC(64→4)     FC(64→4)         FC(64→4)
         Tanh         Tanh             Tanh
              │           │           │
              └───────────┴───────────┘
                     cat(20)
                   + zeros(2)           ← joints 20–21 always 0
                          │
                  joint_angles (B, 22)
```

**Improvements over V1:**

| Feature | V1 | V2 |
|---------|----|----|
| Spatial input dims | 4 | 8 (+wrist quaternion) |
| Hidden size | 128 | 256 |
| Embedding size | 64 | 128 |
| Output head | Single 22-dim | 5 per-finger × 4-dim |
| Normalization | None | LayerNorm after spatial encoder |
| Parameters | ~54k | ~210k |
| Dropout | 0.2 | 0.3 (trunk), 0.15 (spatial) |

**Why wrist quaternion (V2)?**
The hand orientation (pronated vs. supinated, tilted) strongly determines grip shape. If you reach for a mug from above (palm down) vs. from the side (palm inward), the finger curl pattern is different. V1 could not distinguish these because it only had direction + distance. V2 adds the full wrist quaternion.

**Why per-finger heads (V2)?**
A single 22-dim output head forces all fingers to share the same gradient signal. In reality, each finger has semi-independent behavior. Per-finger heads let each finger specialize, and they can be individually analyzed in evaluation (e.g., "thumb error is higher than index finger error").

**Why LayerNorm (V2)?**
The wrist quaternion (range ≈ [-1, 1]) and direction (range ≈ [-1, 1]) have similar scale, but distance (range ≈ [0, 0.4]) is much smaller. LayerNorm normalizes the hidden representation after the spatial encoder, stabilizing training.

---

## Currently Deployed Architecture

The ONNX models in `onnx/` (`auraxr_left.onnx`, `auraxr_right.onnx`, and the v6 variants) all use the **V1 architecture** with `spatial_input_dim=4`. This is confirmed by the `architecture` field in `onnx/model_meta_*.json`:
```json
{"spatial_input_dim": 4, "object_input_dim": 7, "output_dim": 22, "hidden_dim": 128, "embedding_dim": 64}
```
V2 (`model_v2.py`) has been trained but not yet exported and deployed.

---

## ONNX Input/Output Spec

Both V1 and V2 export the same ONNX interface:

```
Input  "spatial_input":  shape [batch, 4]   (V1) or [batch, 8]  (V2)  — normalized
Input  "object_input":   shape [batch, 7]   — normalized
Output "joint_angles":   shape [batch, 22]  — normalized, Tanh range [-1, 1]
```

The batch dimension is dynamic (dynamic_axes). In Unity, batch=1 (one hand per inference call).

---

## Active Joints

Both models define:
```python
ACTIVE_JOINTS = list(range(20))  # indices 0–19
```

Joints 20–21 are excluded from the loss during training. They are set to 0 in the V2 model via a registered zero buffer. In V1 they are predicted but not used (they converge to ~0 naturally since the target is always 0).

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Is the V2 architecture diagram correct? Verify against `model_v2.py` lines 57–97.
- [ ] Does the parameter count (~210k) fit Quest 3 memory budget?
- [ ] Are V5 and V6 checkpoint variants just hyperparameter sweeps of V1, or of V2? (Check `checkpoints/` directory names)
- [ ] Should we add an ablation table comparing V1 / V2 / V5 / V6 MAE scores here?
