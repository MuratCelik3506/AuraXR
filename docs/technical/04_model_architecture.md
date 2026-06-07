# 04 — Model Architecture

**Last updated:** 2026-06-06

**Source file:** `src/model.py`

---

## Design Goal

The model must be:
1. **Accurate enough** to produce realistic, phase-aware pre-shapes
2. **Tiny enough** to run in real-time on Quest 3 (Snapdragon XR2 Gen 2) at 30+ fps
3. **Structured** so spatial and object information are encoded separately before fusion

---

## Architecture (`src/model.py`, ~1.37M params)

```
spatial_input (B, 8)                        object_input (B, 7)
[dir_world(3), dir_obj_local(3),            [grip_oh(4), bbox(3)]
 dist(1), approach_speed(1)]
        │                                          │
  FC(8→512) + LayerNorm + ReLU             FC(7→256) + LayerNorm + ReLU
  Dropout(0.125)                           FC(256→256) + ReLU
  FC(512→256) + ReLU                               │
        │                                          │
        └──────────────── cat(512) ────────────────┘
                               │
                         FC(512→512) + ReLU + Dropout(0.25)
                         FC(512→512) + ReLU + Dropout(0.125)
                         FC(512→512) + ReLU + Dropout(0.0625)   ← 3rd layer (added)
                               │
        ┌──────────────────────┼──────────────────────────┐
        │                      │                          │
  ┌─────┴─────┐        ┌───────┴──────────┐      ┌───────┴────────┐
  │ 5 × Finger│        │ Wrist Rotation   │      │ Grip Classifier│
  │   Heads   │        │     Head         │      │ (train only,   │
  │ (thumb…   │        │  FC(512→64)      │      │  not in ONNX)  │
  │  pinky)   │        │  ReLU Dropout    │      │  FC(512→32)    │
  │ FC(512→128)│       │  FC(64→6)        │      │  ReLU          │
  │  ReLU     │        │        │         │      │  FC(32→4)      │
  │ Dropout   │        │  wrist_rot_6d    │      └────────────────┘
  │ FC(128→4) │        │    (B, 6)        │
  └─────┬─────┘        └──────────────────┘
        │  ↑ finger_hidden: 64→128 (added)
     cat(20) + zeros(2)
        │
  joint_angles (B, 22)
```

**Key design decisions:**

- **Two separate encoders** prevent object type from "leaking" into spatial processing — if merged early, the model could exploit grip-category shortcuts instead of learning geometry
  > *In plain language: if you mix "where is the object" and "what is the object" too early, the model might learn lazy shortcuts like "whenever it's a mug, close fingers" instead of learning the actual geometry of the approach. Keeping them separate forces each branch to do its own job first.*

- **Per-finger heads** allow finger-specific weight allocation in the loss; each finger has independent learned parameters so thumb curling is not constrained by pinky curling. `finger_hidden=128` (raised from 64) gives each head enough capacity to learn the full flexion range.
  > *In plain language: thumb and pinky move very differently. If one shared output layer predicted all fingers together, getting the thumb right might come at the cost of the pinky. Five separate heads let each finger be trained and weighted independently — and each head is now large enough to represent the full range of motion.*

- **`wrist_rotation_head`** predicts palm orientation as 6D rotation relative to approach direction — decoded in Unity via Gram-Schmidt orthogonalization
  > *In plain language: the model also predicts which way the palm should be facing — not just which way the fingers curl. "6D rotation" is just a math-friendly way to represent orientation that doesn't have the discontinuity problems of angles.*

- **Linear (no Tanh/Sigmoid) output** — Tanh saturates gradients near ±1, which would make it hard to learn the full [0, 2.0 rad] flexion range; the compound loss range penalty enforces plausible values without bounding the activations
  > *In plain language: Tanh squashes everything between -1 and +1, which makes it hard for the model to predict large finger bend angles (a fully closed fist is ~2 radians). Instead, the output is unconstrained and a separate penalty during training discourages anatomically impossible angles.*

- **Auxiliary `grip_classifier` head** is active only during training and not exported to ONNX — its sole purpose is to regularize the trunk representation so it stays grip-category-aware. Unity has no use for a grip-category prediction at runtime.
  > *In plain language: during training, the shared middle part of the network is also asked "what grip type is this?" as a side task — not because we need that answer, but because it stops the network from forgetting about grip type when distance dominates the signal. This side task is discarded before shipping to Unity.*

---

## ONNX Input/Output Spec

```
Input  "spatial_input":  shape [batch,  8] — [dir_world(3), dir_obj_local(3), dist(1), approach_speed(1)], normalized
Input  "object_input":   shape [batch,  7] — [grip_oh(4), bbox(3)], normalized
Output "joint_angles":   shape [batch, 22] — normalized; denorm with target_mean/std from model_meta.json
Output "wrist_rot_6d":   shape [batch,  6] — normalized; denorm with wrist_rot_mean/std; then Gram-Schmidt → Quaternion
```

Batch dimension is dynamic. In Unity, batch=1 (one hand per inference call).

---

## Active Joints

```python
ACTIVE_JOINTS = list(range(20))  # indices 0–19
```

Joints 20–21 are placeholder slots in the UmeTrack skeleton that HOT3D always records as 0.0 — they do not correspond to any physical joint. They are excluded from the loss during training and set to 0 via a registered zero buffer in the model, so the ONNX output always has exactly 22 values (matching the fixed-size Unity array) without requiring special-case handling in C#.

---

## split_feature

```python
@staticmethod
def split_feature(feature: torch.Tensor):
    """Split (B, 15) feature vector into spatial (B, 8) and object (B, 7)."""
    return feature[:, :8], feature[:, 8:]
```

Used in `train.py`, `evaluate.py`, and `simulate.py`.

