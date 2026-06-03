# 03 — Feature Engineering

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source files:**
- `hot3d_exploration/build_dataset.py` (feature construction)
- `hot3d_exploration/hot3d_utils.py` (coordinate math)
- `hot3d_exploration/grip_categories.py` (object features)

---

## Overview

The model takes an 11-dimensional feature vector as input. This encodes:
- **Where** is the object relative to the hand (spatial)
- **What kind** of object it is (grip type + size)

The goal is to give the model just enough information to predict a realistic hand pre-shape, without requiring cameras or complex perception.

---

## Feature Vector Layout (11 dims, V1)

```
Index | Name         | Dims | Description
------|--------------|------|-------------------------------------------
0–2   | direction    |  3   | Unit vector from wrist to object, in wrist-local frame
3     | distance     |  1   | Euclidean distance wrist→object (meters)
4–7   | grip_onehot  |  4   | [Power, Precision, Palmar, Pinch] — exactly one is 1.0
8–10  | bbox         |  3   | Object half-extents [x, y, z] in meters
```

**Total: 11 dims**

---

## V2 Extended Feature (15 dims)

`build_dataset_v2.py` extends the spatial input with the wrist orientation quaternion:

```
Index | Name           | Dims | Description
------|----------------|------|-------------------------------------------
0–2   | direction      |  3   | Same as V1
3     | distance       |  1   | Same as V1
4–7   | wrist_quat     |  4   | Wrist orientation [w, x, y, z] in world frame
8–11  | grip_onehot    |  4   | Same as V1
12–14 | bbox           |  3   | Same as V1
```

**Total: 15 dims** → split as spatial(8) + object(7) for the model

---

## Why These Features?

### Direction (3 dims)
The **unit vector** from the wrist to the object in the **wrist-local frame**. This tells the model which direction the hand is approaching from — reaching from above vs. from the side changes the grip shape significantly.

Computing it:
```python
delta     = obj_pos - wrist_pos               # world frame vector
rel_pos   = rotate_vec(inverse(wrist_quat), delta)  # transform to wrist frame
direction = rel_pos / ||rel_pos||              # normalize to unit vector
```

Separating direction from distance (rather than using raw rel_pos) helps the model learn approach angle independently of proximity.

### Distance (1 dim)
Scalar Euclidean distance in meters. The model uses this to determine the interaction phase:
- > 40cm → excluded from training
- 10–40cm → pre-shape (hand opening/preparing)
- < 10cm → grip (fingers curling)

### Grip One-Hot (4 dims)
The grip category tells the model how to shape the fingers for this type of object:

| Category | Index | Hand shape | Example objects |
|----------|-------|-----------|----------------|
| Power | 0 | Full wrap, all fingers | mug, bottle, can |
| Precision | 1 | Extended, 2-3 fingertip | spoon, marker, spatula |
| Palmar | 2 | Flat palm, fingers spread | plate, phone, keyboard |
| Pinch | 3 | Thumb+index pinch | mouse, puzzle, small holder |

Stored as one-hot: exactly one dimension is 1.0, the rest are 0.0.

### Bounding Box Half-Extents (3 dims)
The object's size in x/y/z (half of total dimensions, in meters). A tall thin bottle needs different finger curling than a wide flat plate, even in the same grip category.

---

## Coordinate System: Wrist-Relative Frame

All spatial features are computed in the **wrist-local frame** (HOT3D convention):

```
HOT3D frame: right-handed, Y-up, Z backward
Wrist frame: wrist position as origin, wrist orientation as axes

Transform chain:
  1. obj_pos_world  = object centroid in world space
  2. delta_world    = obj_pos_world - wrist_pos_world
  3. rel_pos_wrist  = rotate_vec(quat_conjugate(wrist_quat), delta_world)
  4. distance       = ||rel_pos_wrist||
  5. direction      = rel_pos_wrist / distance
```

This makes the feature **invariant to absolute position** — the model doesn't care where in the room the hand is, only the local geometry matters.

---

## Normalization

After computing all features across the training set, per-feature normalization is applied:

```python
# During training dataset creation (build_dataset.py):
feat_mean = features.mean(axis=0)   # shape (11,)
feat_std  = features.std(axis=0)    # shape (11,)
feat_std  = where(feat_std < 1e-8, 1.0, feat_std)   # avoid division by zero

# Applied during training (train.py):
feat_normalized = (feat - feat_mean) / feat_std

# Same stats applied in Unity (AuraXRInferenceManager.cs):
feat[i] = (feat[i] - featMean[i]) / featStd[i];
```

**Stats are stored in:**
- `data/{left,right}/dataset.h5` → `attrs["meta"]` JSON → `feature_mean`, `feature_std`
- `onnx/model_meta_{left,right}.json` → same arrays, copied for Unity

---

## Target: 22 UME Joint Angles

The model predicts 22 UME joint angles (radians). These are also normalized during training:
```python
tgt_normalized = (joint_angles - target_mean) / target_std
```

After inference, Unity denormalizes:
```csharp
angles[i] = model_output[i] * tgtStd[i] + tgtMean[i];
```

Joints 20–21 are always 0 in HOT3D (placeholder joints). They are **excluded from the training loss** via the `ACTIVE_JOINTS` mask (indices 0–19 only).

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Is the wrist-frame transformation code in `hot3d_utils.py` correct? (Check `rotate_vec` and `quat_conjugate` implementations)
- [ ] Should we include absolute position in the feature? (Currently excluded — hand is near-invariant to room location)
- [ ] Is one-hot encoding for grip the right choice? Could we use an embedding?
- [ ] V2 adds wrist quaternion — does this actually help? Compare V1 vs V2 eval results in `results/`
