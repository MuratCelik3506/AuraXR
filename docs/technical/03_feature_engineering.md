# 03 — Feature Engineering

**Last updated:** 2026-06-06

**Source files:**
- `src/build_dataset.py` (feature construction)
- `src/hot3d_utils.py` (coordinate math + quaternion helpers)
- `src/grip_categories.py` (object features)

---

## Simple Explanation (Non-Technical)

Imagine you're reaching for a coffee mug on the table. Without looking at your hand, you already know roughly how to shape it: fingers start opening up, curling into a cylinder shape, before you even touch the mug. You do this automatically because your brain knows:

1. **Where the mug is** — it's in front of you, slightly to the right, 20 cm away
2. **Which side you're approaching from** — you're coming from above, not from the side
3. **How fast you're moving** — you're reaching quickly, not slowly placing your hand
4. **What kind of object it is** — it's a mug (cylindrical), so a power grip is needed
5. **How big it is** — it's a medium-sized mug, so fingers need to open a certain amount

The model gets exactly this information as a list of 15 numbers, and from those numbers predicts what shape the hand should be. Nothing more, nothing less.

---

## Overview

The model takes a 15-dimensional feature vector as input. This encodes:
- **Where** the object is relative to the hand, and which surface is being approached (spatial)
- **How fast** the hand is moving toward the object (temporal)
- **What kind** of object it is (grip type + size)

---

## Feature Vector Layout (15 dims)

```
Index | Name          | Dims | Description
------|---------------|------|-------------------------------------------
0–2   | dir_world     |  3   | Unit vector from wrist to object, HOT3D world frame
3–5   | dir_obj_local |  3   | Same vector rotated into the object's local frame
6     | distance      |  1   | Euclidean distance wrist→object (meters)
7     | approach_speed|  1   | dot(wrist_velocity, dir_world) — positive = moving toward object
8–11  | grip_onehot   |  4   | [Power, Precision, Palmar, Pinch] — exactly one is 1.0
12–14 | bbox          |  3   | Object half-extents [x, y, z] in meters
```

**Total: 15 dims** → split as spatial(8) + object(7) for the model

---

## Feature Descriptions

### dir_world (3 dims)
> **In plain language:** "Which direction is the object from my hand?" — e.g. straight ahead, up-left, from above. Stored as a unit arrow pointing from wrist to object, measured in the room's fixed coordinate system (not relative to where the hand is pointing).

Unit vector from wrist to object in HOT3D world frame. Captures approach angle (from above vs. from the side) without depending on wrist rotation.

```python
delta     = obj_pos - wrist_pos    # world frame vector
dir_world = delta / ||delta||
```

**Why NOT wrist-local:** Canonical wrist-local frame (`LookRotation(delta)` → `inv(q) * delta`) always yields `(0,0,1)` — trivially useless. Real wrist quaternion from HOT3D (UME tracker) would create a training-inference mismatch with Unity's controller tracking system.

### dir_obj_local (3 dims)
> **In plain language:** "Which face of the object am I approaching?" — the front of a phone vs. the back, the top of a mug vs. the handle side. The object has its own "up" and "forward" built in; this feature says where the hand is coming from *relative to the object's own orientation*.

Same delta vector, rotated into the object's own coordinate frame using the object's world quaternion `q_wo`. Tells the model which *face or surface* of the object the hand is approaching.

```python
q_obj_inv     = quat_conjugate(obj["quat_world"])   # from dynamic_objects.csv
dir_obj_local = rotate_vec(q_obj_inv, delta / dist)
```

**Why this is safe:** Both HOT3D (CSV `q_wo`) and Unity (`transform.rotation`) carry the same physical object rotation — no tracking-system mismatch. The `ToHOT3D()` coordinate conversion in Unity handles the frame difference.

**Example:** Approaching a phone from the screen side vs. the back produces different `dir_obj_local` values, allowing the model to predict different finger configurations.

### distance (1 dim)
> **In plain language:** "How far away is the object right now?" — in metres. This single number determines whether the hand is in pre-shaping mode (starting to open up) or grip mode (fingers actively closing).

Scalar Euclidean distance in meters. Determines interaction phase:
- > 40cm → excluded from training
- 10–40cm → pre-shape (hand opening/preparing)
- < 10cm → grip (fingers curling)

### approach_speed (1 dim)
> **In plain language:** "Am I moving toward the object right now, and how fast?" — positive means getting closer, negative means pulling away. A fast reach might produce a different pre-shape than a slow, careful placement.

Projection of the wrist velocity vector onto `dir_world`. Positive = hand moving toward object; negative = moving away.

```python
vel_world      = (wrist_pos - prev_wrist_pos) / dt
approach_speed = np.dot(vel_world, dir_world)
```

**Why safe:** Uses only position differences — no rotation data. Unity computes an equivalent value via `OVRInput.GetLocalControllerVelocity()` → HOT3D frame → dot product.

**Implementation note:** `ume_traj` is an unordered dict — `sorted(ume_traj.items())` is mandatory for correct velocity computation.

### grip_onehot (4 dims)
> **In plain language:** "What grip style does this object require?" — wrapping your whole hand around a bottle is different from pinching a pen between two fingers. One of four categories is set to 1, the rest are 0.

Grip category one-hot encoding:

| Category | Index | Hand shape | Example objects |
|----------|-------|-----------|----------------|
| Power | 0 | Full wrap, all fingers | mug, bottle, can |
| Precision | 1 | Extended, 2–3 fingertip | spoon, marker, spatula |
| Palmar | 2 | Flat palm, fingers spread | plate, phone, keyboard |
| Pinch | 3 | Thumb+index pinch | mouse, puzzle, small holder |

### bbox half-extents (3 dims)
> **In plain language:** "How big is the object in each direction?" — a tall thin bottle and a wide flat plate both need a power grip, but fingers must open very differently. These three numbers give width, height, and depth (each halved from the full size).

Object size in x/y/z (half of total dimensions, in meters). A tall thin bottle needs different finger curling than a wide flat plate even in the same grip category.

---

## Coordinate System: HOT3D World Frame

All spatial features are computed in the HOT3D world frame (right-handed, Y-up, Z backward):

```
obj_pos_world  = object centroid in world space (HOT3D)
delta_world    = obj_pos_world - wrist_pos_world
distance       = ||delta_world||
dir_world      = delta_world / distance
dir_obj_local  = rotate(inv(obj_quat_world), dir_world)
```

**Unity ↔ HOT3D conversion:**
```
pos_hot3d  = (x, y, -z)           ← negate Z
quat_hot3d = (qx, qy, -qz, qw)   ← negate Z imaginary component
```

---

## Data Quality Filter

Frames where `hand_confidence < 0.70` are discarded during dataset construction. Low-confidence frames contain noisy or interpolated joint angles that would corrupt training.

---

## Normalization

Per-feature z-score normalization is computed over the training set:

```python
feat_mean = features.mean(axis=0)   # shape (15,)
feat_std  = features.std(axis=0)
feat_std  = where(feat_std < 1e-8, 1.0, feat_std)

# Applied during training and inference:
feat_normalized = (feat - feat_mean) / feat_std
```

**Stats are stored in:**
- `data/{left,right}/dataset.h5` → `attrs["meta"]` JSON
- `onnx/model_meta_{left,right}.json` → same arrays, copied for Unity (15-element arrays)

---

## Training Targets

### Target 1: 22 UME Joint Angles

The model predicts 22 UME joint angles (radians), z-score normalized:
```python
tgt_normalized = (joint_angles - target_mean) / target_std
```

After inference, Unity denormalizes:
```csharp
angles[i] = model_output[i] * tgtStd[i] + tgtMean[i];
```

Joints 20–21 are always 0 in HOT3D (placeholder joints) and are excluded from the training loss via the `ACTIVE_JOINTS` mask (indices 0–19 only).

### Target 2: Wrist Rotation 6D (palm orientation)

A second target encodes **how the palm should be oriented** when reaching for the object. It is stored as a **6D continuous rotation representation** (Zhou et al., 2019) — the first two columns of the relative rotation matrix — which is continuously differentiable and avoids quaternion antipodal discontinuities.

**Computation (in `build_dataset.py` via `hot3d_utils.wrist_rot_to_6d`):**

```python
# 1. Convert wrist quaternion and direction to Unity frame (negate Z)
q_wrist_unity = q_wrist_hot3d * [1, 1, 1, -1]   # wxyz: negate z-component
dir_unity     = dir_world_hot3d * [1, 1, -1]     # negate z

# 2. Canonical rotation: local Z = approach direction
q_canonical = look_rotation(dir_unity)            # Python LookRotation ≡ Unity's

# 3. Relative rotation: how much does the wrist deviate from canonical?
q_rel = quat_conjugate(q_canonical) ⊗ q_wrist_unity

# 4. 6D: first two columns of rotation matrix from q_rel
R     = quat_to_rot_mat(q_rel)
rot6d = concat(R[:, 0], R[:, 1])                  # shape (6,)
```

**Why relative to approach direction:** The model learns "when approaching a mug from the side, tilt palm X degrees" without needing an absolute world frame. This makes the feature rotation-invariant w.r.t. scene orientation — the same manipulation skill generalizes to any table position.

**Why safe:** The wrist quaternion from HOT3D (`q_wxyz`) is only used as a training target. At inference time, Unity predicts the rotation entirely from the 6D output, decoded via Gram-Schmidt. No HOT3D tracking system quaternion is read at runtime.

**Stats stored in `model_meta.json`:**
```json
"wrist_rot_mean": [6 values],
"wrist_rot_std":  [6 values]
```
