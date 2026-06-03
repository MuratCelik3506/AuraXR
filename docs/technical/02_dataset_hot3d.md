# 02 — HOT3D Dataset & Dataset Building

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source files:**
- `hot3d_exploration/build_dataset.py`
- `hot3d_exploration/hot3d_utils.py`
- `hot3d_exploration/grip_categories.py`

---

## What Is HOT3D?

HOT3D (Hand-Object Tracking 3D) is a public dataset from Meta Research that contains recordings of people manipulating household objects. It provides:

- **RGB + depth video** from a head-mounted device (Quest 3 or Project Aria)
- **3D hand poses** at each frame using the UmeTrack skeleton format (22 joint angles per hand)
- **6-DoF object poses** for 33 household objects (BOP IDs 1–33)
- **Wrist transform** (position + quaternion) for each hand

**Dataset location:** `data/quest3/` (Quest 3 recordings), `data/aria/` (Aria recordings)

---

## HOT3D Object Categories (BOP IDs 1–33)

The 33 objects cover 4 grip types. This mapping is defined in `grip_categories.py`:

| Grip Type | Index | Examples | Count |
|-----------|-------|---------|-------|
| Power | 0 | mug, bottle, can, carton, dumbbell | 17 |
| Precision | 1 | spoon, spatula, marker | 3 |
| Palmar | 2 | plate, phone, keyboard, remote | 7 |
| Pinch | 3 | mouse, puzzle, small holder | 6 |

Each object also has **bounding box half-extents (x,y,z in meters)** — how wide/tall/deep the object is, halved. These are manually measured and stored in `grip_categories.py`.

---

## UmeTrack Hand Skeleton (22 Joints)

HOT3D uses the **UmeTrack** skeleton with 22 joint angles per hand. The structure is:

```
Per finger: [abduction/flex, MCP, PIP, DIP]  → 4 joints × 5 fingers = 20 joints
Joints 20–21: placeholder (always 0.0 in HOT3D)
```

| Joint index | Finger | Type |
|-------------|--------|------|
| 0–3 | Thumb | CMC-flex, abduction, MCP, DIP |
| 4–7 | Index | abduction, MCP, PIP, DIP |
| 8–11 | Middle | abduction, MCP, PIP, DIP |
| 12–15 | Ring | abduction, MCP, PIP, DIP |
| 16–19 | Pinky | abduction, MCP, PIP, DIP |
| 20–21 | — | Placeholder (always 0) |

Joint angles are stored in **radians**. The model predicts normalized versions of these, then Unity denormalizes back to radians.

---

## Dataset Building: `build_dataset.py`

This script converts raw HOT3D ZIPs into a training-ready HDF5 file.

### Run command
```bash
python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/right/ --hand right
python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/left/  --hand left
```

### Step-by-step process

**Step 1: Find sequences**
`hot3d_utils.find_sequences(data_dir, split="train")` returns a list of sequence directories in `data/quest3/train/`.

**Step 2: Train/val split by sequence**
Sequences are shuffled (seed=42) and split: 15% go to validation, 85% to train. Split is by whole sequence (not by frame) to prevent data leakage.

**Step 3: Extract frames from each sequence**
For each sequence, `extract_frames()` reads two ZIP files:
- `hand_zip` — UmeTrack hand pose trajectories
- `gt_zip` — object poses and metadata

For each timestamp that has both a hand pose and object poses:
1. Read wrist position and quaternion
2. For each visible object, compute: `rel_pos = rotate_vec(wrist_frame_inverse, obj_pos - wrist_pos)`
3. Keep only the **nearest** object
4. Compute distance = `|rel_pos|`
5. Skip if distance > 40cm (`MAX_DISTANCE`)
6. Label: `"grip"` if distance < 10cm, else `"pre_shape"`
7. Build 11-dim feature vector (see [03_feature_engineering.md](03_feature_engineering.md))

**Step 4: Approach augmentation**
For every grip frame (distance < 15cm), generate 6 synthetic samples at larger distances (0.30m, 0.50m, 0.70m, 1.00m, 1.50m, 2.50m). The target pose is a smoothstep blend between the grip pose and a fully open hand (all zeros). These are labeled `"approach"`.

This teaches the model that far away → open hand, close → closed hand, creating smooth anticipatory pre-shaping.

**Step 5: Normalize**
Compute per-feature mean and std over the **training set only**. Store in `dataset.h5` metadata as a JSON string. Validation set is normalized with training statistics (no leakage).

**Step 6: Write HDF5**
```
dataset.h5
  /train/features   (N_train, 11)  float32, gzip compressed
  /train/targets    (N_train, 22)  float32, gzip compressed
  /train/labels     (N_train,)     string   "grip"/"pre_shape"/"approach"
  /train/distances  (N_train,)     float32
  /val/...          same structure
  attrs["meta"]     JSON with norm stats + architecture config
```

---

## Output Datasets

| File | Hand | Notes |
|------|------|-------|
| `data/left/dataset.h5` | Left | V1 features (11 dims) |
| `data/right/dataset.h5` | Right | V1 features (11 dims) |
| `data/left_v5/dataset.h5` | Left | V5 variant |
| `data/right_v5/dataset.h5` | Right | V5 variant |

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Is the UmeTrack joint order correct? (Can cross-check with HOT3D paper)
- [ ] Does the 40cm distance cutoff make sense? (Objects further than arm reach are excluded)
- [ ] Is the approach augmentation strategy sound? Does smoothstep blending match real human behavior?
- [ ] What is the actual frame count for left/right after filtering? (Run `python build_dataset.py` and note output)
