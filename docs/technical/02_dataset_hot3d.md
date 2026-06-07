# 02 — HOT3D Dataset & Dataset Building

**Last updated:** 2026-06-06

**Source files:**
- `src/build_dataset.py`
- `src/hot3d_utils.py`
- `src/grip_categories.py`

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

This script converts raw HOT3D ZIPs into a training-ready HDF5 file. HDF5 was chosen because it stores all splits, feature matrices, and normalization statistics in a single binary file with gzip compression, and `h5py` supports random-access reads — PyTorch DataLoader can fetch individual rows from disk without loading the entire dataset into RAM.

### Run command
```bash
python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/right/ --hand right
python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/left/  --hand left
```

### Step-by-step process

**Step 1: Find sequences**
`hot3d_utils.find_sequences(data_dir, split="train")` returns a list of sequence directories in `data/quest3/train/`.

**Step 2: Train/val split by sequence**
Sequences are shuffled (seed=42, fixed for reproducibility) and split: 15% go to validation, 85% to train. The split is by whole sequence — not by frame — because consecutive frames within a sequence share the same scene, lighting, and object positions. Splitting by frame would leak near-identical frames into both sets, making validation metrics appear falsely optimistic.

**Step 3: Extract frames from each sequence**
For each sequence, `extract_frames()` reads two ZIP files:
- `hand_zip` — UmeTrack hand pose trajectories
- `gt_zip` — object poses and metadata

For each timestamp that has both a hand pose and object poses:
1. Read wrist position and quaternion
2. For each visible object, compute: `rel_pos = rotate_vec(wrist_frame_inverse, obj_pos - wrist_pos)`
3. Keep only the **nearest** object
4. Compute distance = `|rel_pos|`
5. Skip if distance > 40cm (`MAX_DISTANCE`) or `hand_confidence < 0.70`. The 40cm cutoff matches the maximum distance at which the hand visibly adapts its shape — beyond arm reach, the hand stays in neutral and adds no useful signal. The 0.70 confidence threshold filters frames where UmeTrack was interpolating rather than directly tracking; below this level the joint angles are unreliable.
6. Label: `"grip"` if distance < 10cm, else `"pre_shape"`. At ~10cm the fingertips start making contact with the object surface; this boundary separates the contact phase from the pre-shaping phase.
7. Build 15-dim feature vector (see [03_feature_engineering.md](03_feature_engineering.md))

**Step 4: Normalize**
Compute per-feature mean and std over the **training set only**. Store in `dataset.h5` metadata as a JSON string. Validation set is normalized with training statistics (no leakage).

**Step 5: Write HDF5**
```
dataset.h5
  /train/features   (N_train, 15)  float32, gzip compressed
  /train/targets    (N_train, 22)  float32, gzip compressed
  /train/wrist_rot_6d (N_train, 6) float32, gzip compressed
  /train/labels     (N_train,)     string   "grip" | "pre_shape"
  /train/distances  (N_train,)     float32
  /val/...          same structure
  attrs["meta"]     JSON with norm stats + architecture config
```

**Class balance:** Grip frames (distance < 10cm) make up only ~5–8% of the raw dataset — most of the recording time captures the hand approaching or moving away from objects. Without correction, the model trains almost entirely on pre-shape data and never learns contact-phase finger curling. Grip frames are therefore repeated 10× in the training split. The 10× multiplier brings grip frames to roughly equal representation with pre-shape frames. Normalization statistics are computed *before* oversampling because oversampled copies are not new data — including them would bias the mean/std toward the oversampled class. Validation set is never oversampled.
