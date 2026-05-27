# AuraXR Pipeline Plan — Python Scripts

This document describes every Python script needed to build, train, evaluate, and export the AuraXR hand pose prediction model. Scripts are ordered by execution dependency: each script requires the ones before it.

---

## Utility Files (Run Anytime)

These are shared modules imported by other scripts. They are not run directly.

### `hot3d_utils.py` *(existing — update as needed)*

**Purpose:** Low-level HOT3D data access helpers.

**Contains:**
- ZIP reader for `hand_data.zip` → parses `umetrack_hand_pose_trajectory.jsonl` and `mano_hand_pose_trajectory.jsonl`
- ZIP reader for `ground_truth.zip` → parses `dynamic_objects.csv` (object 6DoF poses)
- Frame iterator with timestamp alignment across hand + object data
- `rotate_vec(q_wxyz, v)` — rotates a 3D vector by a quaternion
- `quat_conjugate(q_wxyz)` — conjugate of a unit quaternion
- BOP ID → grip category + bbox lookup (`OBJ_BBOX` table, 33 objects)

**Data source:** `data/quest3/{split}/{seq_id}/` ZIP files (Quest3 only, downloaded via `06_download_annotations.py`)

**Used by:** `build_dataset.py`, `evaluate.py`

---

### `hot3d_dataset.py` *(existing — rewrite)*

**Purpose:** PyTorch Dataset that streams from the HDF5 file built by `build_dataset.py`.

**Contains:**
- `HOT3DDataset(hdf5_path, split, normalise)` — loads train/val split from HDF5
- Applies feature/target normalisation from stored stats
- Returns `(feature_11, target_22)` tensors per sample

**Used by:** `train.py`, `evaluate.py`

---

### `grip_categories.py` *(new)*

**Purpose:** Maps HOT3D object IDs to grip category and physical dimensions.

**Input:** HOT3D `object_library.json`

**Output (in-memory):** Dict mapping object_id → `{grip_category: int, size_xyz: [float, float, float]}`

**Grip categories:**
| Category | Index | Example Objects |
|----------|-------|----------------|
| Power    | 0     | cup, bottle, container |
| Precision| 1     | spoon, pen, spatula |
| Palmar   | 2     | plate, keyboard, phone |
| Pinch    | 3     | mouse, puzzle, small box |

**One-hot encoding:** Grip index → `[1,0,0,0]`, `[0,1,0,0]`, `[0,0,1,0]`, `[0,0,0,1]`

**Run:** `python grip_categories.py` (runs self-test, prints category table)

---

## Step 1 — Build Dataset

### `build_dataset.py` *(new)*

**Purpose:** Extracts all usable frames from HOT3D and writes the training dataset to disk.

**What it does, step by step:**
1. Scans `data/quest3/train/` and `data/quest3/test/` for sequence directories (Quest3 only)
2. Assigns train/val split: test participants (P0004/5/6/8/16/20) excluded; remaining 70/15 within train participants
3. For each sequence, opens `hand_data.zip` and `ground_truth.zip`
4. Iterates all frames with valid hand annotations:
   - **Skips frames where hand-object distance > 40cm**
   - Reads wrist position and quaternion from `umetrack_hand_pose_trajectory.jsonl`
   - Reads nearest object centroid (world space) from `dynamic_objects.csv`
   - Computes relative position in wrist frame:
     ```python
     delta = obj_centroid_world - wrist_pos_world
     rel_pos = rotate_vec(quat_conjugate(wrist_q), delta)  # (3,)
     ```
   - Computes distance: `distance = ‖rel_pos‖`
   - Maps BOP object ID → grip category one-hot (4) using `OBJ_BBOX` table
   - Reads bbox half-extents from `OBJ_BBOX` lookup → 3 values
   - Assembles feature vector: `[rel_pos(3), grip_onehot(4), bbox(3), distance(1)]` = **11 values**
   - Reads UmeTrack joint angles from same file → **22 values** (target)
   - Labels frame: `pre_shape` (10–40cm) or `grip` (<10cm)
5. Computes normalization stats (mean, std) over train split only
6. Saves to single HDF5 file with train/val groups

**Inputs:**
- `data/quest3/` — downloaded Quest3 ZIP files

**Outputs:**
- `data/right/dataset.h5` — HDF5 file with groups `train/` and `val/`:
  - `features` — shape `(N, 11)`, float32
  - `targets` — shape `(N, 22)`, float32
  - `labels` — shape `(N,)`, bytes: `b"pre_shape"` or `b"grip"`
  - `distances` — shape `(N,)`, float32
  - Attribute `meta` — JSON string with norm stats
- `data/left/dataset.h5` — same structure for left hand

**Run:**
```bash
python build_dataset.py --data_dir data/quest3/ --output_dir data/right/ --hand right
python build_dataset.py --data_dir data/quest3/ --output_dir data/left/  --hand left
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | required | Path to `data/quest3/` directory |
| `--output_dir` | required | Where to save dataset.h5 |
| `--hand` | `right` | Which hand to extract: `right` or `left` |
| `--max_distance` | 0.40 | Max hand-object distance in meters (frames beyond this are skipped) |

**Notes:**
- Run **twice**: once with `--hand right`, once with `--hand left`
- Norm stats computed on train split only, stored in HDF5 `meta` attribute
- P0021_5d8b0988 is missing hand_data.zip — automatically skipped

---

## Step 2 — Define Model

### `model.py` *(new)*

**Purpose:** Defines the two-branch MLP architecture. Imported by `train.py`, `evaluate.py`, `export_onnx.py`.

**Architecture:**
```
Input: 11 values

Branch A — Spatial Encoder:
  [rel_pos(3) + distance(1)] = 4 values
  FC(4 → 64) → ReLU → FC(64 → 32) → ReLU → spatial_emb(32)

Branch B — Object Encoder:
  [grip_onehot(4) + bbox(3)] = 7 values
  FC(7 → 64) → ReLU → FC(64 → 32) → ReLU → obj_emb(32)

Prediction Head:
  Concat [spatial_emb(32), obj_emb(32)] = 64 values
  FC(64 → 64) → ReLU → FC(64 → 22) → Tanh

Output: 22 joint angles (normalized, in [-1, 1])
```

**Class:**
```python
class AuraXRModel(nn.Module):
    def __init__(self):
        # Spatial encoder: 4 → 64 → 32
        # Object encoder:  7 → 64 → 32
        # Head: 64 → 64 → 22

    def forward(self, spatial_input, object_input):
        # spatial_input: (B, 4) — [rel_pos(3), distance(1)]
        # object_input:  (B, 7) — [grip_onehot(4), bbox(3)]
        # returns: (B, 22) — normalized joint angles
```

**Run:** Not run directly. Import with `from model import AuraXRModel`.

---

## Step 3 — Train

### `train.py` *(new)*

**Purpose:** Trains the model and saves checkpoints + metadata.

**What it does:**
1. Loads `data/right/dataset.h5` (train and val groups) — norm stats read from HDF5 `meta` attribute
2. Normalizes features and targets using stored stats
3. Splits feature vector into spatial_input (4) and object_input (7)
4. Computes per-sample loss weight based on distance:
   - distance < 10cm → `weight = 3.0` (grip frames, few but critical)
   - distance 10–40cm → `weight = 1.0` (pre-shape frames)
5. Trains with weighted MSE loss
6. Every epoch: evaluates on val split, saves checkpoint if val loss improved
7. Saves final model + training metadata

**Inputs:**
- `data/right/dataset.h5` (contains train and val groups + norm stats in meta attribute)

**Outputs:**
- `checkpoints/best_model.pt` — PyTorch state dict of best val loss model
- `checkpoints/training_log.json` — loss per epoch (train + val)
- `checkpoints/model_meta.json` — architecture config + norm stats (needed for ONNX export and Unity)

**Run:**
```bash
python train.py --data_dir data/right/ --output_dir checkpoints/right/
python train.py --data_dir data/left/  --output_dir checkpoints/left/
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | required | Directory with train.h5, val.h5, norm_stats.json |
| `--output_dir` | `checkpoints/` | Where to save model and logs |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--weight_decay` | 1e-5 | Adam weight decay |
| `--grip_weight` | 3.0 | Loss weight multiplier for <10cm frames |
| `--seed` | 42 | Random seed for reproducibility |

**Training loop:**
```python
loss = weighted_mse(pred, target, weights)
# weights[i] = grip_weight if distance[i] < 0.10 else 1.0
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Step 4 — Evaluate

### `evaluate.py` *(new)*

**Purpose:** Runs the trained model on the val split and reports quantitative metrics.

**What it does:**
1. Loads `checkpoints/best_model.pt` and `model_meta.json`
2. Loads val group from `data/right/dataset.h5`
3. Runs forward pass on all val frames (no smoothing — raw model performance)
4. Computes:
   - **Joint Angle MAE** per joint (degrees, after denormalization)
   - **Overall MAE** (mean across all 22 joints)
   - **MPJPE** (mean per-joint position error, mm) — requires UmeTrack FK to compute 3D joint positions from angles
   - **Per-object MAE** — separate breakdown for each grip category
   - **Per-phase MAE** — separate breakdown for pre-shape (10–40cm) vs grip (<10cm) frames

**Inputs:**
- `checkpoints/right/best_model.pt`
- `checkpoints/right/model_meta.json`
- `data/right/dataset.h5` (val group)

**Outputs (printed to console + saved):**
- `results/eval_right.json` — all metrics
- Console table: per-joint MAE, overall MAE, MPJPE, per-category error

**Run:**
```bash
python evaluate.py --checkpoint checkpoints/right/ --data_dir data/right/ --output_dir results/
python evaluate.py --checkpoint checkpoints/left/  --data_dir data/left/  --output_dir results/
```

**Target metrics:**
| Metric | Target |
|--------|--------|
| Joint Angle MAE | < 5° |
| MPJPE | < 20 mm |
| Pre-shape MAE | < 6° |
| Grip MAE | < 4° (tighter — these frames have higher weight) |

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | required | Directory with best_model.pt and model_meta.json |
| `--data_dir` | required | Directory with val.h5 |
| `--output_dir` | `results/` | Where to save eval JSON |

---

## Step 5 — Simulate

### `simulate.py` *(new)*

**Purpose:** Creates a synthetic approach trajectory and visualizes predicted joint angles at each distance step. Catches bad transitions before Unity testing.

**What it does:**
1. Loads trained model
2. Creates synthetic input sequence:
   - distance steps: `[0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.02]` meters
   - Fixed object: e.g. Bottle (grip=Power, size=[0.07, 0.22, 0.07])
   - Object approaches from directly ahead: rel_pos = [0, 0, distance] in wrist frame
   - Object moves from 40cm ahead → 2cm ahead along Z axis
3. Runs model at each distance step → 22 joint angle predictions
4. Plots joint angles vs distance on a line chart (one line per finger joint)
5. Checks for large jumps between steps (delta > threshold = warning)

**Inputs:**
- `checkpoints/right/best_model.pt`
- `checkpoints/right/model_meta.json`

**Outputs:**
- `results/simulation_right_bottle.png` — joint angle trajectory plot
- `results/simulation_right_cup.png` — same for cup
- Console warnings if any joint angle delta > 10° between steps

**Run:**
```bash
python simulate.py --checkpoint checkpoints/right/ --object bottle
python simulate.py --checkpoint checkpoints/right/ --object cup
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | required | Directory with best_model.pt and model_meta.json |
| `--object` | `bottle` | Which HOT3D object to simulate (`bottle`, `cup`, `pen`, etc.) |
| `--distance_steps` | 10 | Number of distance steps from 40cm to 2cm |
| `--output_dir` | `results/` | Where to save plots |

**Expected behavior:**
- At 40cm: joint angles ≈ FIXED_DEFAULT_POSE values
- At 20cm: fingers start opening/curling toward grip shape
- At 5cm: clear grip shape visible
- At 2cm: full grasp pose for the object type
- No sudden jumps between steps

---

## Step 6 — Export to ONNX

### `export_onnx.py` *(new)*

**Purpose:** Exports trained PyTorch model to ONNX format for Unity Sentis.

**What it does:**
1. Loads `best_model.pt`
2. Creates dummy input tensors (spatial_input: `(1,4)`, object_input: `(1,7)`)
3. Runs `torch.onnx.export()` with dynamic batch size
4. Verifies export by running ONNX Runtime on same dummy inputs — checks output shape is `(1,22)`
5. Copies `model_meta.json` (norm stats + architecture config) to output directory

**Inputs:**
- `checkpoints/right/best_model.pt`
- `checkpoints/right/model_meta.json`

**Outputs:**
- `onnx/auraxr_right.onnx` — ONNX model file for Unity
- `onnx/model_meta_right.json` — norm stats Unity needs to denormalize output
- Console: ONNX verification result (input/output shapes)

**Run:**
```bash
python export_onnx.py --checkpoint checkpoints/right/ --output_dir onnx/
python export_onnx.py --checkpoint checkpoints/left/  --output_dir onnx/
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | required | Directory with best_model.pt and model_meta.json |
| `--output_dir` | `onnx/` | Where to save .onnx and meta JSON |
| `--opset` | 14 | ONNX opset version (Unity Sentis supports 14+) |

**ONNX input/output spec (for Unity):**
```
Input  "spatial_input": shape [1, 4]  — [rel_pos(3), distance(1)]
Input  "object_input":  shape [1, 7]  — [grip_onehot(4), bbox(3)]
Output "joint_angles":  shape [1, 22] — normalized, apply denorm before use
```

**Denormalization in Unity:**
```csharp
float angle_i = (model_output[i] * target_std[i]) + target_mean[i];
```

---

## Execution Order Summary

```
1. python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/right/ --hand right
2. python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/left/  --hand left
3. python train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/
4. python train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/
5. python evaluate.py --checkpoint ../checkpoints/right/ --data_dir ../data/right/ --output_dir ../results/
6. python evaluate.py --checkpoint ../checkpoints/left/  --data_dir ../data/left/  --output_dir ../results/
7. python simulate.py --checkpoint ../checkpoints/right/ --object bottle --output_dir ../results/
8. python simulate.py --checkpoint ../checkpoints/right/ --object cup    --output_dir ../results/
9. python export_onnx.py --checkpoint ../checkpoints/right/ --output_dir ../onnx/
10. python export_onnx.py --checkpoint ../checkpoints/left/  --output_dir ../onnx/
```

After step 10: copy `onnx/auraxr_right.onnx`, `onnx/auraxr_left.onnx`, and both `model_meta_*.json` files to Unity Assets.

---

## File and Directory Structure

```
hot3d_exploration/
├── grip_categories.py       (new — utility, grip map)
├── hot3d_utils.py           (existing — update)
├── hot3d_dataset.py         (existing — update)
├── build_dataset.py         (new — step 1)
├── model.py                 (new — step 2)
├── train.py                 (new — step 3)
├── evaluate.py              (new — step 4)
├── simulate.py              (new — step 5)
└── export_onnx.py           (new — step 6)

data/
├── right/
│   ├── train.h5
│   ├── val.h5
│   └── norm_stats.json
└── left/
    ├── train.h5
    ├── val.h5
    └── norm_stats.json

checkpoints/
├── right/
│   ├── best_model.pt
│   ├── training_log.json
│   └── model_meta.json
└── left/
    └── (same)

onnx/
├── auraxr_right.onnx
├── auraxr_left.onnx
├── model_meta_right.json
└── model_meta_left.json

results/
├── eval_right.json
├── eval_left.json
├── simulation_right_bottle.png
└── simulation_right_cup.png
```

---

## Key Shared Data Format

### Feature Vector (11 values)

| Index | Values | Source |
|-------|--------|--------|
| 0–2   | Object relative position (x, y, z in wrist frame) | Computed: `R_wrist^T × (obj_world - wrist_world)` |
| 3–6   | Grip category one-hot | BOP ID → 4-class via `OBJ_BBOX` table |
| 7–9   | Object bbox half-extents (x, y, z meters) | `OBJ_BBOX[bop_id]` lookup |
| 10    | Hand-object distance (meters) | `‖rel_pos‖` |

### Target Vector (22 values)

UmeTrack joint angles — 22 finger joints, left or right hand. Wrist is **not** in this vector (wrist is anchored to controller in Unity, not predicted).

### model_meta.json Structure

```json
{
  "feature_mean": [11 floats],
  "feature_std":  [11 floats],
  "target_mean":  [22 floats],
  "target_std":   [22 floats],
  "architecture": {
    "spatial_input_dim": 4,
    "object_input_dim": 7,
    "output_dim": 22,
    "hidden_dim": 64,
    "embedding_dim": 32
  }
}
```

---

## Dependencies

```
pip install torch torchvision
pip install onnx onnxruntime
pip install h5py
pip install numpy matplotlib
```

HOT3D data is read directly from downloaded ZIP files — no additional SDK required.

See `requirements.txt` for pinned versions.
