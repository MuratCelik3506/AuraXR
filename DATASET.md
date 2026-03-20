# Dataset Specification — Intent-Aware XR Framework

This document provides a comprehensive deep-dive into the two datasets used in this framework: **H2O** and **HOT3D**, including their structures, skeletal representations, and how they are unified for intent prediction.

---

## 📅 1. Dataset Overview

To build a robust intent prediction system, we combine two different data sources:
- **H2O:** Provides legacy "interaction logic" and detailed action labels.
- **HOT3D:** Provides modern "egocentric motion" from high-end XR hardware (Meta Quest 3).

| Feature | H2O | HOT3D |
| :--- | :--- | :--- |
| **Source** | KAIST (Static RGB-D Rigs) | Meta (Aria / Quest 3) |
| **Perspective** | Third-person & Ego | Ego-only (First-person) |
| **Labels** | 36 fine-grained actions | 3 coarse temporal phases |
| **Hand Representation** | 21 Cartesian Joints (XYZ) | 22 UmeTrack Angles + Wrist Trans |
| **Object Representation** | 4x4 Ground Truth RT | 6D Pose annotations |

---

## 🧩 2. H2O (Human-to-Object) Dataset

### Structure on Disk
Located at `data/h2o/`.
```bash
annotations/
  Subject{1..4}/
    h1/k1/o1... (Rig IDs)/
      {Clip_ID}/
        cam4/
          hand_pose/   # 21 joints * 3 coords * 2 hands (.txt)
          obj_pose_rt/ # 4x4 Object transform (.txt)
          action_label/# Integer 1-36 (.txt)
```

### Skeletal Format
- **Joints:** 21 joints per hand (MANO convention).
- **Data per joint:** `[visibility, x, y, z]`. We ignore visibility and extract `x, y, z`.
- **Normalization:** We subtract the **Wrist (Joint 0)** from all other joints to achieve translation-invariant motion features.

### Action Mapping (0-indexed in code)
- **0–11:** Pickup/Approach phase.
- **12–23:** Manipulation phase (Pour, Stir, etc.).
- **24–35:** Release/Put-down phase.

---

## 💎 3. HOT3D (Hands-Object Tracking in 3D)

### Data Format
HOT3D is distributed as **WebDataset Tar archives**. Each `.tar` file contains one clip (approx. 150 frames). Our implementation reads these archives directly to save disk space.

### Tar Contents per Frame:
- `XXXXXX.hands.json`: Contains `umetrack_pose`.
- `XXXXXX.objects.json`: Contains 6D object transforms.
- `XXXXXX.info.json`: Frame metadata.

### UmeTrack to 3D Cartesian Reconstruction
HOT3D does not provide raw XYZ joint positions in a standard format. We implement a **Forward Kinematics (FK)** approximation:
1.  **Input:** 22 UmeTrack angles + 4×4 `T_world_from_wrist` matrix.
2.  **Logic:** Start from the wrist position. Apply bone length offsets (canonical MANO lengths) modified by the joint angles in the wrist's local coordinate system.
3.  **Result:** 21 XYZ joints comparable to the H2O format.

### Temporal Labels (Heuristic)
Since HOT3D lacks frame-level labels, we segment each 5-second clip into:
- **0–33%:** `pick-up` (Class 0)
- **33–66%:** `observe` (Class 1)
- **66–100%:** `put-down` (Class 2)

---

## ⚙️ 4. Feature Engineering & Windowing

### The Intent Window
For real-time intent prediction, the model does not look at the whole action. It looks at a **sliding window** (default 30 frames / 1 second) sampled at a specific **Observation Ratio**.

- **Example:** If an action takes 100 frames and `obs_ratio = 0.25`, the model sees frames 0 to 25.
- **Goal:** Train the model to map the features in frames [0-25] to the final label of the whole sequence.

### Feature Vector (142 dimensions)
Each frame is represented by a 142-dimensional vector:
1.  **Hand Joints (126 dims):** 2 hands × 21 joints × 3 coords (XYZ), wrist-relative.
2.  **Object Pose (16 dims):** Flattened 4×4 transformation matrix representing the target object's position/rotation relative to the camera.

---

## 🔗 5. Data Fusion: The "Shared Head" Strategy

To train the **IntentFormer** on both datasets simultaneously, we use a mapping that projects H2O's 36 classes into HOT3D's 3-class space:

| Coarse Label | Coarse Name | H2O Classes (Indices) | HOT3D Phase |
| :--- | :--- | :--- | :--- |
| **0** | **Pick-up** | 0 - 11 (Grab, Lift) | First 33% |
| **1** | **Observe** | 12 - 23 (Hold, Use) | Middle 33% |
| **2** | **Put-down** | 24 - 35 (Place, Drop) | Final 33% |

By training on this **Shared Head**, the model learns general motion primitives (e.g., "fingers closing near an object") that apply to both synthetic rigs (H2O) and real-world Quest 3 sessions (HOT3D).

---

## 🛠️ 6. How to Extend

If you add a new dataset (e.g., **Epic-Kitchens** or **DexYCB**):
1.  Implement a wrapper in `src/data/` that outputs the standard dictionary: `{"hand_flat": (T,126), "obj_rt": (T,16), "label": int, "obs_ratio": float}`.
2.  Add a mapping function in `combined_dataset.py` to target the `shared_head` (3 classes).
3.  The `IntentFormer` will then be able to train on your new data without architecture changes.
