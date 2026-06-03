# HOT3D Dataset — Complete Explanation

## 1. What is HOT3D?

**HOT3D** stands for **Hand and Object Tracking in 3D**. It is a large-scale dataset released by **Meta Reality Labs** designed to benchmark and train models that understand how human hands interact with real-world objects in 3D space.

The core goal: given a stream of egocentric (first-person) video, can a model track *where the hands are in 3D* and *where the objects are in 3D* at every frame?

This is fundamentally harder than 2D pose estimation — the model must reason about depth, occlusion, and physical contact.

---

## 2. Why HOT3D Matters

| Problem | Why it's Hard |
|---|---|
| Hands are highly articulated | 21 joints per hand, self-occlusion is constant |
| Objects vary in shape and texture | From a cup to a book to a stapler |
| Egocentric view is unusual | Camera moves with the head, not fixed |
| 3D labels are expensive to collect | Can't just draw 2D bounding boxes |

Previous datasets either had 2D-only labels, very few objects, lab-only settings, or no hand-object contact. HOT3D fills this gap with real-world egocentric capture at scale.

---

## 3. Capture Devices & Setup

HOT3D uses **two different egocentric capture platforms**:

### Meta Quest 3 (Headset)
- A VR/AR headset worn on the head
- Has **stereo RGB cameras** (outward-facing)
- Has **depth sensors**
- Has **IMU** (inertial measurement unit — gyroscope + accelerometer)
- Resolution: ~1280×1024 per eye camera

### Project Aria (Glasses)
- Lightweight research glasses from Meta
- Has **RGB camera**, **eye-tracking cameras**, **slam cameras**
- Has **IMU**
- Looks like regular glasses — more natural for subjects
- No depth sensor (unlike Quest 3)

**Why two devices?** To test generalization. A model trained only on Quest 3 data might overfit to that camera's optics. Having both makes the benchmark more realistic.

Each recording session = one **sequence**. A subject wears either Quest 3 or Aria and interacts with objects on a table.

---

## 4. Dataset Scale & Statistics

| Property | Value |
|---|---|
| Total sequences | ~800+ |
| Total frames | ~1.5 million |
| Unique subjects | 19 people |
| Unique objects | 33 household objects |
| Devices | Meta Quest 3 + Project Aria |
| Annotation type | 3D hand pose + 6DoF object pose |
| Hand model formats | MANO and UmeTrack |

The 33 objects are everyday items: cups, bottles, scissors, staplers, phones, etc. Each has a precise **3D mesh** captured in a controlled scan.

---

## 5. Data Modalities

For each sequence you get multiple synchronized data streams:

### Images / Video
- **RGB frames** from the ego camera (what the person sees)
- Stored in `.vrs` format (Meta's video recording system)
- Multiple camera streams per device (left/right eye for Quest 3, multiple cameras for Aria)

### Depth (Quest 3 only)
- Per-pixel depth maps synchronized with RGB
- Enables direct 3D reconstruction without stereo estimation

### IMU
- High-frequency motion data (accelerometer + gyroscope)
- Used to understand head motion between frames

### Hand Pose Annotations
- 3D joint positions for both hands at each frame
- Provided in two formats: **MANO** and **UmeTrack** (see Section 7)

### Object Pose Annotations
- **6DoF pose** = 3D position (x, y, z) + 3D orientation (rotation matrix or quaternion)
- One pose per object per frame when the object is visible/held

### Camera Calibration
- Intrinsic parameters (focal length, principal point, distortion)
- Extrinsic parameters (position and rotation of each camera relative to device)
- Required to project 3D points into 2D image space

---

## 6. Folder & File Structure

```
hot3d/
│
├── sequences/
│   ├── <sequence_uid>/               # e.g., "P0001_Q3_0001"
│   │   ├── video.vrs                 # raw multi-stream video (RGB, depth, IMU)
│   │   ├── hand_poses.json           # per-frame 3D hand annotations
│   │   ├── object_poses.json         # per-frame 6DoF object poses
│   │   ├── camera_calibration.json   # intrinsics + extrinsics
│   │   └── metadata.json             # subject ID, device type, object list, etc.
│   │
│   └── <sequence_uid>/               # another sequence (could be Aria device)
│       └── ...
│
├── objects/
│   ├── <object_id>/                  # e.g., "cup_01"
│   │   ├── mesh.obj                  # 3D triangle mesh of the object
│   │   ├── mesh.mtl                  # material/texture reference
│   │   └── texture.png              # texture map for the mesh
│   └── ...
│
├── splits/
│   ├── train.txt                     # list of sequence UIDs for training
│   ├── val.txt                       # validation sequences
│   └── test.txt                      # test sequences (no labels released)
│
└── object_library.json               # metadata for all 33 objects (name, category, etc.)
```

### Sequence UID Convention

A sequence ID like `P0003_Q3_0017` encodes:
- `P0003` → Subject (participant) #3
- `Q3` → Device is Quest 3 (vs `AR` for Aria)
- `0017` → Sequence number for that subject+device

---

## 7. Annotation Formats

### 7a. Hand Pose — MANO Format

**MANO** is an industry-standard parametric hand model from the paper *"MANO: Hand Model with Articulated and Non-rigid Deformations"*.

Instead of storing raw 3D joint coordinates, MANO stores **parameters** that describe the hand:

```
MANO parameters per hand per frame:
  - pose: 48 values  (15 joint rotations as axis-angle + global rotation)
  - shape: 10 values (person-specific hand shape, fixed per subject)
  - translation: 3 values (global 3D position in camera space)
```

From these ~61 numbers, you can reconstruct a full hand mesh (778 vertices) and 21 joint locations using the MANO model code. This is compact and differentiable — great for neural network outputs.

**Why not just store joint XYZ?** MANO ensures anatomically plausible poses (no impossible finger bends) and gives you the full hand surface, not just skeleton joints.

### 7b. Hand Pose — UmeTrack Format

**UmeTrack** is Meta's own hand tracking format, developed alongside the Xtended Hand dataset. It stores:

```
UmeTrack parameters per hand per frame:
  - joint_positions_3d: 21 × 3 = 63 values (raw XYZ for each joint)
  - joint_angles: compact angle representation
  - wrist_transform: 4×4 matrix (position + orientation of wrist)
```

This is more direct than MANO — you work with joint positions without needing the MANO model code. HOT3D provides **both formats** so researchers can use whichever fits their pipeline.

### 7c. Object Pose — 6DoF Format

Each object has a known 3D mesh. The annotation stores:

```json
{
  "frame_id": 1042,
  "object_id": "cup_01",
  "translation": [0.123, -0.045, 0.812],   // x, y, z in meters (camera space)
  "rotation": [[r00, r01, r02],              // 3×3 rotation matrix
               [r10, r11, r12],
               [r20, r21, r22]],
  "visibility": 0.87                         // fraction of object visible (0–1)
}
```

With this pose, you can place the 3D mesh exactly where it was in the scene. You can also project the mesh into the image to overlay it on the RGB frame.

---

## 8. Data Splits

| Split | Purpose | Labels Available? |
|---|---|---|
| `train` | Train your model | Yes — full annotations |
| `val` | Tune hyperparameters, check metrics | Yes — full annotations |
| `test` | Final benchmark evaluation | No — submitted to server |

The test set labels are held by Meta's evaluation server. You submit predictions and receive scores. This prevents overfitting to test labels.

Split files are simple text files, one sequence UID per line:
```
P0001_Q3_0001
P0001_Q3_0002
P0002_AR_0001
...
```

---

## 9. Key Concepts to Understand

### Camera Space vs World Space

All poses in HOT3D are expressed in **camera space** (relative to the camera origin, not a fixed world point). This is important:
- Position `[0, 0, 0.5]` means "0.5 meters directly in front of the camera"
- As the camera moves (person turns head), the same object will have different coordinates next frame

### VRS Format

`.vrs` is Meta's internal container for multi-stream recordings. Like an `.mkv` or `.bag` file — it holds multiple time-synchronized data channels. The **PyVRS** library is needed to extract frames and sensor data. The **PyHOT3D** library wraps this and gives you clean Python objects.

### Synchronization

All data streams are timestamped. When you load a frame, you get:
- The RGB image at timestamp T
- The depth map at (approximately) the same T
- The hand pose annotation at T
- The object pose annotation at T

Small timing offsets exist — the PyHOT3D loader handles interpolation.

### Coordinate System

HOT3D uses a **right-handed coordinate system**:
- X → right
- Y → down
- Z → forward (into the scene)

This is the standard camera coordinate convention.

---

## 10. How Data Flows Together — End to End

```
                    ┌─────────────────────────┐
                    │     sequence_uid dir     │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
         video.vrs        hand_poses.json    object_poses.json
              │                  │                   │
    ┌─────────▼────────┐   ┌─────▼──────┐   ┌───────▼────────┐
    │ Extract RGB frame│   │ Load MANO  │   │ Load 6DoF pose │
    │ at timestamp T   │   │ params → 21│   │ + object mesh  │
    │                  │   │ 3D joints  │   │ from objects/  │
    └─────────┬────────┘   └─────┬──────┘   └───────┬────────┘
              │                  │                   │
              └──────────────────▼───────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  camera_calibration.json │
                    │  Project all 3D points   │
                    │  into 2D image coords    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Final frame with:       │
                    │  - RGB image             │
                    │  - 2D hand landmarks     │
                    │  - 3D hand joints        │
                    │  - Object overlay        │
                    └─────────────────────────┘
```

This pipeline is what tools like `build_dataset.py` in this repo implement — reading raw HOT3D files and converting them into tensors for training.

---

## Summary

| Concept | One-Line Answer |
|---|---|
| What is HOT3D? | Egocentric 3D hand+object tracking dataset from Meta |
| How was it captured? | Quest 3 headset + Project Aria glasses |
| What's in a sequence? | RGB video, depth, IMU, 3D hand poses, 6DoF object poses |
| How are hands represented? | MANO parameters or UmeTrack joint positions |
| How are objects represented? | 6DoF pose (translation + rotation) + 3D mesh |
| File format for video? | `.vrs` (Meta's multi-stream container) |
| How many objects? | 33 unique household objects with 3D meshes |
| How many subjects? | 19 people across train/val/test |
