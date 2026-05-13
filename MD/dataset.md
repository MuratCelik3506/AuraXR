# HOT3D Dataset — Detailed Technical Reference
## For AuraXR Thesis: Hand Pose Estimation from VR Controller Data

> **Source:** https://facebookresearch.github.io/hot3d/
> **Paper:** "HOT3D: Hand and Object Tracking in 3D" — CVPR 2025 Highlight (top 13.5%)
> **ArXiv:** 2411.19167
> **Last Updated:** 2026-04-30
>
> ⚠️ **CORRECTION (verified 2026-05-07):** This document states MANO pose θ = "45 values per hand (15 joint × 3 axis-angle)" in Section 3.2. **This is wrong.** HOT3D actual ground truth provides **15 floats per hand** (1 scalar DoF per joint, not 3D axis-angle). The implemented model output is 78 dims total (39 per hand), not 138 as originally planned. Trust `plan.md` Section 2 for authoritative dimensions.

---

## 1. What HOT3D Is

HOT3D is an **egocentric dataset** for benchmarking 3D hand and object tracking from multi-view synchronized recordings of hand-object interactions. It was captured using two head-mounted devices — Project Aria research glasses and the Meta Quest 3 VR headset — providing a unique dual-device perspective on the same interactions.

Crucially for AuraXR: HOT3D includes a **Quest 3 capture track**, meaning ground-truth MANO hand poses exist for sequences recorded with the same hardware used at inference time. This is the primary reason HOT3D is the right dataset for this thesis.

---

## 2. Dataset Scale

| Metric | Value |
|--------|-------|
| Total recording time | 833+ minutes (over 13.8 hours) |
| Total frames (multi-view) | 1.5M |
| Total images | 3.7M+ |
| Frame rate | **30 FPS** |
| Subjects | 19 participants (diverse hand shapes, nationalities) |
| Objects | 33 rigid household and office objects |
| Curated clips total | 3,832 clips (each 150 frames = 5 seconds) |
| Aria clips | 1,983 |
| Quest 3 clips | 1,849 |
| BOP training clips | 2,969 |
| BOP test clips | 1,148 |
| BOP real training images | 420,600 |
| BOP synthetic (PBR) training images | 50,000 |
| BOP test images | 154,200 |

### Participant Split (Train / Test)

| Split | Participant IDs |
|-------|----------------|
| TEST | P0004, P0005, P0006, P0008, P0016, P0020 |
| TRAIN | All remaining 13 participants |

> **Important for AuraXR:** The train/test split is **participant-level** (different people, not different sessions). This is a strict generalisation test — the model must work on hand shapes it has never seen. Data augmentation on hand shape (β perturbation, left↔right mirroring) is therefore essential.

### Interaction Scenarios

- Kitchen interactions
- Office interactions
- Living room interactions
- Simple pick-up / observe / put-down actions

---

## 3. Data Modalities — Complete Breakdown

### 3.1 Visual Streams

| Device | Stream | Resolution | Format |
|--------|--------|------------|--------|
| Project Aria | RGB | 1408 × 1408 | `.jpg` |
| Project Aria | Monochrome Left | 640 × 480 | `.jpg` |
| Project Aria | Monochrome Right | 640 × 480 | `.jpg` |
| Quest 3 | Monochrome Left | 1280 × 1024 | `.jpg` |
| Quest 3 | Monochrome Right | 1280 × 1024 | `.jpg` |

> **CRITICAL for AuraXR (see Q-B in questions.md):** Quest 3 sequences in HOT3D have **no RGB camera** — only monochrome. The thesis plan assumes RGB passthrough as visual context (Decision D4), but training data from Quest 3 is greyscale. If you use Aria RGB for training the visual branch, you face a domain gap at Quest 3 inference time. This is an unresolved conflict.

### 3.2 Hand Annotations (Ground Truth)

Two parallel annotation formats are provided:

| Format | Description | Availability |
|--------|-------------|-------------|
| **UmeTrack** | 21-joint 3D hand keypoints in world space | Free, no extra license |
| **MANO** | Shape β (10 coeffs) + Pose θ (15 values per hand, 1 DoF per joint) + wrist 6DoF | Requires SMPLX/MANO license acceptance |

For AuraXR, **MANO format is required** as it directly matches the output representation. MANO annotations include:
- `θ` (pose): 15 joints × 1 DoF = **15 floats per hand** (HOT3D uses 1 curl scalar per joint, not 3D axis-angle)
- `β` (shape): 10 shape coefficients per hand
- Wrist transform: quaternion (w,x,y,z) + translation (x,y,z) in world space

Both hands are annotated simultaneously when both are visible.

### 3.3 Object Data

| Modality | Description |
|----------|-------------|
| 6DoF object poses | Position + orientation of each rigid object per frame in world space |
| 3D object models | High-fidelity meshes with PBR (Physically Based Rendering) materials for all 33 objects |
| 2D bounding boxes | Per-object, per-view bounding boxes |
| Object metadata | Category labels, IDs, dimensions |

> **For AuraXR:** The object 6DoF poses are used to derive affordance features (centroid, bounding box, closest surface point, surface normal). The 3D object models enable SDF queries for the Penetration Loss during training.

### 3.4 Scene & Sensor Data

| Modality | Description | AuraXR Usage |
|----------|-------------|--------------|
| Camera intrinsics | Per-view, per-device calibration | Required for projecting hand joints into image space |
| Camera extrinsics | World-to-camera transforms (quaternion + translation) | Required for coordinate frame alignment |
| SLAM point cloud | Room-scale 3D map from Aria glasses | Future work only (excluded from V1) |
| Eye gaze | Gaze direction signal | **Aria only** — excluded from AuraXR (Quest 3 has no eye tracker) |
| IMU | Inertial measurements from Aria | Not used in AuraXR |

### 3.5 Metadata Files (per clip)

Each clip in the WebDataset format contains:

```
clip/
├── cameras.json        # Intrinsics, extrinsics for all views
├── hand_crops.json     # Per-frame hand detection bounding boxes
├── info.json           # participant_id, sequence_id, device, timestamps
├── image_1201-1.jpg    # Images named: image_{stream_id}-{view_id}
├── image_1201-2.jpg
├── image_214-1.jpg
└── [pose .json files]  # Per-frame T_world_from_camera transforms
```

### 3.6 Data Quality Flags (8 CSV mask files per sequence)

Flag frame validity on:
- Object pose availability
- Hand pose availability
- Object visibility
- Hand visibility
- Exposure quality (over/under exposed frames)
- Manual QA pass/fail

> **For AuraXR:** These flags must be applied during preprocessing. Frames with failed QA, poor exposure, or missing hand annotations must be excluded from training.

---

## 4. Capture Hardware

### 4.1 Project Aria (Meta Research Prototype AR Glasses)
- RGB camera: 1408 × 1408 fisheye
- Two monochrome cameras: 640 × 480 each
- Eye gaze sensors
- IMU sensors
- All cameras fisheye — require undistortion before use

### 4.2 Meta Quest 3 (Consumer VR Headset)
- Two monochrome passthrough cameras: 1280 × 1024 each
- **No RGB camera in HOT3D recording**
- Same headset model as the AuraXR target deployment device

### 4.3 Motion Capture Ground Truth System
- Optical markers placed on hands and objects
- Sub-millimetre accuracy
- Used to generate ground-truth poses that are then fitted to MANO model

---

## 5. Access & Download

### 5.1 HOT3D-Clips (Recommended Starting Point)
- **URL:** https://huggingface.co/datasets/bop-benchmark/hot3d
- **Format:** WebDataset `.tar` files (streamable)
- **Size:** 762 GB total
- **Access:** Direct download (Hugging Face login may be required)
- **MANO annotations:** Require separate license acceptance

```python
# Option 1: HuggingFace Datasets API
from datasets import load_dataset
dataset = load_dataset("bop-benchmark/hot3d")
train_data = dataset["train"]
sample = train_data[0]

# Option 2: WebDataset streaming
import webdataset as wds
dataset = wds.WebDataset("hf://datasets/bop-benchmark/hot3d/train_aria/clip-{id}.tar")
```

### 5.2 Full HOT3D Dataset (VRS Format)
- **URL:** https://www.projectaria.com/datasets/hot3d/
- **Format:** VRS (Video Recording System) — requires `projectaria_tools`
- **Access:** Email submission + license agreement required
- **Reader:** Official `Hot3dDataProvider` Python API

### 5.3 License Requirements

| Resource | License |
|----------|---------|
| HOT3D dataset | HOT3D custom agreement (via projectaria.com) |
| MANO hand annotations | SMPLX/MANO research license (separate acceptance required) |
| HOT3D code/tools | Apache 2.0 (open source) |
| 3D object models | HOT3D agreement |

---

## 6. Python API & Tools

### 6.1 Official Repository
- **GitHub:** https://github.com/facebookresearch/hot3d
- **License:** Apache 2.0 (code only)

### 6.2 Core Python Dependencies

```
projectaria_tools==1.5.1   # VRS file reading, camera models, undistortion
torch                       # For MANO model and feature extraction
datasets                    # HuggingFace dataset loading
webdataset                  # Streaming WebDataset format
rerun-sdk==0.16.0           # Official visualiser
matplotlib                  # Plotting
requests                    # Download utilities
smplx                       # MANO forward kinematics
numpy
opencv-python               # Image processing
trimesh                     # 3D mesh operations (SDF queries for penetration loss)
```

### 6.3 Key Python Classes

```python
# Full dataset (VRS format)
from hot3d.data_loaders import Hot3dDataProvider
provider = Hot3dDataProvider(sequence_path)
metadata = provider.get_sequence_metadata()

# Load frame data
frame = provider.get_frame(timestamp_ns)
hand_pose_left  = frame.hand_poses.left   # UmeTrack format
hand_pose_right = frame.hand_poses.right

# Load MANO params (requires MANO license)
mano_left  = frame.mano_poses.left   # θ, β, wrist transform
mano_right = frame.mano_poses.right
```

---

## 7. What HOT3D Does NOT Provide (for AuraXR)

| Missing Data | Impact on AuraXR | Resolution |
|-------------|-----------------|------------|
| **Controller 6DoF poses** | Training input cannot be real — must be derived from MANO wrist | See Q-A in questions.md |
| **Quest 3 RGB frames** | RGB visual branch cannot be trained on Quest 3 sequences | See Q-B in questions.md |
| **Controller button/grip states** | Grip and trigger values must be simulated or excluded | Use binary grip proxy from hand closure |
| **Eye gaze on Quest 3** | Excluded by design — Quest 3 has no eye tracker | Already excluded (Decision D3) |

---

## 8. Objects in the Dataset (33 Total)

HOT3D's 33 objects span household and office categories relevant to the AuraXR use case:

- **Cylindrical grasps:** bottles, mugs, cups, thermos, spray bottles
- **Flat/planar grasps:** books, notebooks, folders, trays, cutting boards
- **Pinch grasps:** pens, markers, scissors, remotes, phones
- **Power grasps:** bowls, containers, boxes, keyboards
- (Full object list in the official repository's `assets/` directory)

> **For POC Subset Selection (Q1 in plan):** Prioritise objects with clear single-hand grasp patterns and strong category-specific finger pose variation. Recommended POC categories: **bottle** (cylindrical power grasp), **mug** (handle grasp), **book** (flat palm pinch), **remote** (precision grasp), **cup** (cylindrical pinch).

---

## 9. Temporal Properties & Frame Rate Implications

### Critical Mismatch: Training vs. Inference Rate

| Context | Frame Rate | T=16 window = |
|---------|------------|--------------|
| HOT3D dataset | **30 FPS** | 533 ms |
| Quest 3 native | **72 Hz** | 222 ms |
| Quest 3 if capped | **30 FPS** | 533 ms |

If the model is trained on T=16 windows at 30 FPS and deployed at 72 Hz, the temporal context seen at inference (222ms) is less than half of what the model was trained on (533ms). This directly affects the quality of anticipatory grasp pose prediction.

**Options (see Q-C in questions.md):**
1. Cap inference at 30 FPS — simpler, matches training, but wastes Quest 3 display budget
2. Train at 30 FPS, deploy at 72 Hz with a longer T (T=38 for 533ms parity) — requires re-sampling
3. Treat temporal position encoding as time-relative (ms-based), enabling cross-frame-rate inference

---

## 10. Benchmark Results on HOT3D (State of the Art)

From BOP Challenge 2024 (the most recent benchmark on HOT3D):

| Method | Task | AP Score |
|--------|------|----------|
| GigaPose + GenFlow | Model-based 6DoF object pose | 26.8 AP |
| GFreeDet | Model-free unseen object detection | — |

> **Note:** HOT3D scores are noticeably lower than BOP-Classic-Core datasets, indicating it is a challenging benchmark. Multi-view methods significantly outperform single-view methods on HOT3D — relevant if you explore multi-camera features from the Quest 3's stereo cameras.

---

## 11. Related Work Built on HOT3D

| Paper/Challenge | Venue | Relevance |
|----------------|-------|-----------|
| HOT3D original paper | CVPR 2025 Highlight | Primary citation |
| BOP Challenge 2024 | ECCV 2024 Workshop | Object pose estimation results |
| Multiview Egocentric Hand Tracking Challenge | ECCV 2024 | Hand tracking results |

**Authors of HOT3D:** Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, Tomas Hodan

---

## 12. Preprocessing Pipeline for AuraXR

Based on the HOT3D structure, the preprocessing pipeline for AuraXR training data must:

### Step 1 — Select and Filter Sequences
```
For each sequence:
  - Apply all 8 quality mask CSV filters (exclude failed QA, bad exposure frames)
  - Keep only frames where BOTH hands are visible (bimanual requirement)
  - Keep only frames where at least one object is within 30cm of either hand
```

### Step 2 — Derive Controller Proxy Poses (unresolved — see Q-A)
```
For each frame:
  - Extract MANO wrist transform (position + quaternion) for each hand
  - Apply a fixed offset to simulate controller tracking origin
  - This becomes the "controller pose" training input
  - Grip proxy: compute hand closure ratio from MANO finger flexion angles
  - Trigger proxy: binary flag based on index finger flexion > threshold
```

### Step 3 — Extract Object Affordance Features
```
For each frame:
  - Find the nearest object to each hand's wrist position
  - Compute: centroid, bounding box half-extents, object category ID
  - Compute: closest surface point and surface normal from 3D mesh
```

### Step 4 — Extract Visual Context
```
For Aria sequences:
  - Load RGB frame (1408×1408)
  - Crop region centred on the interaction zone
  - Resize to CNN input size (e.g., 224×224)
  - Apply standard ImageNet normalisation

For Quest 3 sequences:
  - Load monochrome frame (1280×1024) — greyscale only
  - Same crop/resize pipeline
  - Normalise to [0, 1]
```

### Step 5 — Assemble Temporal Windows
```
Slide a window of T=16 frames (stride=1) over each sequence:
  - Input: [16 × 96] feature tensor (9 left ctrl + 9 right ctrl + 7 obj_L + 7 obj_R + 64 visual zeros)
  - Target: [78] MANO output vector for the final frame in the window
  - Apply quality filter — skip windows with any flagged frame
```

### Step 6 — Split Dataset
```
Split is participant-level (NOT random frame split):
  - Train: 13 participants (all except P0004/5/6/8/16/20)
  - Val: held-out sessions from train participants (used for ALL quantitative metrics)
  - Test: P0004/5/6/8/16/20 — HOT3D test GT is withheld (BOP eval server only)

Note: test GT is NOT available locally. Val split is the evaluation set for this thesis.
```

---

## 13. File Size Estimates for AuraXR Preprocessing

| Asset | Estimated Size |
|-------|---------------|
| HOT3D-Clips (all, raw) | 762 GB |
| POC subset (5 categories, 15% clips) | ~30–50 GB |
| Preprocessed features (numpy, POC) | ~2–5 GB |
| Preprocessed features (full, no images) | ~20–30 GB |
| CNN-extracted visual embeddings (cached) | ~5–10 GB |

> **Recommendation:** Cache CNN embeddings offline during preprocessing. Running ResNet-18 on 1.5M frames online during training will be the bottleneck.

---

## 14. Resources & Links

| Resource | URL |
|----------|-----|
| Project website | https://facebookresearch.github.io/hot3d/ |
| Project Aria dataset page | https://www.projectaria.com/datasets/hot3d/ |
| HuggingFace (clips) | https://huggingface.co/datasets/bop-benchmark/hot3d |
| GitHub (code & tools) | https://github.com/facebookresearch/hot3d |
| ArXiv paper | https://arxiv.org/abs/2411.19167 |
| MANO model | https://mano.is.tue.mpg.de/ |
| BOP Challenge 2024 | https://bop.felk.cvut.cz/challenges/bop-challenge-2024/ |
| BOP paper | https://arxiv.org/html/2504.02812v2 |
| projectaria_tools docs | https://facebookresearch.github.io/projectaria_tools/ |

---

*This document is a reference, not a specification. When HOT3D's actual data structure conflicts with assumptions in thesis_plan.md, this document takes precedence as the ground truth. Update thesis_plan.md accordingly after resolving the open questions.*
