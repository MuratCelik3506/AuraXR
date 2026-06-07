# VR Controller-Based Object-Aware Hand Pose Prediction

## Project Summary

This project aims to develop an AI model that predicts the hand shape of a VR controller user based on the type and shape of the object being approached.

---

## Problem Definition

### Current State
- The user's real hand is invisible while using a VR controller
- The controller only provides position and rotation data
- In virtual environments, the hand is usually displayed as a fixed pose
- This reduces immersion

### Goal
- Hand shape prediction from controller data
- Natural hand movement when approaching an object
- Appropriate grasp position based on object type
- Realistic and fluid hand animation

---

## Approach: Proximity-Based Anticipatory Grasp

A human hand begins adapting its shape *before* contact — when reaching for a mug, fingers start curling at ~40 cm, well before touching the object. This "pre-shaping" is driven by the perceived object type and approach angle.

AuraXR models this by training a small MLP on HOT3D data where real hand poses during manipulation are labeled by distance. The model learns to predict the correct hand pose for any (object type, object size, approach direction, distance) tuple, enabling smooth anticipatory animation in VR without any camera or glove hardware.

## Dataset: HOT3D

### Why HOT3D?

| Feature | HOT3D Advantage |
|---------|----------------|
| Hand format | UmeTrack (VR-compatible) |
| Object data | 33 objects, 6DoF pose |
| Coordinate system | Hand and object in same system |
| Recording device | Quest 3 (same as target platform) |
| Data volume | 3832 clips, 1.5M+ frames |

### Available Data

**Hand Data (Per Frame):**
- UmeTrack format: 22 joint angles
- Wrist position: Quaternion + Translation (7 values)
- Left and right hand separately annotated

**Object Data (Per Frame):**
- Object type and ID
- 6DoF pose (position + rotation)
- Segmentation mask
- Visibility information

**Object Models:**
- 33 unique rigid objects
- 3D mesh (.glb format)
- Dimension data (X, Y, Z, diameter)
- PBR materials

---

## Model Architecture

### Input

| Parameter | Size | Description |
|-----------|------|-------------|
| dir_world | 3 | Unit vector wrist→object in HOT3D world frame |
| dir_obj_local | 3 | Same vector rotated into the object's local frame (encodes which face is approached) |
| distance | 1 | Euclidean distance wrist→object in meters |
| approach_speed | 1 | dot(wrist_velocity, dir_world) — positive = moving toward object |
| grip_onehot | 4 | Power / Precision / Palmar / Pinch — one-hot |
| bbox half-extents | 3 | Object bounding box half-extents [x, y, z] in meters |
| **Total** | **15** | split as spatial(8) + object(7) for the model |

> **Why not wrist-local direction?** Computing direction in the wrist frame would always yield `(0,0,1)` in a canonical frame — trivially useless. Using the real HOT3D wrist quaternion would create a training-inference mismatch with Unity's OVR controller. World-frame direction avoids this: only positions are needed. Absolute wrist position is excluded for the same reason — it varies across recordings and does not generalize.

> **Why not 33 one-hot?** Representing objects by physical properties (grip type + size) rather than ID lets the model generalize to unseen objects. The 33 HOT3D objects naturally collapse into 4 grip categories.

### Output

| Parameter | Size | Description |
|-----------|------|-------------|
| joint_angles | 22 | UME joint angles (radians, normalized) — joints 20–21 always 0 (placeholder) |
| wrist_rot_6d | 6 | Palm orientation as 6D continuous rotation; decoded via Gram-Schmidt in Unity |

> The model predicts **both** finger shape and palm orientation. Wrist position is not predicted — it is always anchored to the controller via a fixed physical offset. See `handPivotOffset` in `AuraXRInferenceManager.cs`.
### Architecture — Two-Branch MLP (~294k parameters)

Spatial approach information and object type/shape information are different kinds of input. Two separate encoders are used before fusion:

```
spatial_input (B, 8)                        object_input (B, 7)
[dir_world(3), dir_obj_local(3),            [grip_oh(4), bbox(3)]
 dist(1), approach_speed(1)]
        │                                          │
  FC(8→256) + LayerNorm + ReLU             FC(7→128) + LayerNorm + ReLU
  Dropout(0.20)                            FC(128→128) + ReLU
  FC(256→128) + ReLU                               │
        │                                          │
        └──────────────── cat(256) ────────────────┘
                               │
                         FC(256→256) + ReLU + Dropout(0.40)
                         FC(256→256) + ReLU + Dropout(0.20)
                               │
        ┌──────────────────────┼─────────────────────────┐
        │                      │                         │
  5 × Finger Heads      Wrist Rotation Head      Grip Classifier
  (thumb–pinky)         FC(256→64)+ReLU          (train only, not in ONNX)
  FC(256→64)+ReLU       Dropout(0.10)            FC(256→32)+ReLU
  Dropout(0.10)         FC(64→6)                 FC(32→4)
  FC(64→4)                    │
        │               wrist_rot_6d (B, 6)
   cat(20) + zeros(2)
        │
  joint_angles (B, 22)
```

- **Linear output** (no Tanh) — range enforced via loss range penalty instead of output bounding
- **Per-finger heads** — independent learned parameters per finger; finger-specific loss weights
- **Grip classifier** — active only during training; regularizes trunk to stay grip-category-aware
- **Separate models** for left and right hand (same architecture, different weights)

### Why Two Separate Models?

- Separate model for left hand, separate model for right hand
- Advantage: independent training, easier to debug, each model specializes
---

## Unity Integration

### Platform
- Unity 2021.3+ LTS
- Meta XR SDK
- Quest 2/3/Pro

### Logic

1. **Every frame:**
   - Get controller position and rotation
   - Detect closest object within 40cm (`ProximityDetector`)
   - Compute distance, dir_world, dir_obj_local, approach_speed

2. **Decision based on distance:**
   - > 40cm: Model does not run → fixed neutral pose
   - 10–40cm: Model runs → pre-shape prediction
   - < 10cm: Model runs → grip prediction
   - Contact + grip button: pose locks (grab override blends to closed fist)

3. **Inference every N frames** (default N=2, ~36 fps at 72 Hz):
   - Assemble 15-dim feature vector
   - Normalize with `model_meta.json` stats
   - Run ONNX model via Unity Sentis
   - Denormalize 22 UME angles and 6D wrist rotation
   - EMA smooth angles (α=0.35)
   - Map UME[22] → MANO[15] → drive finger rig

4. **Every frame:**
   - Position virtual hand at `controller.position + predictedRot * handPivotOffset`
   - Apply cached rotation from last inference

### Contact Detection

- Collider per finger
- Collision detection with object surface
- Grip button state check

### Pose Smoothing

The model is a stateless MLP, so large jumps can occur between consecutive frames. A layered smoothing system prevents this:

**Step 1 — Threshold Transition Blend Zone**
Turning the model on/off hard at 40cm causes a snap. Instead, between 30–40cm the model output is blended with FIXED_DEFAULT_POSE based on distance:
```
α = clamp((40 - distance) / 10, 0, 1)
pose = lerp(FIXED_DEFAULT_POSE, model_output, α)
```
- At 40cm: α=0 → pure default pose
- At 30cm: α=1 → pure model output
- In between: smooth transition

**Step 2 — Input EMA (Input Smoothing)**
Input data is smoothed before the model runs. Prevents jumps from controller tracking noise and sudden nearest-object switches.
```
smooth_input[t] = α_in × raw_input[t] + (1 - α_in) × smooth_input[t-1]
```
- Applied to: object relative position, object distance
- Recommended α_in: 0.3–0.5

**Step 3 — Model Inference (Every N Frames)**
With `Inference Every N Frames = 2`, the model doesn't run every frame. Unity lerp fills the gaps.

**Step 4 — Output EMA (Output Smoothing)**
Instead of applying model output directly, it is blended with the previous smooth pose:
```
smooth_pose[t] = α_out × model_output[t] + (1 - α_out) × smooth_pose[t-1]
```
- Applied to: all 22 joint angle values
- Recommended α_out: 0.2–0.4

**Step 5 — Delta Clamp (Hard Jump Prevention)**
Physically limit how much a joint angle can change in a single frame:
```
delta = smooth_pose[t] - current_pose[t-1]
new_pose = current_pose[t-1] + clamp(delta, -max_delta, max_delta)
```
- max_delta tuned experimentally per joint

**Implementation Priority:**
1. Step 1 (blend zone) — solves threshold transition
2. Steps 2 + 4 (EMA input/output) — reduces general noise
3. Add Step 5 (delta clamp) if still not smooth enough

---

## Object Categories

### By Grip Type

| Category | Objects | Grip Type |
|----------|---------|-----------|
| Cylindrical | Cup, bottle, container | Power grip (fist wrap) |
| Thin/Long | Spoon, pen, spatula | Precision grip (fingertip) |
| Flat/Wide | Plate, keyboard, phone | Palmar grip (palm) |
| Small | Mouse, puzzle | Pinch grip |

### By Size (bbox half-extents as stored in grip_categories.py)

| Category | Half-extent range | Example Objects |
|----------|------------------|----------------|
| Small | < 0.05m | Mouse, puzzle, small holder |
| Medium | 0.05–0.10m | Cup/mug, bottle, phone |
| Large | > 0.10m | Plate, keyboard, vase |
---

## Testing Process

The model is tested in four stages. Each stage is a prerequisite for the next.

### Stage 1 — Python Offline Evaluation

The model runs on HOT3D val split frames and predictions are compared against ground truth. No smoothing applied — raw model performance is measured.

**Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| Joint Angle MAE | Mean absolute error in joint angles (degrees) | < 5° |
| MPJPE | Mean per-joint position error (mm) | < 20 mm |
| Per-object error | Separate MAE for bottle and cup | Should be balanced |

> Note: Low numerical error does not guarantee natural appearance in VR. The true criterion is subjective naturalness.

### Stage 2 — Python Simulation Test

A synthetic approach trajectory is created instead of real data:

```
distance: 40cm → 30cm → 20cm → 10cm → 5cm → 2cm
object: fixed (e.g. Bottle)
```

Predicted joint angles are plotted at each distance step. Expected behavior:
- Fingers should begin opening at 40cm
- Grip shape should become defined at 10cm
- Full grasp pose at 2cm

Any jumps or nonsensical transitions are caught here before moving to Unity.

### Stage 3 — Unity Editor Test (No Quest Required)

After ONNX export, tested in Unity Play mode. A virtual controller path is simulated — no device needed.

**Checks:**
- Does the hand pose change when approaching Bottle?
- Is inference time reasonable in Console? (target < 5ms on Quest)
- Does EMA + delta clamp smoothing work?
- Does the pose break when switching from Bottle to Cup?

### Stage 4 — Full On-Device Test (Quest)

Every frame is logged by `SessionDataLogger`. Logs are analyzed in Python after the session.

**Checks:**
- Is the joint angle distribution within expected ranges during real use?
- Are correct angles produced during the grip phase?
- Three-condition comparison: VirtualHands vs Controller vs StaticPose

**Evaluation protocol (minimum):**

| Criterion | Method |
|-----------|--------|
| Naturalness | 5-point Likert ("did the hand movement look natural?") |
| Task time | Automatic measurement via SessionDataLogger |
| Workload | NASA-TLX questionnaire |
| Comparison | 3 conditions, Latin square order |
| Participant count | ~15–20 (medium effect size, 80% power) |
| Ethics approval | IRB/ethics committee approval required before data collection |

> Low numerical error (MAE) does not guarantee good appearance in VR — subjective evaluation is mandatory.

---

## HOT3D → Unity Bridge

HOT3D has no controller data. Instead, every frame provides a **wrist position and rotation**. In Unity it is the opposite: the controller is available, but the wrist is not.

### Why This Is Not a Problem

The model uses **world-frame direction** (not wrist-local), so no wrist quaternion is needed at runtime:

```
Training  →  dir_world = (obj_world − wrist_world) / dist     (HOT3D wrist position only)
Unity     →  dir_world = (obj_world − ctrl_world)  / dist     (controller position only)
```

Since controller position ≈ wrist position when holding a Quest controller, the two are virtually identical. No wrist quaternion from HOT3D is used — this eliminates any tracking-system mismatch between HOT3D's UME tracker and Unity's OVR system.

The wrist rotation output (`wrist_rot_6d`) is predicted **relative to the approach direction** via a canonical frame (`LookRotation(dir_world)`), so it too requires only the world-frame direction — no absolute wrist quaternion.

### Wrist Visualization

The wrist position is anchored to the controller via a fixed physical offset. The model predicts **palm orientation** (wrist_rot_6d) and **finger angles** (joint_angles):

```csharp
// Position: always anchored to controller + measured offset
virtualHand.position = controller.position + predictedRot * handPivotOffset;

// Rotation: from model wrist_rot_6d prediction (decoded via Gram-Schmidt)
virtualHand.rotation = predictedRot;

// Fingers: from model joint_angles prediction (UME[22] → MANO[15])
ApplyFingerAngles(manoJointAngles);
```

`handPivotOffset = (0.1685, 0, 0.0351)` — measured from one session. This shifts the wrist mesh 16.85 cm in X and 3.51 cm in Z from the Quest 3 controller tracking origin to match where the wrist physically sits when holding the controller.

### Coordinate System

Both HOT3D and Unity use world space. HOT3D world space is per-recording (each sequence has its own origin). Unity world space is the scene coordinate system. This does not matter because the model input is always a relative position in the wrist/controller frame — coordinate-system agnostic by construction.

### Summary

| | Training (HOT3D) | Unity Runtime |
|---|---|---|
| Wrist position | HOT3D `wrist_xform.t_xyz` | Controller `position` |
| Wrist rotation | Not used in spatial features | Not used in spatial features |
| dir_world | `(obj − wrist) / dist` (world frame) | `(obj − ctrl) / dist` (world frame) |
| dir_obj_local | `rotate(inv(q_obj), dir_world)` | `ToHOT3DQuat(obj.rotation)^-1 * dirWorld` |
| approach_speed | `dot(wrist_velocity, dir_world)` | `dot(ctrl_velocity_hot3d, dirWorld)` |
| Object info | Ground truth CSV annotation | `ProximityDetector` + `InteractableObject` |
| Wrist visualization | — | `controller.position + predictedRot * handPivotOffset` |

---

## Training Strategy

### Data Preparation

```bash
python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/right/ --hand right
python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/left/  --hand left
```

1. Extract frames from HOT3D Quest 3 ZIPs
2. Keep only frames where distance ≤ 40cm and `hand_confidence ≥ 0.70`
3. For each frame: compute 15-dim feature vector (dir_world, dir_obj_local, dist, approach_speed, grip_oh, bbox)
4. Label: `"grip"` if dist < 10cm, else `"pre_shape"`
5. 85/15 train/val split by whole sequence (no frame-level leakage)
6. Compute per-feature z-score normalization on training set only
7. Oversample grip frames 10× in training split to balance class distribution
8. Save to `dataset.h5` (features, targets, wrist_rot_6d, distances, labels)

### Training

```bash
python train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/ --resume
python train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/  --resume
```

- Loss: **Compound loss** — weighted Huber (β=0.5) over joints 0–19, 2× weight for grip frames (< 10cm), range penalty [0, 2.0 rad], DIP-PIP coupling (DIP ≈ 0.67×PIP), grip classifier CE; plus 0.3 × wrist rotation MSE
- Optimizer: **AdamW** (lr=5e-3, weight_decay=3e-4)
- Batch size: **131072** (entire training set fits in device memory — one gradient step per epoch)
- Epochs: up to **50000** (early stopping: patience=2000)
- LR schedule: Linear warmup (200 epochs) + CosineAnnealingWarmRestarts (T₀=4000, T_mult=2)
- Hardware: auto-detects CUDA > MPS (Apple Silicon) > CPU

### Export

- PyTorch → ONNX
- UmeTrack format preserved
- Inference with Unity Sentis

---

## Advantages

| Approach | Advantage |
|----------|-----------|
| HOT3D + UmeTrack | Natural fit with Quest SDK |
| Proximity-based | Natural human behavior |
| Object-aware | Different grip per object |
| Independent hands | Realistic two-hand interaction |
| Smooth transition | No snapping, fluid animation |

---

## Potential Challenges

### Critical — Address Now

| Challenge | Solution |
|-----------|----------|
| Far frames corrupt training | Use only 0–40cm frames |
| Close frames are few, loss dominated by easy cases | Grip 10× oversampling + 2× Huber weight for grip frames in compound loss |
| No evaluation protocol | Define Likert + NASA-TLX + task time |

### Medium — Fix If Needed

| Challenge | Solution |
|-----------|----------|
| Nearest object switches rapidly | Hysteresis: only switch when new object is significantly closer |
| Model receives garbage when no object nearby | Don't run model — lerp to default pose |
| Anatomically impossible poses | Try first; clamp to joint angle limits if needed |

### Known Limitations — Out of Thesis Scope

| Challenge | Status |
|-----------|--------|
| Hand-object interpenetration | Requires IK/physics simulation — future work |
| Free hand vs controller-gripping hand domain gap | Low impact — predicting virtual pose, not real hand |
| Coordinate conversion | Handled by AuraXRFeatureAssembler |
| Quest inference < 5ms | MLP is small — measure to confirm |

---

## Project Phases

### Phase 1: Data Preparation ✅ DONE
- [x] Extract training data from HOT3D Quest 3 ZIPs
- [x] Build 15-dim feature vectors (dir_world + dir_obj_local + dist + approach_speed + grip_oh + bbox)
- [x] Label by distance thresholds (grip/pre_shape), 85/15 train/val split by sequence
- [x] Grip frame 10× oversampling; normalization stats stored in dataset.h5

### Phase 2: Model Training ✅ DONE
- [x] Two-branch MLP with per-finger heads (~294k params)
- [x] Compound loss (Huber + range penalty + DIP-PIP coupling + grip classifier)
- [x] Separate training for left and right hand
- [x] Evaluation: right MAE ≈ 14.0°, left MAE ≈ 13.8° (target < 5°)

### Phase 3: ONNX Export ✅ DONE
- [x] PyTorch → ONNX (opset 14, bitwise validation with ONNX Runtime)
- [x] model_meta.json with normalization stats copied to onnx/ for Unity

### Phase 4: Unity Integration 🔄 IN PROGRESS
- [x] Meta XR SDK setup, Unity Sentis ONNX loading
- [x] Feature assembly (AuraXRFeatureAssembler.cs)
- [x] Inference pipeline (AuraXRInferenceManager.cs)
- [x] Hand rig visualization (HandRigController.cs, HandSkeletonAnchor.cs)
- [x] Interaction and task system (VirtualHandGrab.cs, TaskScoreUI.cs)
- [ ] Quest 3 on-device validation

### Phase 5: Testing & Refinement ⏳ WAITING
- [ ] On-device testing on Quest 3 (BackendType.CPU first)
- [ ] Inference timing (target < 5ms/frame)
- [ ] Ablation table (feature masking experiments)
- [ ] User study (IRB required)

---

## Expected Results

1. **Naturally appearing hand pre-shaping** — fingers begin curling at ~40cm approach
2. **Object-specific grip** — power grip for mug, precision grip for spoon, etc.
3. **Smooth transitions** — EMA smoothing (α=0.35) eliminates frame-to-frame jitter
4. **Real-time performance** — ~294k parameter MLP targets < 5ms/frame on Quest 3
5. **Two-hand support** — left and right models run independently each frame
---

## References

- HOT3D Dataset: Meta Reality Labs, CVPR 2025
- UmeTrack: Meta, SIGGRAPH Asia 2022
- MANO: Max Planck Institute, 2017
- Anticipatory Grasp: Widely referenced concept in literature

---

*This document explains the project logic. Technical implementation details will be covered in separate documents.*
