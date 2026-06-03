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

The human hand pre-shapes itself as it reaches toward an object. The system covers 4 states:

### State 0: Default (> 40cm)

**State:** Hand is far from any object

**Behavior:**
- Model does not run
- Hand stays in fixed default pose — `FIXED_DEFAULT_POSE` (measured once: natural, slightly curled finger position when holding the controller)

---

### Phase 1: Pre-Shape (10cm – 40cm)

**State:** Hand is beginning to approach the object

**Behavior:**
- Hand leaves its default pose
- Preparation begins based on object type
- Example: fingers open and palm rounds for a cup
- Example: thumb and index finger prepare for a pen

**Model Task:** Predict "preparation pose" based on object type and dimensions

---

### Phase 2: Approach (< 10cm)

**State:** Hand is very close to the object, ready to grasp

**Behavior:**
- Hand transitions to full grip position
- Fingers orient toward the object surface
- Pose becomes more precise as distance decreases
- Smooth interpolated transition

**Model Task:** Predict "grasp pose" appropriate for object shape

---

### Phase 3: Grasp (Contact + Grip)

**State:** Hand has made contact, user pressed grip button

**Behavior:**
- Fingers wrap around the object
- Finger angles adjusted to object shape
- Physical attachment established (object sticks to hand)
- Pose maintained during hold

**Model Task:** Predict "final grasp pose" based on object position and grip category — pose is locked and held for the duration of the grasp

---

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
| Object relative position | 3 | x, y, z offset in wrist frame — encodes both direction and distance |
| Grip category | 4 | Power / Precision / Palmar / Pinch — one-hot |
| Object dimensions | 3 | Bounding box half-extents (x, y, z in meters) |
| Distance | 1 | Scalar, in meters |
| **Total** | **11** | |

> **Why not wrist position/rotation?** Grip shape depends on where the object is relative to the hand and what type of object it is — not on where the hand is in the room. Absolute wrist position varies across recordings and does not generalize. Wrist rotation is already implicit in the relative position (which is computed in the wrist frame). Dropping both keeps the model simple and coordinate-system agnostic.

> **Why not 33 one-hot?** Representing objects by physical properties rather than ID allows the model to generalize to unseen objects. Learning "large cylinder 12cm ahead" is far more meaningful than memorizing "object_id=17". The 33 HOT3D objects naturally collapse into 4 grip categories.

### Output

| Parameter | Size | Description |
|-----------|------|-------------|
| Joint angles | 22 | UmeTrack format — finger angles only |

> **Wrist does not come from the model.** The wrist is anchored to the controller via a fixed physical offset. The model only predicts fingers. See HOT3D → Unity Bridge.

### Architecture — Two-Branch MLP

Spatial approach information and object type/shape information are different kinds of input. Two separate encoders are used:

```
Relative position (3) + Distance (1)     →  [Spatial Encoder  2×FC]  →  spatial_emb (32)
Grip category (4) + Size (3)             →  [Object Encoder   2×FC]  →  obj_emb     (32)
                                                                               ↓
                                                      [Concat → Prediction Head 2×FC]
                                                                               ↓
                                                                   Joint angles (22)
```

- Each encoder: 2 fully-connected layers, ReLU activation
- Prediction head: 2 fully-connected layers, Tanh on final layer
- Tanh output [-1, 1] → denormalized back to real angle values
- Normalization parameters stored in model metadata file

### Why Two Separate Models?

- Separate model for left hand
- Separate model for right hand
- Advantage: Independent training, easier to debug
- Same architecture, different weights

---

## Unity Integration

### Platform
- Unity 2021.3+ LTS
- Meta XR SDK
- Quest 2/3/Pro

### Logic

1. **Every frame:**
   - Get controller position
   - Detect closest object
   - Calculate distance

2. **Decision based on distance:**
   - > 40cm: Model does not run, FIXED_DEFAULT_POSE
   - 10–40cm: Model → pre-shape prediction + interpolation
   - < 10cm: Model → grip prediction
   - Contact + Grip button: Final grip pose locked

3. **Visualization:**
   - UmeTrack joint angles → apply to Unity hand model
   - Smooth transition (Lerp/Slerp)
   - Both hands updated independently

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

### By Size

| Category | Diameter | Example Objects |
|----------|---------|----------------|
| Small | < 12cm | Mouse, puzzle, box |
| Medium | 12–22cm | Cup, bottle, phone |
| Large | > 22cm | Plate, keyboard, vase |

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

The model input is the object's position **relative to the hand in the hand's own frame** — not the hand's absolute pose in the world. This relative position is computed differently in training vs. runtime, but the result is equivalent:

```
Training  →  rel_pos = R_wrist^T × (obj_world − wrist_world)      (HOT3D wrist data)
Unity     →  rel_pos = R_controller^T × (obj_world − ctrl_world)  (real controller data)
```

When holding a Quest controller, controller position ≈ wrist position and controller rotation ≈ wrist rotation. So the relative position computed from either source is virtually identical.

### Wrist Visualization

The wrist is anchored to the controller with a fixed physical offset. The model only predicts finger angles — it never touches wrist position:

```csharp
// Wrist — always coincides with the controller
anchor.position = controller.position + FIXED_GRIP_OFFSET;
anchor.rotation = controller.rotation * FIXED_GRIP_ROTATION;

// Model updates fingers only
ApplyFingerAngles(modelOutput.jointAngles);
```

`FIXED_GRIP_OFFSET` and `FIXED_GRIP_ROTATION` are measured once physically: the position of the wrist center relative to the Quest controller's tracked origin when held naturally.

### Coordinate System

Both HOT3D and Unity use world space. HOT3D world space is per-recording (each sequence has its own origin). Unity world space is the scene coordinate system. This does not matter because the model input is always a relative position in the wrist/controller frame — coordinate-system agnostic by construction.

### Summary

| | Training (HOT3D) | Unity Runtime |
|---|---|---|
| Wrist/controller data | HOT3D wrist_transform | Real controller transform |
| Relative position | `R_wrist^T × (obj − wrist)` | `R_ctrl^T × (obj − ctrl)` |
| Object info | Ground truth annotation | ProximityDetector |
| Wrist visualization | — | Controller + FIXED_GRIP_OFFSET |

---

## Training Strategy

### Data Preparation

1. Extract frames from HOT3D clips
2. **Keep only frames where distance is 0–40cm** — model runs from 40cm in Unity, frames beyond that are unused
3. For each frame:
   - Hand pose (UmeTrack 22 joints)
   - Closest object type and dimensions
   - Hand-object distance
4. Label by distance thresholds:
   - Pre-shape (10–40cm): Preparation pose sample
   - Grip (< 10cm): Grasp pose sample

### Training

- Loss: **Weighted MSE** — higher weight for 0–10cm frames (few but critical)
- Optimizer: Adam
- Batch size: 64
- Epochs: 50–100

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
| Close frames are few, loss dominated by easy cases | Weighted MSE, higher weight for close frames |
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

### Phase 1: Data Preparation
- [ ] Extract training data from HOT3D
- [ ] Build hand-object pairs
- [ ] Label by distance thresholds

### Phase 2: Model Training
- [ ] UmeTrack-based MLP model
- [ ] Separate training for left and right hand
- [ ] Validation and testing

### Phase 3: ONNX Export
- [ ] PyTorch → ONNX conversion
- [ ] Input/output format validation

### Phase 4: Unity Integration
- [ ] Meta XR SDK setup
- [ ] ONNX model loading (Sentis)
- [ ] Proximity system implementation
- [ ] Hand model visualization

### Phase 5: Testing & Refinement
- [ ] Testing on Quest
- [ ] Performance optimization
- [ ] User experience evaluation

---

## Expected Results

1. **Naturally appearing hand movement** — when approaching an object
2. **Object-specific grip** — appropriate grasp for each object
3. **Smooth transitions** — fluidity between phases
4. **Real-time performance** — 72+ FPS on Quest
5. **Two-hand support** — independent and coordinated

---

## References

- HOT3D Dataset: Meta Reality Labs, CVPR 2025
- UmeTrack: Meta, SIGGRAPH Asia 2022
- MANO: Max Planck Institute, 2017
- Anticipatory Grasp: Widely referenced concept in literature

---

*This document explains the project logic. Technical implementation details will be covered in separate documents.*
