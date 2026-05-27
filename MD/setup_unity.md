# AuraXR Unity — Setup Guide

> This guide contains all steps to set up the Unity project from scratch. The project is already configured; this document is kept as a reference and for troubleshooting.

---

## CRITICAL SETUP ORDER

```
1.  Create Unity project + switch platform to Android
2.  Install packages (Sentis, Oculus, XRI, TMP)
3.  Copy all scripts + model files into Assets
4.  Create custom layers (before any scene objects)
5.  Set Layer Collision Matrix
6.  Add OVRInteractionComprehensive prefab
7.  Add CharacterController to OVRCameraRig
8.  Add ThumbstickLocomotion to OVRCameraRig
9.  Build scene environment (Floor, Walls, Table) — assign Environment layer immediately
10. Add Bottle and Cup — assign Interactable layer immediately
11. Add GameManager + all script components
12. Add AuraXRAutoWire to GameManager — auto-wires all Inspector references
13. Create LeftHandRig and RightHandRig — assign HandRig layer
14. Materials: hand = Fade rendering, Bottle/Cup = Standard + Emission ON
15. Build TaskCanvas (World Space)
16. Press Play and verify Console

NOTE: AuraXRAutoWire automatically wires all Inspector cross-references in Awake.
      No manual Inspector assignments needed — the script self-destructs after running.
```

---

# PHASE 1: PROJECT SETUP

## Chapter 1.1: Create Unity Project

1. Open **Unity Hub** → **New Project**
2. Template: **3D (Built-in RP)**
3. Name: `AuraXR_HandPose_V1`
4. Location: `/Users/muratcelik/Desktop/Thesis/Unity_Projects/AuraXR_HandPose_V1`
5. Unity version: **2023.2.20f1**
6. **Create Project** (wait 2–3 min)

### Switch Platform to Android

1. **File → Build Settings**
2. Select **Android** → **Switch Platform** (wait 1–2 min)

### Install Packages

**Window → Package Manager → + → Add package from git URL**

Add in order:
1. `com.unity.sentis`
2. `com.unity.xr.plugin.oculus`
3. `com.unity.xr.interaction.toolkit`
4. `com.unity.textmesh-pro`

After TMP installs: **Window → TextMesh Pro → Import TMP Essentials**

### Configure XR Plugin

1. **Edit → Project Settings → XR Plugin Management → Android tab**
2. Check ✓ **Oculus** → wait for recompilation

### Create Folder Structure

**Project** tab → right-click empty area → **Create Folder** for each:

```
Assets/AuraXR/Scripts
Assets/AuraXR/Models
Assets/AuraXR/Prefabs
Assets/AuraXR/Rigs
Assets/AuraXR/Materials
Assets/AuraXR/UI
Assets/AuraXR/Data
```

---

## Chapter 1.2: Import Model Files

Copy from `/Users/muratcelik/Desktop/Thesis/Workspace/V3/data/`:
- `intentformer.onnx`
- `intentformer_meta.json`

Paste into `Assets/AuraXR/Models/`.

In **Project** tab → click `intentformer_meta.json` → Inspector confirms **TextAsset** → **Apply**.

---

# PHASE 2: SCRIPTS & INFERENCE PIPELINE

## Chapter 2.1: Copy All Scripts

Copy all files from `UnityScripts/` to `Assets/AuraXR/Scripts/`:

```
AuraXRMetaLoader.cs
AuraXRFeatureAssembler.cs
AuraXRInferenceManager.cs
HandRigController.cs
InteractableObject.cs
ProximityDetector.cs
HandVisibilityController.cs
ThumbstickLocomotion.cs
GraspIndicator.cs
VirtualHandGrab.cs
HandProximityVisibility.cs
SessionDataLogger.cs
ScenarioKitchenTask.cs
HapticFeedbackManager.cs
UITaskDisplay.cs
SoundManager.cs
ConditionManager.cs
```

Verify: no red icons in **Console**.

---

## Chapter 2.2: Create GameManager

1. **Hierarchy** → right-click → **Create Empty** → name `GameManager`
2. Select `GameManager` → **Add Component** for each of the following:

```
AuraXRMetaLoader
AuraXRFeatureAssembler
AuraXRInferenceManager
ProximityDetector
HandProximityVisibility
VirtualHandGrab
SessionDataLogger
ScenarioKitchenTask
HapticFeedbackManager
SoundManager
ConditionManager
```

### Inspector Assignments

| Component | Field | Value |
|-----------|-------|-------|
| AuraXRMetaLoader | Meta Json | `intentformer_meta.json` |
| AuraXRInferenceManager | Model Asset | `intentformer.onnx` |
| AuraXRInferenceManager | Meta Loader | `GameManager` |
| AuraXRInferenceManager | Feature Assembler | `GameManager` |
| AuraXRInferenceManager | Inference Every N Frames | `2` |
| AuraXRFeatureAssembler | Left Controller Transform | `LeftControllerAnchor` ← assign after Phase 6.1 |
| AuraXRFeatureAssembler | Right Controller Transform | `RightControllerAnchor` ← assign after Phase 6.1 |
| ProximityDetector | Feature Assembler | `GameManager` |
| ProximityDetector | Search Radius | `2.0` |
| VirtualHandGrab | Left Hand Wrist | `LeftHandRig` ← assign after Phase 4 |
| VirtualHandGrab | Right Hand Wrist | `RightHandRig` ← assign after Phase 4 |
| VirtualHandGrab | Grab Radius | `0.15` |
| VirtualHandGrab | Grip Threshold | `0.7` |
| VirtualHandGrab | Throw Multiplier | `1.5` |
| SessionDataLogger | Inference Manager | `GameManager` |
| SessionDataLogger | Feature Assembler | `GameManager` |
| SessionDataLogger | Enable Logging | ✓ |

---

## Chapter 2.3: Play Mode Test

Press **Play** (▶). Console must show:
```
[AuraXR] Meta loaded. Feature=96  Target=78  T=16
[AuraXR] Model loaded. Input: [1,16,96]  Output: [1,78]
```

If not:
- `metaJson not assigned` → drag `intentformer_meta.json` to AuraXRMetaLoader
- `modelAsset not assigned` → drag `intentformer.onnx` to AuraXRInferenceManager
- `NullReferenceException` → a component reference is missing

Stop Play mode.

---

# PHASE 3: CONTROLLER TRACKING

## Chapter 3.1: Wire Controller Transforms

After adding `OVRInteractionComprehensive` in Phase 6.1, come back here:

1. Select `GameManager` → **AuraXRFeatureAssembler** in Inspector
2. **Left Controller Transform** → drag `LeftControllerAnchor` (OVRCameraRig → TrackingSpace)
3. **Right Controller Transform** → drag `RightControllerAnchor` (OVRCameraRig → TrackingSpace)

### Disable Hand Visuals

1. Select `OVRLeftHandVisual` → uncheck the GameObject
2. Select `OVRRightHandVisual` → uncheck the GameObject

### Disable Locomotor

Select `Locomotor` → uncheck the GameObject

---

# PHASE 4: HAND RIGGING

## Chapter 4.1: Import MANO Hand Model

> **Note:** The MANO download provides `.pkl` files (Python parametric model), not FBX. Do NOT try to import the pkl into Unity. Instead use the OVR hand prefabs already included in the Oculus SDK — they are already in `Assets/AuraXR/Rigs/` as `OVRCustomHandPrefab_L` and `OVRCustomHandPrefab_R`. No conversion or import needed.

The pkl files are used only by the Python training pipeline (HOT3D dataset annotations). The Unity visual mesh uses the OVR hand model.

---

## Chapter 4.2: Create Hand Prefabs

### OVR Bone → MANO Joint Mapping (Left Hand)

The OVRCustomHandPrefab uses this bone naming. Map them to MANO order for the `fingerJoints` array:

| Element | MANO joint | OVR bone |
|---------|-----------|---------|
| 0 | Thumb proximal | `b_l_thumb1` (child of b_l_thumb0) |
| 1 | Thumb middle | `b_l_thumb2` |
| 2 | Thumb distal | `b_l_thumb3` |
| 3 | Index proximal | `b_l_index1` |
| 4 | Index middle | `b_l_index2` |
| 5 | Index distal | `b_l_index3` |
| 6 | Middle proximal | `b_l_middle1` |
| 7 | Middle middle | `b_l_middle2` |
| 8 | Middle distal | `b_l_middle3` |
| 9 | Ring proximal | `b_l_ring1` |
| 10 | Ring middle | `b_l_ring2` |
| 11 | Ring distal | `b_l_ring3` |
| 12 | Pinky proximal | `b_l_pinky1` (child of b_l_pinky0) |
| 13 | Pinky middle | `b_l_pinky2` |
| 14 | Pinky distal | `b_l_pinky3` |

> **Skip** `b_l_thumb0`, `b_l_pinky0`, `b_l_forearm_stub`, and all `_marker` bones.
> For the right hand replace `b_l_` with `b_r_`.

### LeftHandRig

1. **Hierarchy** → right-click → **Create Empty** → name `LeftHandRig`
2. Set **Layer** → `HandRig`
3. Drag `OVRCustomHandPrefab_L` from `Assets/AuraXR/Rigs/` into `LeftHandRig` as a child
4. Select `LeftHandRig` → **Add Component → HandRigController**
5. In **HandRigController** Inspector:
   - **Inference Manager** → drag `GameManager`
   - **Is Left Hand** → ✓
   - **Finger Joints** → Size: `15`, drag the 15 bones per the table above
6. Right-click `LeftHandRig` → **Prefab → Create Prefab** → save to `Assets/AuraXR/Prefabs/`

### RightHandRig

Repeat all steps above using `OVRCustomHandPrefab_R`, name `RightHandRig`, **Is Left Hand** → ☐, use `b_r_` bone names.

### Wire to InferenceManager

Select **GameManager** → **AuraXRInferenceManager** component:
- **Virtual Hand Left** → drag `LeftHandRig`
- **Virtual Hand Right** → drag `RightHandRig`

---

# PHASE 5: OBJECT SCRIPTS

Scripts already copied in Chapter 2.1 and ProximityDetector already added to GameManager in Chapter 2.2.

`InteractableObject.cs` and `GraspIndicator.cs` are added directly to scene objects in Phase 6.3.

---

# PHASE 6: SCENE SETUP

## Chapter 6.1: Add OVRInteractionComprehensive

1. **Project** tab → search `OVRInteractionComprehensive` → drag into **Hierarchy**
2. Do **not** add a second `OVRManager`

```
OVRCameraRig
└── TrackingSpace
     ├── CenterEyeAnchor
     ├── LeftControllerAnchor           ← FeatureAssembler.leftControllerTransform
     │    └── OVRLeftControllerVisual
     ├── RightControllerAnchor          ← FeatureAssembler.rightControllerTransform
     │    └── OVRRightControllerVisual
     ├── LeftHandAnchor
     ├── RightHandAnchor
     └── OVRInteractionComprehensive
          ├── OVRLeftHandVisual          ← disable
          ├── OVRRightHandVisual         ← disable
          └── Locomotor                  ← disable
```

Now return to **Chapter 3.1** to assign LeftControllerAnchor / RightControllerAnchor.

---

## Chapter 6.2: Layer & Collision Matrix

### Create Custom Layers

**Edit → Project Settings → Tags and Layers**

| Slot | Name |
|------|------|
| 6 | `Environment` |
| 7 | `Player` |
| 8 | `Interactable` |
| 9 | `HandRig` |

### Layer Collision Matrix

**Edit → Project Settings → Physics** → **Layer Collision Matrix** at the bottom:

| Pair | Collide? |
|------|----------|
| `Player` ↔ `Environment` | ✓ ON |
| `Interactable` ↔ `Environment` | ✓ ON |
| `HandRig` ↔ `Environment` | OFF |
| `HandRig` ↔ `Interactable` | OFF |
| `Player` ↔ `Interactable` | OFF |
| `Player` ↔ `HandRig` | OFF |

### Physics Solver

Still in **Edit → Project Settings → Physics**:
- **Default Solver Iterations**: `12`
- **Default Solver Velocity Iterations**: `4`

> Complete **step2_unity.md** (CharacterController + ThumbstickLocomotion) before building the environment below.

---

## Chapter 6.3: VR Kitchen Environment

### Floor

1. **Hierarchy** → right-click → **3D Object → Plane** → name `Floor`
2. Position `(0, 0, 0)`, Scale `(0.4, 1, 0.4)`
3. **Layer** → `Environment`
4. Verify **MeshCollider** present, **Is Trigger = OFF**
5. Create `FloorMaterial`: Standard, Albedo RGB (180, 160, 130), Metallic 0, Smoothness 0.2
6. Apply to `Floor`

### Walls

1. **Hierarchy** → right-click → **Create Empty** → name `Walls`
2. Right-click `Walls` → **3D Object → Cube** for each:

| Name | Position | Scale |
|------|----------|-------|
| `WallBack` | (0, 1.25, 2.0) | (4, 2.5, 0.1) |
| `WallLeft` | (-2.0, 1.25, 0) | (0.1, 2.5, 4) |
| `WallRight` | (2.0, 1.25, 0) | (0.1, 2.5, 4) |

Each wall: **Layer** → `Environment`, BoxCollider **Is Trigger = OFF**

Create `WallMaterial`: Standard, Albedo RGB (220, 215, 200), Smoothness 0.05. Apply to all three.

### Table

1. **Hierarchy** → right-click → **Create Empty** → name `Table`, position `(0, 0, 1.4)`
2. Right-click `Table` → **3D Object → Cube** → name `TableTop`
   - Local position `(0, 0.75, 0)`, Scale `(1.1, 0.04, 0.65)`
   - **Layer** → `Environment`, BoxCollider **Is Trigger = OFF**
   - Create `TableMaterial`: Standard, Albedo RGB (160, 100, 60), Smoothness 0.4, Metallic 0
3. Right-click `Table` → **3D Object → Cube** for each leg:

| Name | Local Position | Scale |
|------|---------------|-------|
| `LegFrontLeft` | (-0.50, 0.365, 0.29) | (0.05, 0.73, 0.05) |
| `LegFrontRight` | (0.50, 0.365, 0.29) | (0.05, 0.73, 0.05) |
| `LegBackLeft` | (-0.50, 0.365, -0.29) | (0.05, 0.73, 0.05) |
| `LegBackRight` | (0.50, 0.365, -0.29) | (0.05, 0.73, 0.05) |

Each leg: **Layer** → `Environment`, BoxCollider **Is Trigger = OFF**, apply `TableMaterial`.

### Bottle

1. **Hierarchy** → right-click → **3D Object → Cylinder** → name `Bottle`
2. Position `(0.15, 0.925, 1.4)`, Scale `(0.045, 0.155, 0.045)`
3. **Layer** → `Interactable`, CapsuleCollider **Is Trigger = OFF**
4. Create `BottleMaterial`: Standard, Albedo RGB (80, 160, 90), Metallic 0, Smoothness 0.8
   - **Emission**: check the checkbox, leave color black
5. **Add Component → InteractableObject**: Category Id `1`, Name `bottle`
6. **Add Component → Rigidbody**:
   - Mass `0.3`, Drag `1.0`, Angular Drag `0.5`
   - Use Gravity ✓, Is Kinematic ☐
   - Interpolate: `Interpolate`, Collision Detection: `Continuous Dynamic`
   - Constraints → Freeze Rotation: X ✓, Z ✓
7. **Add Component → GraspIndicator**:
   - Left Hand Rig → `LeftHandRig` ← assign after Phase 4
   - Right Hand Rig → `RightHandRig` ← assign after Phase 4
   - Highlight Distance `0.15`

### Cup

1. **Hierarchy** → right-click → **3D Object → Cylinder** → name `Cup`
2. Position `(-0.18, 0.855, 1.4)`, Scale `(0.055, 0.085, 0.055)`
3. **Layer** → `Interactable`, CapsuleCollider **Is Trigger = OFF**
4. Create `CupMaterial`: Standard, Albedo RGB (240, 235, 220), Smoothness 0.6
   - **Emission**: check the checkbox, leave color black
5. **Add Component → InteractableObject**: Category Id `3`, Name `cup`
6. **Add Component → Rigidbody**: same settings as Bottle
7. **Add Component → GraspIndicator**: same settings as Bottle

### Plate (decoration only)

1. **Hierarchy** → right-click → **3D Object → Cylinder** → name `Plate`
2. Position `(-0.18, 0.775, 1.4)`, Scale `(0.14, 0.005, 0.14)`
3. Layer: Default, no components

### Lighting

Delete the default Directional Light if present.

1. **Hierarchy** → right-click → **Light → Directional Light** → name `SunLight`
2. Rotation `(55, -30, 0)`
3. Intensity `1.1`, Color RGB (255, 248, 235), Shadow Type **Soft Shadows**, Shadow Strength `0.7`

**Window → Rendering → Lighting → Environment**:
- Source: Color, Ambient Color RGB (80, 90, 110), Intensity Multiplier `0.6`

### Final Scene Hierarchy

```
Scene
├── GameManager
├── OVRCameraRig  (Layer: Player, CharacterController, ThumbstickLocomotion)
│    └── TrackingSpace
│         ├── CenterEyeAnchor
│         ├── LeftControllerAnchor   ← FeatureAssembler.leftControllerTransform
│         │    └── OVRLeftControllerVisual
│         ├── RightControllerAnchor  ← FeatureAssembler.rightControllerTransform
│         │    └── OVRRightControllerVisual
│         ├── LeftHandAnchor
│         ├── RightHandAnchor
│         └── OVRInteractionComprehensive
│              ├── OVRLeftHandVisual  (disabled)
│              ├── OVRRightHandVisual (disabled)
│              └── Locomotor          (disabled)
├── LeftHandRig   (Layer: HandRig)
├── RightHandRig  (Layer: HandRig)
├── SunLight
├── Floor
├── Walls
│    ├── WallBack
│    ├── WallLeft
│    └── WallRight
├── Table
│    ├── TableTop
│    ├── LegFrontLeft / LegFrontRight
│    └── LegBackLeft  / LegBackRight
├── Bottle  (Layer: Interactable, InteractableObject, Rigidbody, GraspIndicator)
├── Cup     (Layer: Interactable, InteractableObject, Rigidbody, GraspIndicator)
├── Plate   (decoration)
└── TaskCanvas (World Space, pos 0/1.8/1.5)
     ├── InstructionText (TMP)
     └── TimerText (TMP)
```

---

## Chapter 6.4: Test the Full Pipeline

Press **Play** (▶). Console must show:
```
[AuraXR] Meta loaded. Feature=96  Target=78  T=16
[AuraXR] Model loaded. Input: [1,16,96]  Output: [1,78]
```

Stop Play mode.

---

# PHASE 7: HAND MATERIAL

1. **Project** → right-click `Assets/AuraXR/Materials/` → **Create → Material** → name `HandSkinMaterial`
2. Inspector: Shader `Standard`, **Rendering Mode: Fade**, Albedo RGB (230, 180, 140), Metallic 0, Smoothness 0.6
3. Drag onto hand mesh in Hierarchy

---

# PHASE 8: DATA LOGGING

SessionDataLogger already added in Chapter 2.2. Verify Inspector:
- Inference Manager → `GameManager`
- Feature Assembler → `GameManager`
- Enable Logging → ✓

---

# PHASE 8B: UX SCENARIOS

## Chapter 8B.1: Kitchen Scenario State Machine

`ScenarioKitchenTask` already on GameManager. Wire Inspector:

| Field | Value |
|-------|-------|
| Bottle | `Bottle` |
| Cup | `Cup` |
| Feature Assembler | `GameManager` |
| UI Display | `TaskCanvas` ← build first in 8B.4 |
| Sound Manager | `GameManager` |
| Grip Threshold | `0.15` |
| Grip Input Threshold | `0.7` |
| Auto Start | ✓ |

---

## Chapter 8B.2: Haptic Feedback

`HapticFeedbackManager` already on GameManager. Wire Inspector:
- Feature Assembler → `GameManager`
- Haptic Trigger Distance → `0.12`

---

## Chapter 8B.3: Grasp Indicator

Already on Bottle and Cup from Chapter 6.3. Verify both objects have:
- Left Hand Rig → `LeftHandRig`
- Right Hand Rig → `RightHandRig`
- Highlight Distance → `0.15`

---

## Chapter 8B.4: UI Task Display

1. **Hierarchy** → right-click → **UI → Canvas** → name `TaskCanvas`
2. Render Mode: **World Space**
3. Position `(0, 1.8, 1.5)`, Scale `(0.005, 0.005, 0.005)`
4. Right-click `TaskCanvas` → **UI → TextMeshPro - Text** → name `InstructionText`
   - Font size `0.08`, Alignment: Center/Middle, Color: white
5. Right-click `TaskCanvas` → **UI → TextMeshPro - Text** → name `TimerText`
   - Place above InstructionText, Font size `0.06`, Color: light grey
6. Select `TaskCanvas` → **Add Component → UITaskDisplay**
   - Instruction Text → `InstructionText`
   - Timer Text → `TimerText`

---

## Chapter 8B.5: Sound Manager

`SoundManager` already on GameManager. Assign AudioClips in Inspector (leave empty if not yet sourced):
- Pickup Clip, Pour Clip, Place Clip, Complete Clip

---

## Chapter 8B.6: Condition Manager

`ConditionManager` already on GameManager. Wire Inspector:

| Field | Value |
|-------|-------|
| Left Hand Rig | `LeftHandRig` |
| Right Hand Rig | `RightHandRig` |
| Left Controller Model | `OVRLeftControllerVisual` |
| Right Controller Model | `OVRRightControllerVisual` |
| Inference Manager | `GameManager` |
| Debug Condition | `VirtualHands` |

Set condition before each session:
```bash
adb shell am broadcast -a com.aura.setcondition --ei condition 0   # A: VirtualHands
adb shell am broadcast -a com.aura.setcondition --ei condition 1   # B: Controller
adb shell am broadcast -a com.aura.setcondition --ei condition 2   # C: StaticPose
```

Latin square order (3 conditions, 20 participants):
- Group 1: A→B→C
- Group 2: B→C→A
- Group 3: C→A→B

---

## Chapter 8B.7: Final GameManager Component List

| Component | Required Fields |
|-----------|----------------|
| AuraXRMetaLoader | Meta Json |
| AuraXRFeatureAssembler | Left/Right Controller Transform |
| AuraXRInferenceManager | Model Asset, Meta Loader, Feature Assembler, Virtual Hand Left/Right |
| ProximityDetector | Feature Assembler, Search Radius |
| HandProximityVisibility | Left/Right Controller, Left/Right Hand Rig |
| VirtualHandGrab | Left/Right Hand Wrist |
| ScenarioKitchenTask | Bottle, Cup, Feature Assembler, UI Display, Sound Manager, Auto Start ✓ |
| HapticFeedbackManager | Feature Assembler |
| SoundManager | (audio clips optional) |
| ConditionManager | Left/Right Hand Rig, Left/Right Controller Model, Inference Manager |
| SessionDataLogger | Inference Manager, Feature Assembler, Enable Logging ✓ |

---

# PHASE 9: DEPLOYMENT

## Chapter 9.1: Build Settings

1. **File → Build Settings** → confirm **Android** selected
2. **Player Settings**:
   - Package Name: `com.yourname.auraxr`
   - Minimum API Level: `24`, Target API Level: `33`
   - Graphics APIs: Vulkan
   - XR Plug-in Management → Android: Oculus ✓
3. **Edit → Project Settings → Quality** → Level: **High**
   - MSAA: 4x, Anisotropic Filtering: Per Texture

## Chapter 9.2: Build & Deploy

```bash
adb devices    # verify device listed
```

1. **File → Build Settings → Build** → folder: `AuraXR_Build`
2. Install:
```bash
adb install -r AuraXR_Build/app-release.apk
```

Stream logs:
```bash
adb logcat -s "AuraXR"
```

---

# PHASE 12: FINAL CHECKLIST

**Scripts & References**
- [ ] `LeftControllerAnchor` → AuraXRFeatureAssembler.leftControllerTransform
- [ ] `RightControllerAnchor` → AuraXRFeatureAssembler.rightControllerTransform
- [ ] `LeftHandRig` / `RightHandRig` → AuraXRInferenceManager.virtualHandLeft / virtualHandRight
- [ ] `OVRLeftControllerVisual` → ConditionManager.leftControllerModel
- [ ] `OVRRightControllerVisual` → ConditionManager.rightControllerModel
- [ ] `OVRLeftHandVisual` and `OVRRightHandVisual` disabled
- [ ] `Locomotor` disabled

**Layers & Physics**
- [ ] 4 custom layers created (Environment/Player/Interactable/HandRig)
- [ ] Layer Collision Matrix set (Player↔Environment ON, HandRig↔Environment OFF)
- [ ] All Is Trigger = OFF on Floor, Walls, Table, legs
- [ ] CharacterController on OVRCameraRig (step2 Block D.4)
- [ ] ThumbstickLocomotion: Camera Transform = CenterEyeAnchor, Head Collision Layers = Environment
- [ ] Bottle and Cup: Rigidbody + Continuous Dynamic

**Materials**
- [ ] HandSkinMaterial Rendering Mode = **Fade**
- [ ] Emission checkbox ON on BottleMaterial and CupMaterial

**UX**
- [ ] autoStart ✓ on ScenarioKitchenTask
- [ ] GraspIndicator: leftHandRig / rightHandRig assigned on Bottle and Cup
- [ ] TaskCanvas Render Mode = World Space

**Console (Play mode)**
- [ ] Zero red errors
- [ ] `[AuraXR] Meta loaded. Feature=96  Target=78  T=16`
- [ ] `[AuraXR] Model loaded. Input: [1,16,96]  Output: [1,78]`

**Build**
- [ ] `adb devices` lists Quest 3
- [ ] APK installs and launches
- [ ] `adb logcat -s "AuraXR"` confirms inference on device

---

# APPENDIX A: FILE STRUCTURE

```
Assets/
├── AuraXR/
│   ├── Scripts/     (all .cs files)
│   ├── Models/      (intentformer.onnx, intentformer_meta.json)
│   ├── Prefabs/     (LeftHandRig.prefab, RightHandRig.prefab)
│   ├── Rigs/        (hand FBX)
│   ├── Materials/   (HandSkinMaterial, FloorMaterial, WallMaterial, TableMaterial, BottleMaterial, CupMaterial)
│   ├── UI/
│   └── Data/
└── TextMesh Pro/
```

---

# APPENDIX B: Key Formulas

**Output Layout (78 floats):**
```
[0–14]   mano_pose_h0      Left hand joint angles (1 DoF per joint)
[15–24]  mano_betas_h0     Left hand shape (ignored in v1)
[25–27]  wrist_t_h0        Left wrist position (metres)
[28–31]  wrist_q_h0        Left wrist rotation (quaternion w,x,y,z)
[32–34]  delta_t_h0        Controller→wrist offset (translation)
[35–38]  delta_q_h0        Controller→wrist offset (rotation)
[39–77]  same layout for right hand
```

**Virtual Wrist Placement:**
```csharp
anchor.position = controller.position + pose.DeltaPosition;
anchor.rotation = controller.rotation * pose.DeltaRotation;
```

**Feature Normalization:**
```
normalized[i] = (raw[i] - feature_mean[i]) / feature_std[i]
denormalized[i] = normalized[i] * target_std[i] + target_mean[i]
```
