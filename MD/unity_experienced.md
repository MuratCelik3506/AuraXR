# AuraXR — Unity Scenario Experience Overview

> **Purpose:** Describes what the Unity scenario experience is, what is inside it, and how the user goes through it. Used to analyze gaps before the next session.

---

## What Is the Experience?

The user puts on a Meta Quest 3 headset and enters a VR kitchen room. Instead of seeing the default Quest 3 controller models in their hands, they see **animated virtual hands** — driven in real-time by a deep learning model (IntentFormer, 5.4M parameters) running locally on the headset at < 10 ms latency.

The user's task is a short kitchen interaction sequence:
1. Pick up a **bottle** from the table
2. **Pour** the bottle over a cup
3. **Place** the bottle back
4. Pick up the **cup**

The experience lasts roughly 2–5 minutes per condition. It is used in a **3-condition user study** to compare presence and embodiment across hand representations.

---

## What Is Inside the Scene

### Physical Environment

| Object | Description |
|--------|-------------|
| Floor | 4 m × 4 m plane, wood-tone material |
| Walls | 3 cube walls (back, left, right), open front side |
| Table | Wooden table at Z=1.4, surface at Y≈0.77 m — arm-reach height |
| Bottle | Green cylinder on table, left side. Interactable. Category ID 1 |
| Cup | Cream cylinder on table, right side. Interactable. Category ID 3 |
| Plate | Flat disc (decoration only, not interactable) |
| SunLight | Directional, warm white, 55° angle — soft shadows on table |
| TaskCanvas | World-space UI panel (pos 0, 1.8, 1.5) — shows instruction + timer |

### GameObjects & Components on GameManager

| Component | Role |
|-----------|------|
| `AuraXRMetaLoader` | Loads `intentformer_meta.json` — normalization stats (mean/std for 96 inputs + 78 outputs) |
| `AuraXRFeatureAssembler` | Reads controller 6DoF pose + nearest object geometry every frame; maintains T=16 ring buffer → 96-dim feature vector |
| `AuraXRInferenceManager` | Runs ONNX model; decodes 78-dim output into left/right `HandPose`; places virtual wrist anchors via `controller.pos + ΔT` |
| `ProximityDetector` | Finds nearest `InteractableObject` per hand; feeds centroid + bbox + category into FeatureAssembler |
| `SessionDataLogger` | Writes CSV every 100 frames: timestamps, grip/trigger, hand positions, object categories |
| `ScenarioKitchenTask` | State machine: Idle → PickBottle → PourBottle → PlaceBottle → PickCup → Done |
| `HapticFeedbackManager` | Pulses controller vibration when hand is within 12 cm of an interactable object |
| `SoundManager` | Plays audio cue at each state transition |
| `VirtualHandGrab` | OVRInput grip trigger → kinematic follow of object to virtual wrist anchor → throw on release |
| `ConditionManager` | Switches between Condition A, B, C without rebuild — stored in PlayerPrefs |

### Hand System (Condition A — Virtual Hands)

```
LeftHandRig  (wrist anchor positioned by InferenceManager)
 └── MANO hand mesh (SkinnedMeshRenderer)
      └── 15 finger joints (HandRigController drives curl angles from model output)

RightHandRig — same layout
```

- Model predicts wrist position + rotation via `ΔT` offset from controller origin
- 15 MANO pose values (1 DoF per joint = 1 curl angle) drive joint flexion
- Hand material is Standard shader, Rendering Mode = Transparent/Fade
- Hand fades in when controller approaches within ~40 cm of an object (`HandProximityVisibility`)
- Controller model fades out simultaneously (cross-fade)

### Input Feature Vector (96 dims per frame)

```
[0..8]    Left controller:  xyz + wxyz quaternion + grip + trigger  (9 values)
[9..17]   Right controller: same                                     (9 values)
[18..24]  Nearest object LEFT:  centroid(3) + bbox(3) + categoryId(1)  (7 values)
[25..31]  Nearest object RIGHT: same                                 (7 values)
[32..95]  Visual embedding: 64 zeros (placeholder, not activated in v1)
```

### Output Vector (78 dims per frame)

```
Per hand (×2):
  [0..14]  MANO pose θ  — 15 curl angles (radians, converted to degrees × Rad2Deg for joints)
  [15..24] MANO shape β — 10 values (IGNORED in Unity v1; fixed average shape used)
  [25..27] Wrist position (world metres) — used only as auxiliary training target
  [28..31] Wrist rotation (quaternion w,x,y,z)
  [32..34] ΔT translation — controller-to-wrist offset (USED at runtime)
  [35..38] ΔT rotation    — controller-to-wrist rotation (USED at runtime)
```

### Three Conditions

| Condition | What user sees | Inference running? |
|-----------|----------------|-------------------|
| A — Virtual Hands | MANO rig animated by IntentFormer | Yes |
| B — Controller | Default Quest 3 controller models | No (controllers visible) |
| C — Static Pose | MANO rig frozen at T=0 (no motion) | No |

Condition is set via `adb` before each session:
```bash
adb shell am broadcast -a com.aura.setcondition --ei condition 0   # A
adb shell am broadcast -a com.aura.setcondition --ei condition 1   # B
adb shell am broadcast -a com.aura.setcondition --ei condition 2   # C
```

---

## What the User Does — Step by Step

1. **Put on Quest 3** headset, hold controllers
2. **Scene loads** — user stands in kitchen, table is in front at arm reach
3. **Instruction appears** on floating UI panel: "Pick up the bottle"
4. **Timer starts** (autoStart is enabled on ScenarioKitchenTask)
5. **User reaches toward bottle** → hand fades in as controller approaches, bottle highlights yellow (within 15 cm), controller vibrates (haptic at 12 cm)
6. **User squeezes grip trigger** (grip > 0.7) → VirtualHandGrab picks up bottle, bottle follows virtual wrist
7. **User moves bottle over cup** → state machine detects proximity + grip → transitions to PourBottle state → UI updates to "Pour the bottle"
8. **User tilts/holds bottle over cup** → Pour state times out or user triggers next state
9. **UI updates** → "Place the bottle" → user releases grip → bottle drops with physics, rests on table
10. **UI updates** → "Pick up the cup" → user grabs cup same way
11. **Task complete** → Done state → sound plays → timer stops → session end

---

## What Is Currently Broken / Pending

### Inspector Tasks (must be done manually in Unity Editor)

| # | Task | Impact if missing |
|---|------|------------------|
| 1 | Add `InteractableObject` component to Bottle and Cup | Cannot grab either object; hands stay invisible (ProximityVisibility finds nothing) |
| 2 | Set hand materials Rendering Mode → Transparent (or Fade) | `HandProximityVisibility` sets `material.color.a` — has zero effect on Opaque shader |
| 3 | Verify all Transform cross-references (see table below) | Silent NullReferenceExceptions; grab or inference fails |
| 4 | Disable or remove `AuraXRObjectTracker` if present in scene | Conflicts with ProximityDetector — both write to `featureAssembler.nearestObject` and overwrite each other |
| 5 | Tick `autoStart` on `ScenarioKitchenTask` | Task state machine stays in Idle forever; no instructions shown |
| 6 | Enable Emission checkbox on Bottle and Cup materials | GraspIndicator uses `EnableKeyword("_EMISSION")` — does nothing if Emission is disabled |

### Critical Transform Cross-References to Verify

| Script | Field | Must point to |
|--------|-------|--------------|
| `VirtualHandGrab` | `leftHandWrist` | `LeftHandRig` Transform (the wrist anchor) |
| `VirtualHandGrab` | `rightHandWrist` | `RightHandRig` Transform |
| `GraspIndicator` | (auto-finds FeatureAssembler via FindObjectOfType) | GameManager must be in scene |
| `AuraXRHandRenderer` | `wristAnchor` | `LeftHandRig` / `RightHandRig` |
| `HandProximityVisibility` | `leftController` | `OVRLeftControllerVisual` (or `LeftControllerAnchor`) |
| `AuraXRInferenceManager` | `virtualHandLeft` | `LeftHandRig` |
| `AuraXRInferenceManager` | `virtualHandRight` | `RightHandRig` |

### Code Issues Fixed (do NOT redo)

| Script | What was wrong | Status |
|--------|---------------|--------|
| `HandProximityVisibility.cs` | LayerMask default = 0 → no objects found → hands always invisible | ✅ Fixed: direct distance loop, no layer dependency |
| `HandRigController.cs` | Wrist position overwrite fighting InferenceManager; finger axis was Y (yaw) instead of X (flexion); angles in radians not degrees | ✅ Fixed |
| `HandVisibilityController.cs` | LateUpdate forced alpha=1 every frame → overrode fade system | ✅ Fixed: LateUpdate body removed |
| `VirtualHandGrab.cs` | No grab offset (object snapped to wrist origin); null refs; weak throw | ✅ Fixed: grab offset stored at pickup, throwMultiplier → 1.5 |
| `ScenarioKitchenTask.cs` | StartTask() never called automatically | ✅ Fixed: autoStart flag + Start() added |
| `ThumbstickLocomotion.cs` | Player could walk through walls and table | ✅ Fixed 2026-05-13: ClampHeadPosition added |

### What Still Needs to Run (Python side)

- Full 100-epoch training NOT yet launched (most urgent; POC checkpoints from May 1 are too weak)
- After training: evaluate with `12_evaluate.py` (target MPJPE < 50 mm)
- After evaluation: export new ONNX with `13_export_onnx.py`, replace in Unity Assets

---

## Expected Experience Quality (Current State)

If all Inspector tasks above are completed and the POC ONNX is used:

- **Wrist tracking**: works — controller + ΔT offset positions hand near correct location
- **Finger pose**: partial — 15 curl angles drive joints but POC model is undertrained (only a few epochs from May 1); pose accuracy is low
- **Grab**: works — VirtualHandGrab is fixed; object follows wrist correctly
- **Hand visibility fade**: works — if Transparent material is set
- **GraspIndicator highlight**: works — if Emission is enabled on materials
- **Task state machine**: works — if autoStart is ticked and bottle/cup have InteractableObject
- **Haptics**: works
- **UI timer + instructions**: works
- **Session CSV log**: works
- **Condition switching**: works via adb or PlayerPrefs

After full training (100 epochs): finger pose quality should improve significantly. The MPJPE target is < 50 mm on the validation split.

---

## Immediate Next Actions (Ordered)

1. **[Unity Inspector]** Add `InteractableObject` to Bottle and Cup
2. **[Unity Inspector]** Set hand materials to Transparent/Fade mode
3. **[Unity Inspector]** Enable Emission on Bottle and Cup materials
4. **[Unity Inspector]** Tick `autoStart` on ScenarioKitchenTask
5. **[Unity Inspector]** Verify all Transform cross-references (table above)
6. **[Unity Inspector]** Disable `AuraXRObjectTracker` if in scene
7. **[Terminal]** Launch full training: `python3 11_train.py --epochs 100 --batch 64`
8. **[After training]** Evaluate and export new ONNX; replace in Unity
9. **[Quest 3]** Deploy and self-test the full kitchen task end-to-end
