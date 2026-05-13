# AuraXR — Session Log & Action Plan
**Date:** 2026-05-07 (updated 2026-05-13)  
**Session goal:** Fix grab, hold, throw, and hand visibility bugs. Full codebase audit. Identify all missing pieces before full training.

> **Status summary:** Code fixes (Part 1) all applied. Inspector tasks (Part 2) still pending — must be done manually in Unity Editor. Augmentation and MPS crash fix (Part 3 Phase A) done. Full training not yet launched. Wall/table penetration bug additionally fixed on 2026-05-13 (see below).

---

## PART 1 — What Claude Fixed (Code Changes)

### Fix 1 — `HandProximityVisibility.cs`
**Problem:** `Physics.OverlapSphere` was called with a `LayerMask` field.  
If `interactableLayer` is left at 0 in the Inspector (the default), it matches **zero layers** — so the sphere never finds any object, always returns `float.MaxValue`, and the hands stay at alpha=0 (invisible) permanently.

**What was changed:**
- Removed `public LayerMask interactableLayer` field entirely
- Added `private AuraXR.InteractableObject[] _allInteractables` cached in `Start()`
- Replaced `NearestInteractableDistance` with a direct distance loop — no Physics layer config needed
- Added warnings if no `InteractableObject` found, or if hand rig has no `SkinnedMeshRenderer`

---

### Fix 2 — `HandRigController.cs` (Critical — hands at wrong position)
**Problem 1:** `transform.position = pose.WristPosition` and `transform.rotation = pose.WristRotation` were overwriting the correct wrist placement every frame.  
`pose.WristPosition` is in HOT3D world-space (metres from capture rig origin) — completely wrong for Quest 3. The correct formula is already in `AuraXRInferenceManager.ApplyToAnchor`: `controller.position + pose.DeltaPosition`. Having both fight each other caused hands to fly to a wrong position.

**Problem 2:** `fingerJoints[i].localEulerAngles = Vector3.up * angle` rotates around the **Y axis (yaw)**. Finger flexion rotates around the **X axis**. So joint angles from the model had zero visible effect.

**What was changed:**
- Removed the two wrist position/rotation lines entirely
- Changed `Vector3.up * angle` → `Vector3.right * angle * Mathf.Rad2Deg` (correct axis + unit conversion)
- Added a comment explaining why wrist placement must NOT be done here

---

### Fix 3 — `HandVisibilityController.cs` (Conflict)
**Problem:** The `LateUpdate()` body set `_handMaterial.color = new Color(1, 1, 1, 1)` every frame — forcing alpha=1 and overriding `HandProximityVisibility`'s fade system.

**What was changed:**
- Removed the `LateUpdate()` body entirely
- Script is kept as a stub for future use

---

### Fix 4 — `VirtualHandGrab.cs` (Grab + Hold + Throw)
**Problem 1 (No grab):** If `leftHandWrist`/`rightHandWrist` are null or no `InteractableObject` is found in scene, grab silently did nothing — no errors, no feedback.

**Problem 2 (Bad hold):** `FollowWrist` teleported the object to the exact wrist bone origin with no offset. The object popped to a wrong position on grab.

**Problem 3 (Weak throw):** `throwMultiplier = 0.6` produced barely any throw velocity.

**Problem 4 (Compile risk):** `ref` conditional ternary (`ref (a ? ref x : ref y)`) can fail on some Unity/C# versions.

**What was changed:**
- Added `_leftGrabPosOffset`, `_rightGrabPosOffset`, `_leftGrabRotOffset`, `_rightGrabRotOffset` fields
- `Grab()` now stores `wrist.InverseTransformPoint(obj.position)` and `Quaternion.Inverse(wrist.rotation) * obj.rotation`
- `FollowWrist()` now applies offsets via `wrist.TransformPoint(posOffset)` and `wrist.rotation * rotOffset`
- `throwMultiplier` default raised from `0.6` → `1.5`
- Added `Debug.LogError` in `Start()` if wrist refs are null or no interactables found
- Replaced `ref` ternary with simple `if/else`

---

### Fix 5 — `ScenarioKitchenTask.cs` (Task never starts)
**Problem:** `StartTask()` was never called automatically. The task state machine stayed in `Idle` forever — no instructions showed, no state transitions happened.

**What was changed:**
- Added `public bool autoStart = false` field
- Added `void Start()` that calls `StartTask()` if `autoStart` is true
- `StartTask()` now also calls `uiDisplay?.StartTimer()` so the timer begins with the task

---

## PART 2 — Your Parts (Inspector / Unity Editor Tasks)

These cannot be done in code — you must do them manually in the Unity Editor.

### YOUR TASK 1 — Add `InteractableObject` to every grabable object
Every object that should be grabbed (bottle, cup, etc.) needs **two** components:
- `GraspIndicator` — already there (this is what makes it turn yellow)
- **`InteractableObject`** — ADD THIS. It is missing. Without it:
  - `VirtualHandGrab` finds nothing → cannot grab
  - `HandProximityVisibility` finds nothing → hands stay invisible

**Steps:**
1. Select the bottle GameObject in Hierarchy
2. Inspector → Add Component → search `InteractableObject`
3. Set `categoryId` to the HOT3D BOP ID (e.g., bottle = category 3)
4. Repeat for the cup and any other grabable props

---

### YOUR TASK 2 — Remove one of the duplicate proximity detectors
Both of these scripts write to `featureAssembler.nearestObjectLeft/Right` every frame and overwrite each other:
- `ProximityDetector` (uses `InteractableObject`)
- `AuraXRObjectTracker` (uses `AuraXRTrackable`)

**Action:** Find which one is in your scene (check the Hierarchy or a Manager GameObject). Keep `ProximityDetector` (it uses `InteractableObject` which you already have). Disable or remove `AuraXRObjectTracker` from the scene.

---

### YOUR TASK 3 — Set hand materials to Transparent mode
`HandProximityVisibility` controls hand visibility by setting `material.color.a`. This only works if the hand mesh shader supports transparency.

**Steps:**
1. Select the hand mesh material in the Project panel
2. Inspector → Rendering Mode → change from **Opaque** to **Transparent** (or **Fade** for soft edges)
3. Do this for both left and right hand materials

If you are using URP or HDRP: find the `Surface Type` property and set it to **Transparent**.

---

### YOUR TASK 4 — Choose `HandRigController` OR `AuraXRHandRenderer` (not both)
If both scripts are on the same hand rig, they both drive the 15 finger joints every frame in unspecified order.

**Recommendation:** Use `AuraXRHandRenderer` — it has smoothing, configurable axis, and angle scale.

**Steps:**
1. Select `LeftHandRig` in Hierarchy
2. If both `HandRigController` AND `AuraXRHandRenderer` are in the Inspector, disable `HandRigController` (uncheck the checkbox at the top of the component)
3. Repeat for `RightHandRig`

---

### YOUR TASK 5 — Verify Transform references are consistent across scripts
These Inspector fields must all point to the **same Transform** for each hand:

| Script | Field | Should point to |
|--------|-------|----------------|
| `VirtualHandGrab` | `leftHandWrist` | `virtualHandLeft` Transform (the wrist anchor) |
| `GraspIndicator` | `leftHandRig` | same `virtualHandLeft` |
| `AuraXRHandRenderer` | `wristAnchor` | same `virtualHandLeft` |
| `HandProximityVisibility` | `leftController` | OVRLeftControllerAnchor (the physical controller) |
| `AuraXRInferenceManager` | `virtualHandLeft` | the virtual wrist anchor |

If any of these are null or point to the wrong object, the system breaks silently.

---

### YOUR TASK 6 — Enable `autoStart` on `ScenarioKitchenTask`
Now that the fix is in code, you need to tick the checkbox in the Inspector:

**Steps:**
1. Find the GameObject that has `ScenarioKitchenTask` (probably on a Manager)
2. Inspector → ScenarioKitchenTask → check **Auto Start**

Or alternatively, wire the `StartTask()` method to a button or the scene's `Start` event via UnityEvent.

---

## PART 3 — What is Still Missing (Action Items by Phase)

### PHASE A — Training (Week 9)

#### ✅ DONE: Data Augmentation in `hot3d_dataset.py` + `11_train.py`
Three transforms implemented and smoke-tested (500 samples, 3 epochs — clean):
- Controller position noise: ±1 cm uniform offset per window (both hands)
- Beta perturbation: ±0.5 Gaussian on MANO β (both hands)
- Mirror flip: 50% chance swap hand-0 ↔ hand-1 slots in features AND targets
Augmentation is ON by default for `intentformer`, OFF for `gru`/`mlp` baselines.
Pass `--no_aug` to disable for ablation.

#### ✅ DONE: In-place autograd crash fixed in `10_intentformer.py`
`IntentFormer.forward()` had in-place quaternion normalization causing MPS backward version mismatch.
Block removed — redundant since `geodesic_quat_loss` normalizes at training time.
All three models now train without errors.

#### ✅ DECIDED: Visual Branch → Option A (keep 96-dim with zeros)
Dims `[32..95]` stay as zeros. No HDF5 rebuild, no dim change. Document as limitation.

#### ❌ MISSING: Full Training Run
Checkpoints in `data/checkpoints/` are from May 1 (POC run only).
**Launch full training now:**
```bash
cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration
source .venv/bin/activate
python3 11_train.py --epochs 100 --batch 64
```

---

### PHASE A — Evaluation (Week 10)

After full training:

```bash
# Evaluate on val split
python3 12_evaluate.py

# Export new ONNX for Unity
python3 13_export_onnx.py
# or
python3 15_export_onnx_unity.py
```

Run all three baselines for comparison table:
```bash
python3 11_train.py --model gru --epochs 100
python3 11_train.py --model mlp --epochs 100
```

---

### PHASE B — Unity Scene (Week 12–13)

| Item | Status | Who |
|------|--------|-----|
| `ScenarioKitchenTask.cs` | ✅ Fixed | Claude |
| `VirtualHandGrab.cs` | ✅ Fixed | Claude |
| `HandProximityVisibility.cs` | ✅ Fixed | Claude |
| `GraspIndicator.cs` | ✅ OK | — |
| `HapticFeedbackManager.cs` | ✅ OK | — |
| `SessionDataLogger.cs` | ✅ OK | — |
| Add `InteractableObject` to props | ❌ Missing | **You** |
| Fix material transparency | ❌ Missing | **You** |
| Fix duplicate proximity detector | ❌ Missing | **You** |
| PBR hand mesh import | ❌ Week 13 | **You** |
| Environment (walls, floor, lighting) | ❌ Week 13 | **You** |
| Sound assets wired | ❌ Week 13 | **You** |
| 4× MSAA, readable UI text | ❌ Week 13 | **You** |
| Deploy to Quest 3 and self-test | ❌ Week 13 | **You** |

---

### PHASE C — User Study (Week 15)

| Item | Status |
|------|--------|
| Ethics / advisor approval | ❌ Not done — start Week 13 |
| Questionnaire finalized | ✅ In `plan.md` Section 7 |
| Consent form | ❌ Not created |
| Participant recruitment | ❌ Start Week 14 |
| CSV logging (`SessionDataLogger.cs`) | ✅ Implemented |

---

## PART 4 — Quick Reference: Current Health of Every Script

| Script | Status | Notes |
|--------|--------|-------|
| `AuraXRInferenceManager.cs` | ✅ | ONNX runner, wrist placement, interpolation — correct |
| `AuraXRFeatureAssembler.cs` | ✅ | 96-dim ring buffer — correct |
| `AuraXRHandRenderer.cs` | ✅ | Preferred joint driver — use instead of HandRigController |
| `HandRigController.cs` | ✅ Fixed | Wrist overwrite removed, axis fixed |
| `HandVisibilityController.cs` | ✅ Fixed | LateUpdate body removed (was breaking fade) |
| `HandProximityVisibility.cs` | ✅ Fixed | Layer dependency removed |
| `VirtualHandGrab.cs` | ✅ Fixed | Grab offset, better errors, throw fix |
| `GraspIndicator.cs` | ✅ | Yellow highlight — correct |
| `InteractableObject.cs` | ✅ | Must be added to scene objects in Inspector |
| `ProximityDetector.cs` | ✅ | Keep this one; remove AuraXRObjectTracker |
| `AuraXRObjectTracker.cs` | ⚠️ | Conflicts with ProximityDetector — disable in scene |
| `ScenarioKitchenTask.cs` | ✅ Fixed | autoStart added, timer wired |
| `HapticFeedbackManager.cs` | ✅ | OK |
| `ConditionManager.cs` | ✅ | 3-condition switch — OK |
| `SessionDataLogger.cs` | ✅ | CSV logging — OK |
| `UITaskDisplay.cs` | ✅ | Timer + instructions — OK |
| `SoundManager.cs` | ✅ | OK |
| `ThumbstickLocomotion.cs` | ✅ | OK |
| `AuraXRMetaLoader.cs` | ✅ | JSON normalization stats loader — OK |

---

## ADDITIONAL FIX — 2026-05-13

### Fix 6 — `ThumbstickLocomotion.cs` (Wall/Table Penetration)
**Problem:** Player could walk through walls and the table — CharacterController was moving but not being clamped against colliders.

**What was changed:**
- Added `ClampHeadPosition()` method that uses `Physics.CheckSphere` + `Physics.ComputePenetration` to push the camera rig out of overlapping colliders each frame.

---

## PART 5 — Immediate Next Steps (Ordered by Priority)

1. **[You]** Add `InteractableObject` component to bottle and cup in Unity — this unblocks grab entirely
2. **[You]** Set hand materials to Transparent rendering mode — this unblocks hand visibility
3. **[You]** Verify all Transform cross-references in Inspector (Task 5 above)
4. **[✅ Done]** Augmentation added to `hot3d_dataset.py` + `11_train.py`
5. **[✅ Done]** MPS autograd crash fixed in `10_intentformer.py`
6. **[You — NOW]** Launch full training:
   ```bash
   cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration
   source .venv/bin/activate
   python3 11_train.py --epochs 100 --batch 64
   ```
7. **[You]** Disable `AuraXRObjectTracker` if it is in the scene
8. **[You]** After training: run `python3 12_evaluate.py` and report MPJPE result
9. **[You]** After evaluation: run `python3 15_export_onnx_unity.py` and drop new ONNX into Unity
