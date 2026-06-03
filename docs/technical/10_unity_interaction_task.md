# 10 — Unity Interaction & Task System

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source files:**
- `InteractableObject.cs`
- `VirtualHandGrab.cs`
- `ProximityDetector.cs`
- `SnapZone.cs`
- `ScenarioKitchenTask.cs`
- `TaskScoreUI.cs`
- `UITaskDisplay.cs`
- `SessionDataLogger.cs`
- `AuraXRLogger.cs`

---

## Overview

The interaction and task system layers on top of the inference pipeline to create a usable VR experience. It handles:
1. **Object interaction** — which objects can be grabbed, how proximity is detected
2. **Task scenario** — what the user is asked to do (kitchen task)
3. **UI feedback** — live task checklist, timer, star rating
4. **Logging** — session data to CSV for analysis

---

## Object Interaction

### `InteractableObject.cs`
Tags a GameObject as interactable. Stores:
- **HOT3D BOP category ID** (1–33) — used by `AuraXRInferenceManager` to look up grip type and bbox
- **Object name** for logging

All objects in the scene that the user might reach for must have this component.

### `ProximityDetector.cs`
Each hand has a `ProximityDetector` that runs every frame to find the nearest `InteractableObject` within range. When it finds one:
```csharp
featureAssembler.nearestObjectLeft  = nearestTransform;
featureAssembler.nearestObjectCategoryLeft = nearestObject.categoryId;
```
This feeds into both `AuraXRFeatureAssembler` (for the ring buffer) and `AuraXRInferenceManager` (for inference).

### `VirtualHandGrab.cs`
Handles the physical grab interaction:
- When grip trigger > threshold → attempt grab of nearest object
- On grab: object follows hand transform, physics disabled
- On release: physics re-enabled, object thrown with hand velocity
- Sets `IsGrabbing = true` — read by `HandRigController` to override pose with closed-fist angles

### `SnapZone.cs`
Defines a position where an object should snap to when placed nearby. Used in the kitchen task for placing objects on the table or in specific locations. When a grabbed object enters a snap zone and is released, it snaps to the zone's position/rotation.

---

## Task Scenario: `ScenarioKitchenTask.cs`

Defines the kitchen manipulation task used in experiments. A typical task sequence:
1. Pick up [object] from [location]
2. Carry it to [target location]
3. Place it in the snap zone

The script:
- Defines a list of subtasks with completion conditions
- Monitors object positions and grab events
- Fires events when subtasks complete
- Reports task completion to `TaskScoreUI`

---

## Task UI: `TaskScoreUI.cs`

Live task display shown to the user in VR:

| Feature | Description |
|---------|-------------|
| Checklist | Shows each subtask with a checkmark when completed |
| Timer | Counts up from task start, stops on completion |
| Star rating | 1–3 stars based on completion time and errors |
| Feedback text | "Good!", "Well done!", etc. |

The UI is a world-space canvas in VR. It stays visible without requiring the user to look at a controller menu.

---

## Logging: `SessionDataLogger.cs`

Writes a CSV file per session to `Application.persistentDataPath/Logs/`.

### CSV columns:
```
timestamp          — seconds since session start
left_grip          — left hand grip trigger (0–1)
left_trigger       — left hand index trigger (0–1)
right_grip         — right hand grip trigger (0–1)
right_trigger      — right hand index trigger (0–1)
left_hand_x/y/z    — left wrist position (Unity world space)
right_hand_x/y/z   — right wrist position (Unity world space)
left_obj_category  — HOT3D BOP ID of nearest left object (0=none)
right_obj_category — HOT3D BOP ID of nearest right object (0=none)
```

**Flush interval:** Every 100 frames (~1.4s at 72Hz), ensuring data is not lost if the app crashes.

### `AuraXRLogger.cs`
Centralized debug logging utility. Wraps `Debug.Log` with category tags and severity levels, making it easier to filter logs in the Unity console.

---

## Component Wiring Diagram

```
[GameManager GameObject]
  ├── AuraXRInferenceManager
  │     reads ← AuraXRFeatureAssembler
  │     reads ← nearestObjectLeft/Right (set by ProximityDetector)
  │     exposes → LeftHand, RightHand (HandPose)
  │
  ├── AuraXRFeatureAssembler
  │     reads ← leftControllerTransform, rightControllerTransform
  │     written ← ProximityDetector (nearestObject, categoryId)
  │
  ├── VirtualHandGrab
  │     reads ← OVR grip input
  │     sets → IsGrabbing (read by HandRigController)
  │
  └── SessionDataLogger
        reads ← AuraXRInferenceManager, AuraXRFeatureAssembler

[LeftHandRig]
  └── HandRigController (isLeftHand=true)
        reads ← AuraXRInferenceManager.LeftHand
        drives → fingerJoints[15]
        reads ← VirtualHandGrab.IsGrabbing

[RightHandRig]
  └── HandRigController (isLeftHand=false)
        reads ← AuraXRInferenceManager.RightHand
        drives → fingerJoints[15]

[Each interactable object]
  ├── InteractableObject (categoryId = HOT3D BOP ID)
  └── ProximityDetector (range=0.3m, updates featureAssembler)

[UI Canvas]
  └── TaskScoreUI
        reads ← ScenarioKitchenTask (task events)
```

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Are all 33 HOT3D objects present in the Unity scene? Each needs `InteractableObject` with the correct BOP category ID.
- [ ] Is `ProximityDetector` range calibrated? (Default range — what radius triggers nearest-object detection?)
- [ ] Does `VirtualHandGrab` use the correct grab threshold? Test with grip=0.7 threshold.
- [ ] Is the CSV being written to on Quest 3? (Check `Application.persistentDataPath` on Android — it goes to internal storage)
- [ ] Open a session CSV and plot `left_grip` vs `left_obj_category` — do grip values rise when an object is nearby?
