# 09 — Unity Hand Rendering

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source files:**
- `HandRigController.cs`
- `HandSkeletonAnchor.cs`
- `HandVisibilityController.cs`
- `HandProximityVisibility.cs`

---

## Overview

After `AuraXRInferenceManager` produces `HandPose.ManoJointAngles[15]`, these angles need to drive an actual 3D hand mesh in Unity. The rendering pipeline is:

```
HandPose.ManoJointAngles[15] (radians)
         │
         ▼
HandRigController.cs
  → Applies angles to fingerJoints[] bone transforms
  → Handles grab pose override, joint clamping, sign correction
         │
         ▼
HandSkeletonAnchor.cs
  → Forward kinematics: converts joint angles to 3D bone positions
  → Runs after HandRigController (execution order 1000 vs 500)
         │
         ▼
3D hand mesh rendered in VR
```

---

## HandRigController (`execution order 500`)

Connects the inference output to the skeleton bones.

### Key Inspector Fields

| Field | Description |
|-------|-------------|
| `inferenceManager` | Reads `LeftHand`/`RightHand` pose each frame |
| `isLeftHand` | Which hand this controller drives |
| `fingerJoints[15]` | Bone transforms in MANO order (must be wired in Inspector) |
| `grabSystem` | Reads whether user is currently grabbing an object |
| `grabPoseAnglesRad[15]` | Hard-coded closed-fist pose used during grab |
| `grabBlendSpeed` | How fast to blend into/out of grab pose (deg/s) |
| `jointSignMultipliers[15]` | Per-joint sign (+1 or -1) for rig-specific conventions |
| `smoothing` | Additional per-frame smoothing (0=none, 0.95=heavy) |
| `jointMinAngleDeg` | Lower clamp (default -15°, allows slight hyperextension) |
| `jointMaxAngleDeg` | Upper clamp (default 130°, allows full thumb curl) |

### How It Applies Angles

Each frame in `LateUpdate()`:
```
For each joint i (0–14):
  1. Get angle = pose.ManoJointAngles[i]                      ← radians from inference
  2. Apply smoothing: angle = lerp(prev, angle, 1-smoothing)  ← optional damping
  3. Apply sign: angle *= jointSignMultipliers[i]             ← rig-specific
  4. Clamp: angle = clamp(angle, minDeg, maxDeg) in degrees
  5. Apply to bone: fingerJoints[i].localRotation =
       Quaternion.AngleAxis(angle * Rad2Deg, bone's flexion axis)
```

### Grab Pose Override

When `grabSystem.IsGrabbing`:
- Blend `grabBlendWeight` toward 1.0 at `grabBlendSpeed` per second
- Final angle = lerp(inference_angle, grabPoseAnglesRad[i], grabBlendWeight)

This ensures the hand visually closes completely around the object even if the model's angle is slightly off.

---

## Joint Sign Convention

The `jointSignMultipliers` field handles differences between how the MANO model encodes angles and how the Unity rig's bones are oriented. With the geometric KP model, all values are +1 (flexion is positive). Only change these if switching hand mesh models.

The default closed-fist angles (`grabPoseAnglesRad`) in radians:
```
Thumb:  MCP=0.30, PIP=0.40, DIP=0.30  (thumb tucks under)
Index:  MCP=1.30, PIP=1.20, DIP=0.90  (full curl)
Middle: MCP=1.40, PIP=1.30, DIP=0.90
Ring:   MCP=1.40, PIP=1.30, DIP=0.90
Pinky:  MCP=1.30, PIP=1.20, DIP=0.80
```

These approximate a natural power grip (fist around a cylinder).

---

## Hand Pivot Offset (16.85cm X, 3.51cm Z)

The hand mesh's wrist bone does not align with the Quest 3 controller tracking origin. The offset was measured by logging the delta between controller position and wrist bone position at runtime:

```csharp
// In AuraXRInferenceManager.cs, LogWristOffset():
deltaLocal = Quaternion.Inverse(ctrl.rotation) * (wristBoneWorld - ctrlPos);
// Measured: deltaLocal ≈ (0.1685, 0, 0.0351) in controller-local space
```

Applied every frame:
```csharp
virtualHandLeft.SetPositionAndRotation(
    leftCtrl.position + leftCtrl.rotation * new Vector3(0.1685f, 0f, 0.0351f),
    leftCtrl.rotation);
```

---

## HandVisibilityController & HandProximityVisibility

`HandVisibilityController` manages when the hand mesh is shown:
- Always visible by default
- Can be hidden during certain UI interactions

`HandProximityVisibility` makes the hand fade in/out based on distance to the nearest object. This avoids visual clipping when the hand mesh intersects an object during approach.

---

## HandSkeletonAnchor (`execution order 1000`)

Runs after `HandRigController`. Converts the bone angles set by `HandRigController` into final 3D world positions using forward kinematics. The execution order (1000) ensures it reads the freshly applied angles, not the previous frame's values.

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Are all 15 `fingerJoints` wired in the Inspector? (A common bug: some finger joints are None → that finger doesn't move)
- [ ] What is the finger bone rotation axis? (Open `HandRigController.cs` past line 80 to see how `localRotation` is applied)
- [ ] Does the grab blend feel natural? (Try grabbing a mug — does the hand visually close around it?)
- [ ] Is (16.85cm X, 3.51cm Z) the right offset for both left and right hands, or does it differ?
- [ ] Test `debugBypassModel=true` in `AuraXRInferenceManager` — all joints at 0.5 rad — do all 15 finger bones visibly curl?
