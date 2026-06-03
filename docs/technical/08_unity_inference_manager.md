# 08 — Unity Inference Manager

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source file:** `AuraXRInferenceManager.cs`

---

## Purpose

`AuraXRInferenceManager` is the core runtime component. It:
1. Loads both ONNX models (left + right hand) at startup
2. Every N frames, assembles the 11-dim feature, runs ONNX inference, denormalizes output
3. Applies EMA smoothing over time
4. Maps 22 UME angles → 15 MANO angles
5. Exposes `LeftHand` and `RightHand` as `HandPose` objects for other components to read

---

## Inference Pipeline (One Hand, Per-Frame)

```
Controller Transform + Nearest Object
         │
         ▼
1. COORDINATE CONVERSION
   Unity (left-handed, Z forward)  →  HOT3D (right-handed, Z backward)
   pos_hot3d = (x, y, -z)
   quat_hot3d = (x, y, -z, w)  ← Z imaginary negated

         │
         ▼
2. COMPUTE DIRECTION + DISTANCE
   delta_world = obj_pos_hot3d - wrist_pos_hot3d
   dist        = ||delta_world||
   rel_local   = inverse(wrist_rot_hot3d) * delta_world
   direction   = rel_local / dist          ← unit vector in wrist frame

         │
         ▼
3. LOOK UP GRIP + BBOX
   grip = BopToGrip[categoryId]            ← Power/Precision/Palmar/Pinch
   bbox = BopToBbox[categoryId]            ← [x, y, z] half-extents

         │
         ▼
4. ASSEMBLE RAW 11-DIM FEATURE
   feat = [dir.x, dir.y, dir.z, dist, grip_oh(4), bbox.x, bbox.y, bbox.z]

         │
         ▼
5. NORMALIZE
   feat[i] = (feat[i] - featMean[i]) / featStd[i]

         │
         ▼
6. SPLIT → ONNX INPUTS
   spatialInput = feat[0..3]   shape [1, 4]
   objectInput  = feat[4..10]  shape [1, 7]

         │
         ▼
7. RUN SENTIS WORKER
   worker.SetInput("spatial_input", spatialTensor)
   worker.SetInput("object_input",  objectTensor)
   worker.Schedule()
   outTensor = worker.PeekOutput("joint_angles")
   cpu = outTensor.ReadbackAndClone()    ← GPU → CPU transfer

         │
         ▼
8. DENORMALIZE 22 UME ANGLES
   angles[i] = cpu[0, i] * tgtStd[i] + tgtMean[i]   ← radians

         │
         ▼
9. EMA SMOOTHING (over 22 UME angles)
   smooth[i] = alpha * angles[i] + (1 - alpha) * smooth[i]
   alpha = 0.35  (configured in Inspector)

         │
         ▼
10. MAP UME[22] → MANO[15]
    MANO keeps only flexion joints (not abduction)
    UmeToMano = [0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    pose.ManoJointAngles[m] = smooth[UmeToMano[m]]

         │
         ▼
Output: HandPose (read by HandRigController each frame)
```

---

## UME → MANO Joint Mapping

UME has 4 joints per finger: [abduction, MCP, PIP, DIP]  
MANO uses only flexion (3 per finger): [MCP, PIP, DIP]

The mapping drops the abduction angle (index 0, 4, 8, 12, 16):

```csharp
// UME indices kept for MANO:
int[] UmeToMano = { 0, 2, 3,      // Thumb: CMC-flex, MCP, DIP
                    5, 6, 7,      // Index: MCP, PIP, DIP
                    9, 10, 11,    // Middle: MCP, PIP, DIP
                    13, 14, 15,   // Ring: MCP, PIP, DIP
                    17, 18, 19 }; // Pinky: MCP, PIP, DIP
```

**Note:** For the thumb, index 0 is the CMC/MCP flexion (not abduction), because the thumb's CMC joint dominates thumb pose. Index 1 (thumb abduction) is dropped.

---

## HandPose Output Structure

```csharp
public class HandPose {
    public float[]    ManoJointAngles;   // [15] radians, MANO order
    public float[]    ManoShapeBetas;    // [10] zeros (shape not predicted)
    public Vector3    WristPosition;
    public Quaternion WristRotation;
    public Vector3    DeltaPosition;
    public Quaternion DeltaRotation;
}
```

`HandRigController` and `AuraXRHandRenderer` read `ManoJointAngles` to drive the hand rig.

---

## Hand Pivot Offset

```csharp
public Vector3 handPivotOffset = new Vector3(0.1685f, 0f, 0.0351f);
```

The virtual hand root is placed at:
```csharp
virtualHandLeft.SetPositionAndRotation(
    leftCtrl.position + leftCtrl.rotation * handPivotOffset,
    leftCtrl.rotation);
```

This offset (16.85cm X, 3.51cm Z in controller-local space) was measured by comparing the controller tracking origin to the wrist bone position in the hand mesh. It is calibrated per-device.

---

## EMA Smoothing

Exponential Moving Average smooths the joint angles over time to reduce jitter:
```
alpha = 0.35  → new frame weighted 35%, history weighted 65%
```
Lower alpha = smoother but more lag. 0.35 is a balanced value for 30fps inference.

---

## Debug Modes

**`debugBypassModel = true`:** Sets all 15 MANO joints to 0.5 rad (~28°) — a half-curl test pose. Use this to verify the hand rig is connected before testing the model.

**Debug log file:** Written to `Application.persistentDataPath/Logs/auraxr_debug_*.txt`. Logs every ~60 frames: raw features, normalized features, raw model output, denormalized UME angles, final MANO angles.

---

## Inference Rate

```csharp
public int inferenceEveryNFrames = 2;
```
Inference runs every 2 Unity frames at 72Hz → ~36 Hz inference rate, matching training FPS.

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Trace through the coordinate conversion for a specific frame — does `ToHOT3D()` and `ToHOT3DQuat()` match what `hot3d_utils.py` does?
- [ ] Is the UME→MANO mapping correct? (Line 103: `UmeToMano` array) Compare to HOT3D UmeTrack specification.
- [ ] What does `ReadbackAndClone()` do to performance? Is it blocking? (This copies GPU tensor to CPU — may need async version for Quest 3)
- [ ] Does `handPivotOffset` need per-user calibration or is (16.85cm, 0, 3.51cm) fixed?
- [ ] Check the debug log file after a session — are the MANO angles in a reasonable range (0.0–1.5 rad for most joints)?
