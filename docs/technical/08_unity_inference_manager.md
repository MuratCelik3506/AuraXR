# 08 — Unity Inference Manager

**Status:** DRAFT | **Last updated:** 2026-06-06

**Source file:** `Unity/core/AuraXRInferenceManager.cs`

---

## Purpose

`AuraXRInferenceManager` is the core runtime component. It:
1. Loads both ONNX models (left + right hand) at startup
2. Every frame: positions the virtual hand at the controller anchor using the **model-predicted rotation**
3. Every N frames: assembles the 15-dim feature, runs ONNX inference, produces two outputs
4. Output 1 — **joint_angles (22)**: finger pose → UME→MANO mapping → drives finger rig
5. Output 2 — **wrist_rot_6d (6)**: palm orientation → Gram-Schmidt decode → drives hand placement rotation

---

## Inference Pipeline (One Hand, Per Inference Frame)

```
Controller Transform + Nearest Object Transform
         │
         ▼
1. POSITIONS → HOT3D FRAME
   wristPosH = ToHOT3D(ctrl.position)   = (x, y, -z)
   objPosH   = ToHOT3D(obj.position)    = (x, y, -z)
   relWorld  = objPosH - wristPosH
   dist      = ||relWorld||
   dirWorld  = relWorld / dist           ← world-frame unit vector (HOT3D)

         │
         ▼
2. OBJECT-LOCAL DIRECTION
   objRotH    = ToHOT3DQuat(nearestObj.rotation)
   dirObjLoc  = Inverse(objRotH) * dirWorld
                                          ← which face of object is approached

         │
         ▼
3. APPROACH SPEED
   velWorld     = (wristPosH - prevWristPosH) / dt
   approachSpeed = dot(velWorld, dirWorld)   ← positive = moving toward object

         │
         ▼
4. GRIP + BBOX
   grip = BopToGrip[categoryId]            ← Power/Precision/Palmar/Pinch
   bbox = BopToBbox[categoryId]            ← [x, y, z] half-extents (metres)

         │
         ▼
5. ASSEMBLE RAW 15-DIM FEATURE
   feat = [dirWorld(3), dirObjLoc(3), dist(1), approachSpeed(1), grip_oh(4), bbox(3)]

         │
         ▼
6. NORMALIZE
   feat[i] = (feat[i] - featMean[i]) / featStd[i]   (15 entries from model_meta.json)

         │
         ▼
7. SPLIT → ONNX INPUTS
   spatialInput = feat[0..7]   shape [1, 8]
   objectInput  = feat[8..14]  shape [1, 7]

         │
         ▼
8. RUN SENTIS WORKER  (two outputs)
   worker.SetInput("spatial_input", spatialTensor)
   worker.SetInput("object_input",  objectTensor)
   worker.Schedule()
   anglesTensor = PeekOutput("joint_angles")   shape [1, 22]
   rotTensor    = PeekOutput("wrist_rot_6d")   shape [1,  6]

         │
         ▼
9. DENORMALIZE 22 UME ANGLES
   angles[i] = angleCpu[0,i] * tgtStd[i] + tgtMean[i]   ← radians

         │
         ▼
10. EMA SMOOTHING
    smooth[i] = alpha * angles[i] + (1 - alpha) * smooth[i]
    alpha = 0.35 (Inspector)

         │
         ▼
11. MAP UME[22] → MANO[15]
    UmeToMano = [0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    → HandPose.ManoJointAngles[15]

         │
         ▼
12. DECODE WRIST ROTATION (6D → Quaternion)
    Denormalize: rot6d[i] = rotCpu[0,i] * rotStd[i] + rotMean[i]

    Gram-Schmidt orthogonalization:
      col0 = normalize(rot6d[0..2])
      col1 = normalize(rot6d[3..5] - dot(rot6d[3..5], col0) * col0)
      col2 = cross(col0, col1)

    Rotation matrix → quaternion (Unity Matrix4x4.rotation):
      q_rel = mat.rotation   ← relative rotation in Unity frame

    Canonical rotation (local Z = direction toward object in Unity):
      dirWorldUnity = (dirWorld.x, dirWorld.y, -dirWorld.z)  ← negate Z for Unity
      qCanonical    = Quaternion.LookRotation(dirWorldUnity, Vector3.up)

    Predicted wrist rotation:
      predictedRot = qCanonical * q_rel   ← Unity world-frame quaternion

         │
         ▼
Output:
  HandPose.ManoJointAngles[15]    → HandRigController drives finger bones
  _predictedRot{Right,Left}       → Update() places virtual hand with this rotation
```

---

## Hand Placement (Every Frame)

```csharp
// Every frame — uses cached _predictedRot from last inference
virtualHandRight.SetPositionAndRotation(
    rightCtrl.position + _predictedRotRight * handPivotOffset,
    _predictedRotRight);

// Fallback when no nearest object
if (nearestR == null)
    _predictedRotRight = rightCtrl.rotation;
```

The position always comes from the controller. The rotation comes from the model prediction, cached between inference frames. When no object is in range, falls back to controller rotation.

---

## Why 6D Continuous Rotation

The 6D representation (Zhou et al., 2019) stores the first two columns of the rotation matrix. Unlike quaternions, it is **continuously differentiable** everywhere — no discontinuity at antipodal quaternions — making it the best choice for neural network regression. The third column is recovered at decode time via Gram-Schmidt, so the network only needs to predict 6 values.

---

## Why World-Frame Direction (Not Wrist-Local)

Using wrist-local direction requires a wrist quaternion consistent between HOT3D's UME tracker and Quest 3's OVR controller — two different systems with different conventions. World-frame direction avoids this: only positions are needed, no rotation mismatch possible.

The `wrist_rot_6d` prediction is expressed **relative to the approach direction** (via `q_canonical = LookRotation(dir_world)`), so the model learns "when approaching a mug from the left, tilt palm X degrees" without needing absolute world orientation.

---

## UME → MANO Joint Mapping

UME: 4 joints per finger [abduction, MCP, PIP, DIP]
MANO: 3 per finger [MCP, PIP, DIP] — abduction dropped

```csharp
int[] UmeToMano = { 0, 2, 3,      // Thumb: CMC-flex, MCP, DIP
                    5, 6, 7,      // Index: MCP, PIP, DIP
                    9, 10, 11,    // Middle: MCP, PIP, DIP
                    13, 14, 15,   // Ring: MCP, PIP, DIP
                    17, 18, 19 }; // Pinky: MCP, PIP, DIP
```

---

## Hand Pivot Offset

`handPivotOffset = (0.1685, 0, 0.0351)` — shifts the wrist 16.85 cm in X and 3.51 cm in Z from the Quest controller tracking origin. Tune in Inspector during Play mode.

---

## EMA Smoothing

```
alpha = 0.35  → new frame 35%, history 65%
```
Lower alpha = smoother but more lag. Applied to UME joint angles only; wrist rotation is not EMA-smoothed (the 6D decode is stable frame-to-frame).

---

## Debug Modes

**`debugBypassModel = true`:** All 15 MANO joints set to 0.5 rad (~28°), wrist rotation = controller rotation. Verifies rig wiring before testing the model.

**Debug log file:** `Application.persistentDataPath/Logs/auraxr_debug_*.txt`. Logs every ~60 frames: raw features, model outputs, denormalized UME angles, MANO angles, predicted wrist euler angles.

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Verify `model_meta.json` has **15** feature entries AND **6** wrist_rot entries before deploying to Unity
- [ ] Check `ToHOT3D()` / `ToHOT3DQuat()` are used consistently (Z negation both places)
- [ ] After new training, check debug log — MANO angles 0–1.5 rad range, wrist euler angles reasonable (~0–90°)
- [ ] Is `ReadbackAndClone()` blocking on Quest 3? May need async version for performance
- [ ] Test with `debugBypassModel=true` first to confirm rig wiring before enabling model
