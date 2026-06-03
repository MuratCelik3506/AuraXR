# 01 — Pipeline Overview

**Status:** DRAFT | **Last updated:** 2026-06-03

This document gives the big picture. Read this first before any other document.

---

## What Is AuraXR?

AuraXR predicts how a human hand should be shaped when reaching for or holding an object in VR. Instead of using cameras or gloves, it uses only:
- The position and orientation of the VR controller (wrist proxy)
- The type and size of the nearest object

From these inputs a small neural network predicts 22 joint angles that drive a realistic hand mesh in Unity.

---

## The Full Pipeline (Stage A → K)

```
┌──────────────────────────────────────────────────────────────────┐
│  STAGE A — Data Download                                         │
│  Source: HOT3D dataset (Quest 3 + Aria recordings)              │
│  Output: data/quest3/ ZIPs (~1.5M frames of manipulation)       │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE B — Dataset Building                                      │
│  Script:  hot3d_exploration/build_dataset.py                     │
│  Input:   data/quest3/ ZIPs                                      │
│  Process: Parse frames → transform to wrist frame →             │
│           extract 11-dim features → approach augmentation →     │
│           70/15 train/val split → normalize                      │
│  Output:  data/left/dataset.h5  +  data/right/dataset.h5        │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE C — Model Design                                          │
│  Files:   hot3d_exploration/model.py   (V1, ~54k params)        │
│           hot3d_exploration/model_v2.py (V2, ~210k params)      │
│  V1: spatial(4) + object(7) → 22 joints                        │
│  V2: spatial(8, adds wrist quaternion) + per-finger heads       │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE D — Training                                              │
│  Script:  hot3d_exploration/train.py                             │
│  Input:   dataset.h5 (either hand)                              │
│  Process: Adam optimizer, 500 epochs, weighted Huber loss,      │
│           cosine LR schedule, auto-selects MPS/CUDA/CPU         │
│  Output:  checkpoints/{left,right}/best_model.pt                │
│           checkpoints/{left,right}/model_meta.json              │
│  Status: ✅ DONE (v5 and v6 variants also trained)               │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE E — Evaluation                                            │
│  Script:  hot3d_exploration/evaluate.py                          │
│  Metrics: Per-joint MAE (degrees), per-phase breakdown          │
│           (pre-shape 10–40cm vs grip <10cm),                    │
│           per-grip-category (Power/Precision/Palmar/Pinch)      │
│  Output:  results/eval_left.json + results/eval_right.json      │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE F — ONNX Export                                           │
│  Script:  hot3d_exploration/export_onnx.py                       │
│  Input:   checkpoints/{left,right}/best_model.pt                │
│  Process: PyTorch → ONNX (opset 14), bitwise validation,        │
│           copy model_meta.json for Unity                         │
│  Output:  onnx/auraxr_left.onnx + onnx/auraxr_right.onnx       │
│           onnx/model_meta_left.json + model_meta_right.json     │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE G — Unity Feature Assembly                                │
│  Script:  AuraXRFeatureAssembler.cs                              │
│  Input:   OVR controller transforms + nearest InteractableObject│
│  Process: Build 11-dim feature from controller pos/rot,         │
│           object centroid, grip type, bbox.                      │
│           Convert Unity (left-hand) → HOT3D (right-hand) coords.│
│  Status: 🔄 IN PROGRESS — wired, frame verified                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE H — Unity Inference                                       │
│  Script:  AuraXRInferenceManager.cs                              │
│  Input:   11-dim feature (from Stage G)                         │
│  Process: Normalize → feed to Unity Sentis ONNX worker →        │
│           denormalize 22 UME angles → EMA smooth →             │
│           map UME[22] → MANO[15]                                │
│  Output:  HandPose.ManoJointAngles[15] per hand                 │
│  Status: 🔄 IN PROGRESS — runs, pivot offset tuned              │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE I — Hand Rendering                                        │
│  Scripts: HandSkeletonAnchor.cs, HandRigController.cs           │
│  Input:   HandPose.ManoJointAngles[15]                          │
│  Process: Forward kinematics → drive bone rotations on rig      │
│  Output:  Animated hand mesh visible in VR                      │
│  Status: 🔄 IN PROGRESS — rendering works, FK tuning needed     │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE J — VR Interaction & Task                                 │
│  Scripts: InteractableObject.cs, VirtualHandGrab.cs,            │
│           ScenarioKitchenTask.cs, TaskScoreUI.cs                │
│  Process: Object grabbing, task checklist, timer, star rating   │
│  Status: ✅ PARTIAL                                              │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE K — Device Optimization & Paper                          │
│  Tasks:   Quest 3 device testing, quantization, user study,     │
│           ablation (V1 vs V2 vs V5 vs V6), paper writing        │
│  Status: ⏳ WAITING                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Numbers at Each Stage

```
HOT3D ZIPs
  → ~1.5M raw frames (Quest 3 + Aria)
  → filter: distance < 40cm, hand visible
  → ~877,985 frames (left hand)  +  ~1,021,853 frames (right hand)
  → + approach augmentation (6 synthetic distances per grip frame)
  → train (70%) / val (15%) split by sequence
  → normalize (per-feature mean/std stored in dataset.h5 metadata)

dataset.h5
  /train/features  (N, 11)  float32
  /train/targets   (N, 22)  float32   ← 22 UME joint angles (radians)
  /train/labels    (N,)     string    ← "grip" | "pre_shape" | "approach"
  /train/distances (N,)     float32

  Left:  train=738,466  val=139,519  (total 877,985)
  Right: train=868,806  val=153,047  (total 1,021,853)

best_model.pt  →  auraxr_{left,right}.onnx  →  Unity Sentis
  (Deployed model: V1 architecture, spatial_input_dim=4)
  Input  spatial_input  [1, 4]
  Input  object_input   [1, 7]
  Output joint_angles   [1, 22]
  → denormalize → 22 UME angles (radians)
  → map UME[22] → MANO[15]  (drop abduction, keep flexion only)
  → HandRigController drives hand mesh
```

---

## Hand Interaction Phases

The model was trained with 4 distinct phases in mind:

| Phase | Distance | Hand behavior |
|-------|----------|---------------|
| Default | > 40 cm | Fixed neutral pose |
| Pre-shape | 10–40 cm | Hand opens to match approaching object |
| Grip | < 10 cm | Fingers curl to final grasp |
| Contact | touching | Pose locks (held object) |

---

## Key Files Quick Reference

| Stage | Python Script | Unity Script |
|-------|--------------|-------------|
| B | `hot3d_exploration/build_dataset.py` | — |
| C | `hot3d_exploration/model.py` + `model_v2.py` | — |
| D | `hot3d_exploration/train.py` | — |
| E | `hot3d_exploration/evaluate.py` | — |
| F | `hot3d_exploration/export_onnx.py` | — |
| G | — | `AuraXRFeatureAssembler.cs` |
| H | — | `AuraXRInferenceManager.cs` |
| I | — | `HandSkeletonAnchor.cs`, `HandRigController.cs` |
| J | — | `InteractableObject.cs`, `VirtualHandGrab.cs` |

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Does the stage order make sense?
- [ ] Are the status labels (✅/🔄/⏳) still accurate?
- [ ] Are the frame counts for left/right filled in? (XXX placeholders)
- [ ] Is the MANO mapping explanation clear enough?
