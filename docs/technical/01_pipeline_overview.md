# 01 — Pipeline Overview

**Last updated:** 2026-06-06

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
│  Script:  src/build_dataset.py                                   │
│  Helpers: src/hot3d_utils.py, src/grip_categories.py            │
│  Input:   data/quest3/ ZIPs                                      │
│  Process: Parse frames → extract 15-dim features →              │
│           hand_confidence filter (< 0.70 discarded) →           │
│           85/15 train/val split by sequence →                    │
│           normalize → grip 10× oversample                        │
│  Output:  data/left/dataset.h5  +  data/right/dataset.h5        │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE C — Model Design                                          │
│  File:    src/model.py  (~294k params)                          │
│  Design:  spatial(8) + object(7) → 5 per-finger heads → 22 j.  │
│           + wrist_rotation_head → 6D palm orientation           │
│           spatial: dir_world(3) + dir_obj_local(3) + dist + spd │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE D — Training                                              │
│  Script:  src/train.py                                           │
│  Input:   dataset.h5 (either hand)                              │
│  Process: AdamW optimizer, compound Huber loss + 6D wrist loss, │
│           cosine LR w/ restarts, auto-selects MPS/CUDA/CPU,     │
│           early stopping (patience=2000), up to 50000 epochs    │
│  Output:  checkpoints/{left,right}/best_model.pt                │
│           checkpoints/{left,right}/model_meta.json              │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE E — Evaluation                                            │
│  Script:  src/evaluate.py                                        │
│  Metrics: Per-joint MAE (degrees), per-phase breakdown          │
│           (pre-shape 10–40cm vs grip <10cm),                    │
│           per-grip-category (Power/Precision/Palmar/Pinch)      │
│  Output:  results/eval_left.json + results/eval_right.json      │
│  Status: ✅ DONE                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE F — ONNX Export                                           │
│  Script:  src/export_onnx.py                                     │
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
│  Process: Build 15-dim feature from controller pos,             │
│           object centroid+rotation, grip type, bbox, velocity.   │
│           Convert Unity (left-hand) → HOT3D (right-hand) coords.│
│  Status: 🔄 IN PROGRESS — wired, frame verified                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE H — Unity Inference                                       │
│  Script:  AuraXRInferenceManager.cs                              │
│  Input:   15-dim feature (from Stage G)                         │
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
│           ablation, user study, paper writing                    │
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
  → train (85%) / val (15%) split by sequence
  → normalize (per-feature mean/std stored in dataset.h5 metadata)

dataset.h5
  /train/features  (N, 15)  float32
  /train/targets   (N, 22)  float32   ← 22 UME joint angles (radians)
  /train/labels    (N,)     string    ← "grip" | "pre_shape"
  /train/distances (N,)     float32

  Left:  train=738,466  val=139,519  (total 877,985)
  Right: train=868,806  val=153,047  (total 1,021,853)

best_model.pt  →  auraxr_{left,right}.onnx  →  Unity Sentis
  Input  spatial_input  [1, 8]
  Input  object_input   [1, 7]
  Output joint_angles   [1, 22]  → denorm → UME[22] → MANO[15] → finger rig
  Output wrist_rot_6d   [1,  6]  → denorm → Gram-Schmidt → Quaternion → hand rotation
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

**Why these boundaries:** 40cm is roughly maximum arm reach — beyond this the hand has not yet started adapting and stays neutral. 10cm is approximately one hand-width from the object surface, where fingertips begin active contact shaping. These boundaries are encoded as `MAX_DISTANCE=0.40` and `GRIP_THRESHOLD=0.10` in `build_dataset.py` and determine which frames are included in training and how they are labeled.

---

## Key Files Quick Reference

| Stage | Python Script | Unity Script |
|-------|--------------|-------------|
| B | `src/build_dataset.py`, `src/hot3d_utils.py`, `src/grip_categories.py` | — |
| C | `src/model.py` | — |
| D | `src/train.py` | — |
| E | `src/evaluate.py`, `src/evaluate_onnx.py` | — |
| E+ | `src/simulate.py`, `src/test_onnx_live.py` | — |
| F | `src/export_onnx.py` | — |
| G | — | `AuraXRFeatureAssembler.cs` |
| H | — | `AuraXRInferenceManager.cs` |
| I | — | `HandSkeletonAnchor.cs`, `HandRigController.cs` |
| J | — | `InteractableObject.cs`, `VirtualHandGrab.cs` |
