# AuraXR Technical Documentation

**Purpose:** Step-by-step documentation for professors and collaborators covering every stage of the pipeline — from raw HOT3D data to live Unity VR inference. Read documents in order (01 → 11). This README is the permanent status tracker; update it every session.

---

## Document Status

| # | File | Topic | Status | Last Updated |
|---|------|-------|--------|-------------|
| 01 | [01_pipeline_overview.md](01_pipeline_overview.md) | Full pipeline A→K, data flow, stage diagram, key files | REVIEWED | 2026-06-06 |
| 02 | [02_dataset_hot3d.md](02_dataset_hot3d.md) | HOT3D dataset structure, ZIPs, UME hand joints, build_dataset.py | REVIEWED | 2026-06-06 |
| 03 | [03_feature_engineering.md](03_feature_engineering.md) | 15-dim features: dir_world, dir_obj_local, approach_speed, normalization | REVIEWED | 2026-06-06 |
| 04 | [04_model_architecture.md](04_model_architecture.md) | Dual-encoder MLP, spatial(8)+object(7), layer shapes, design rationale | REVIEWED | 2026-06-06 |
| 05 | [05_training_evaluation.md](05_training_evaluation.md) | train.py flow, loss function, checkpoints, per-phase evaluation | REVIEWED | 2026-06-06 |
| 06 | [06_onnx_export.md](06_onnx_export.md) | export_onnx.py, model_meta.json schema, Unity Sentis loading | REVIEWED | 2026-06-06 |
| 07 | [07_unity_feature_assembler.md](07_unity_feature_assembler.md) | AuraXRFeatureAssembler: 96-dim ring buffer, Z-negation, frame sampling | DRAFT | 2026-06-03 |
| 08 | [08_unity_inference_manager.md](08_unity_inference_manager.md) | AuraXRInferenceManager: 15-dim assembly, Sentis inference, 6D wrist decode | DRAFT | 2026-06-06 |
| 09 | [09_unity_hand_rendering.md](09_unity_hand_rendering.md) | HandRigController, HandSkeletonAnchor, pivot offset, grab pose blend | DRAFT | 2026-06-05 |
| 10 | [10_unity_interaction_task.md](10_unity_interaction_task.md) | InteractableObject, VirtualHandGrab, ScenarioKitchenTask, SessionDataLogger | DRAFT | 2026-06-03 |
| 11 | [11_known_issues_gaps.md](11_known_issues_gaps.md) | Open bugs, missing pieces, resolved issues log | DRAFT | 2026-06-07 |

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `TODO` | Not started |
| `DRAFT` | Written from code, not yet reviewed with professor |
| `REVIEWED` | Reviewed together, minor edits possible |
| `DONE` | Finalized, no further changes expected |

---

## Current Review Focus

Docs 01–06 are reviewed. The remaining DRAFT documents cover the Unity-side pipeline:

| Priority | Doc | Key question |
|----------|-----|-------------|
| High | [07](07_unity_feature_assembler.md) | Is Z-negation applied correctly for both position and quaternion? |
| High | [08](08_unity_inference_manager.md) | Does `ReadbackAndClone()` block on Quest 3? Is GPUCompute backend safe? |
| High | [09](09_unity_hand_rendering.md) | Are all 15 `fingerJoints` wired in the Inspector for both hands? |
| Medium | [10](10_unity_interaction_task.md) | Are all 33 HOT3D objects in the Unity scene with correct BOP category IDs? |
| Ongoing | [11](11_known_issues_gaps.md) | Track and close open issues — especially C1, C2, C3 (Quest 3 blockers) |

---

## Review Session Log

| Date | Docs Covered | Outcome |
|------|-------------|---------|
| 2026-06-06 | 01–06 | Marked REVIEWED; D1, D3, M1, P1 resolved (see doc 11) |

---

## Iteration Workflow

Each review session:
1. Open this README → find all `DRAFT` documents in the table above
2. Read the doc together with the professor, line by line
3. Fix any errors or gaps found in the doc itself
4. Update the Status and Last Updated columns in this table
5. Any new gap discovered → add a row to [11_known_issues_gaps.md](11_known_issues_gaps.md)
6. When all issues in a document are resolved → mark `DONE`

---

## Project Structure (Quick Reference)

```
AuraXR/
├── src/            Python ML pipeline (build_dataset → train → evaluate → export)
├── data/           HOT3D source ZIPs + processed HDF5 datasets
├── checkpoints/    Trained PyTorch model weights (left + right)
├── onnx/           Exported ONNX models + normalization metadata JSON
├── results/        Evaluation metrics (JSON)
├── docs/           ← You are here
└── Unity/          Reference copies of Unity C# scripts

Unity/AURAXR/Assets/AuraXR/Scripts/
├── core/
│   ├── AuraXRAutoWire.cs           Scene wiring helper
│   ├── AuraXRFeatureAssembler.cs   96-dim ring buffer → 15-dim feature (doc 07)
│   ├── AuraXRInferenceManager.cs   ONNX inference, 6D wrist decode, EMA (doc 08)
│   └── AuraXRMetaLoader.cs         Load normalization stats from model_meta.json
├── hand/
│   ├── HandRigController.cs        Apply MANO angles to finger bones (doc 09)
│   ├── HandSkeletonAnchor.cs       Forward kinematics, wrist anchor (doc 09)
│   ├── HandProximityVisibility.cs  Fade hand mesh near objects (doc 09)
│   └── VirtualHandGrab.cs         Grip trigger → grab logic (doc 10)
├── interaction/
│   ├── InteractableObject.cs       Tags object with HOT3D BOP category ID (doc 10)
│   ├── ProximityDetector.cs        Nearest-object detection per hand (doc 10)
│   └── SnapZone.cs                 Place-and-snap target zones (doc 10)
├── task/
│   ├── ScenarioKitchenTask.cs      Kitchen task sequence and completion logic (doc 10)
│   ├── TaskScoreUI.cs              VR checklist, timer, star rating (doc 10)
│   └── UITaskDisplay.cs            World-space UI panel (doc 10)
└── data/
    ├── AuraXRLogger.cs             Centralized debug logger (doc 10)
    └── SessionDataLogger.cs        CSV session recording (doc 10)
```
