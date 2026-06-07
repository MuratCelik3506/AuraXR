# AuraXR Pipeline Documentation

**Purpose:** Step-by-step documentation for professors and collaborators to inspect every part of the pipeline together — from raw HOT3D data to live Unity VR inference.

**How to use:** Read documents in order (01 → 11). Each document stands alone and links to the actual source files. This README is the permanent status tracker — update it every session.

---

## Document Status

| # | File | Topic | Status | Last Updated |
|---|------|-------|--------|-------------|
| 1 | [01_pipeline_overview.md](01_pipeline_overview.md) | Full pipeline diagram, stages A→K, data flow, key files | REVIEWED | 2026-06-06 |
| 2 | [02_dataset_hot3d.md](02_dataset_hot3d.md) | HOT3D dataset structure, ZIPs, UME hand joints, build_dataset.py | REVIEWED | 2026-06-06 |
| 3 | [03_feature_engineering.md](03_feature_engineering.md) | 15-dim features, dir_world + dir_obj_local + approach_speed, normalization | REVIEWED | 2026-06-06 |
| 4 | [04_model_architecture.md](04_model_architecture.md) | AuraXRModel architecture, spatial(8)+object(7), layer shapes | REVIEWED | 2026-06-06 |
| 5 | [05_training_evaluation.md](05_training_evaluation.md) | train.py flow, loss function, checkpoints, per-phase evaluation | REVIEWED | 2026-06-06 |
| 6 | [06_onnx_export.md](06_onnx_export.md) | export_onnx.py, model_meta.json schema, Unity Sentis loading | REVIEWED | 2026-06-06 |
| 7 | [07_unity_feature_assembler.md](07_unity_feature_assembler.md) | AuraXRFeatureAssembler: ring buffer, controller data, coordinate frames | DRAFT | 2026-06-03 |
| 8 | [08_unity_inference_manager.md](08_unity_inference_manager.md) | AuraXRInferenceManager: inference loop, denormalization, UME→MANO mapping | DRAFT | 2026-06-03 |
| 9 | [09_unity_hand_rendering.md](09_unity_hand_rendering.md) | HandSkeletonAnchor FK, HandRigController, pivot offset, visibility | DRAFT | 2026-06-03 |
| 10 | [10_unity_interaction_task.md](10_unity_interaction_task.md) | InteractableObject, VirtualHandGrab, TaskScoreUI, SessionDataLogger | DRAFT | 2026-06-03 |
| 11 | [11_known_issues_gaps.md](11_known_issues_gaps.md) | Missing pieces, open bugs, what to test next | DRAFT | 2026-06-03 |

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `TODO` | Not started |
| `DRAFT` | Written from code, not yet reviewed with professor |
| `REVIEWED` | Reviewed together, minor edits possible |
| `DONE` | Finalized, no further changes expected |

---

## Iteration Workflow (Recycle Approach)

Each review session:
1. Open this README — find all `DRAFT` documents
2. Read the doc together with the professor, line by line
3. Fix any errors or gaps found
4. Update status to `REVIEWED`
5. Any missing piece discovered → add to [11_known_issues_gaps.md](11_known_issues_gaps.md)
6. When all issues in that doc are resolved → mark `DONE`

---

## Project Structure (Quick Reference)

```
V3/
├── src/           Python ML pipeline (build_dataset → train → evaluate → export)
├── data/          HOT3D source ZIPs + processed HDF5 datasets
├── checkpoints/   Trained PyTorch model weights
├── onnx/          Exported ONNX models + normalization metadata
├── results/       Evaluation metrics (JSON)
├── docs/          ← You are here
└── Unity/         Reference copies of Unity C# scripts

Unity/AURAXR/Assets/AuraXR/Scripts/
├── AuraXRFeatureAssembler.cs   Build 15-dim feature vector each frame
├── AuraXRInferenceManager.cs   Run ONNX inference, output 22 UME → 15 MANO joint angles
├── AuraXRMetaLoader.cs         Load normalization stats from JSON
├── HandSkeletonAnchor.cs       Forward kinematics: joints → bone positions
├── HandRigController.cs        Drive hand animation rig
└── SessionDataLogger.cs        Log poses and events to CSV
```
