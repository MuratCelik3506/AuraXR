# AuraXR — Real-Time Hand Pose Prediction for VR

AuraXR predicts how a human hand should be shaped when reaching for or holding an object in VR. It uses only the wrist position/orientation (from a VR controller) and the type and size of the nearest object to drive 22 joint angles on a realistic hand mesh in Unity — no cameras, no gloves, no finger tracking hardware.

## Overview

| | |
|---|---|
| **Input** | 15-dim feature: approach direction, object type, bounding box, distance, speed |
| **Output** | 22 UME joint angles + 6D palm orientation |
| **Dataset** | HOT3D (Meta) — ~1.9M frames of real hand-object manipulation |
| **Model** | Dual-encoder MLP, ~1.37M parameters |
| **Runtime** | Unity Sentis (ONNX), targets Quest 3 / Meta XR |

## Pipeline Stages

```
A. Download HOT3D         →  data/quest3/  (~1.5M frames)
B. Build Dataset          →  data/{left,right}/dataset.h5
C. Model Design           →  src/model.py
D. Train                  →  checkpoints/{left,right}/best_model.pt
E. Evaluate               →  results/eval_{left,right}.json
F. Export to ONNX         →  onnx/auraxr_{left,right}.onnx
G. Unity Feature Assembly →  AuraXRFeatureAssembler.cs
H. Unity Inference        →  AuraXRInferenceManager.cs
I. Hand Rendering         →  HandSkeletonAnchor.cs, HandRigController.cs
J. VR Interaction         →  InteractableObject.cs, VirtualHandGrab.cs
K. Device Optimization    →  (in progress)
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
# Apple Silicon: replace onnxruntime with onnxruntime-silicon
```

### 1. Download the HOT3D dataset

```bash
python scripts/download_hot3d.py
```

### 2. Build the HDF5 dataset

```bash
cd src
python build_dataset.py --input_dir ../data/quest3/ --output_dir ../data/
```

### 3. Train

```bash
cd src
python train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/
python train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/
```

Add `--resume` to continue from a saved checkpoint.

### 4. Evaluate

```bash
cd src
python evaluate.py --checkpoint ../checkpoints/right/ --data_dir ../data/right/
python evaluate.py --checkpoint ../checkpoints/left/  --data_dir ../data/left/
```

### 5. Export to ONNX

```bash
cd src
python export_onnx.py
```

Output: `onnx/auraxr_{left,right}.onnx` + `onnx/model_meta_{left,right}.json`

### 6. Verify ONNX (optional)

```bash
cd src
python evaluate_onnx.py --hand right left
python test_onnx_live.py   # simulates 40cm → 2cm approach trajectories
```

## Model Performance

| Hand | Overall MAE | Pre-shape MAE | Grip MAE |
|------|-------------|---------------|----------|
| Left | 12.5° | 12.4° | 18.8° |
| Right | 12.4° | 12.3° | 15.1° |

Per grip category (right hand):

| Power | Precision | Palmar | Pinch |
|-------|-----------|--------|-------|
| 12.2° | 13.7° | 12.8° | 11.8° |

## Architecture

```
spatial_input (B, 8)              object_input (B, 7)
[dir_world(3), dir_obj_local(3),  [grip_one_hot(4), bbox(3)]
 dist(1), approach_speed(1)]
       │                                  │
 FC(8→512) + LayerNorm + ReLU    FC(7→256) + LayerNorm + ReLU
 Dropout(0.125)                  FC(256→256) + ReLU
 FC(512→256) + ReLU                        │
       └──────────── cat(512) ─────────────┘
                         │
                   FC(512→512) + ReLU + Dropout  ×3
                         │
          ┌──────────────┼──────────────────┐
          │              │                  │
   5 × Finger Heads   Wrist Rotation Head  Grip Classifier
   (thumb → pinky)    FC(512→64→6)         (training only)
   FC(512→128→4)      → wrist_rot_6d(B,6)
      → cat(22)
   joint_angles(B,22)
```

Two separate encoders keep spatial geometry and object semantics from mixing before the fusion layer — preventing the model from learning shortcut correlations like "whenever it's a mug, close fingers."

The **grip classifier head** is active only during training (not exported to ONNX). Its purpose is to regularize the shared trunk so it stays grip-category-aware when distance dominates the signal.

## Hand Interaction Phases

| Phase | Distance | Behavior |
|-------|----------|----------|
| Default | > 40 cm | Neutral pose |
| Pre-shape | 10–40 cm | Hand opens to match approaching object |
| Grip | < 10 cm | Fingers curl to final grasp |
| Contact | touching | Pose locks |

## Dataset Statistics

| Split | Left hand | Right hand |
|-------|-----------|------------|
| Train | 738,466 frames | 868,806 frames |
| Val | 139,519 frames | 153,047 frames |
| **Total** | **877,985** | **1,021,853** |

Grip frames (distance < 10 cm) are oversampled 10× during dataset building to balance the heavy pre-shape majority. Normalization statistics are computed before oversampling.

## Repository Structure

```
src/                    Python pipeline scripts
  build_dataset.py      HOT3D → HDF5 feature extraction
  model.py              AuraXRModel architecture
  train.py              Training loop (AdamW, cosine LR, early stopping)
  evaluate.py           Per-joint / per-phase / per-category MAE
  evaluate_onnx.py      ONNX numerical validation
  export_onnx.py        PyTorch → ONNX (opset 14)
  simulate.py           Approach trajectory simulation
  test_onnx_live.py     Live ONNX inference test

Unity/
  Core/                 AuraXRAutoWire.cs, AuraXRFeatureAssembler.cs,
                        AuraXRInferenceManager.cs, AuraXRMetaLoader.cs
  Hand/                 HandSkeletonAnchor.cs, HandRigController.cs,
                        HandProximityVisibility.cs, VirtualHandGrab.cs
  Data/                 AuraXRLogger.cs, SessionDataLogger.cs
  Interaction/          InteractableObject.cs, ProximityDetector.cs

checkpoints/{left,right}/
  best_model.pt         Trained PyTorch weights
  model_meta.json       Normalization stats + architecture config

onnx/
  auraxr_{left,right}.onnx        Exported models
  model_meta_{left,right}.json    Stats for Unity denormalization

data/{left,right}/dataset.h5      Processed training data
results/                          Evaluation JSON outputs
docs/technical/                   11 detailed technical documents
```

## ONNX Input/Output Spec

```
Input  "spatial_input"  [batch, 8]   — normalized
Input  "object_input"   [batch, 7]   — normalized
Output "joint_angles"   [batch, 22]  — normalized; denorm with model_meta.json
Output "wrist_rot_6d"   [batch, 6]   — normalized; denorm → Gram-Schmidt → Quaternion
```

In Unity, batch = 1 (one inference call per hand per frame).

## Technical Documentation

Detailed documentation for each pipeline stage is in [docs/technical/](docs/technical/):

| Doc | Topic |
|-----|-------|
| [01_pipeline_overview.md](docs/technical/01_pipeline_overview.md) | Full pipeline, data flow, stage statuses |
| [02_dataset_hot3d.md](docs/technical/02_dataset_hot3d.md) | HOT3D dataset structure and download |
| [03_feature_engineering.md](docs/technical/03_feature_engineering.md) | 15-dim feature construction |
| [04_model_architecture.md](docs/technical/04_model_architecture.md) | Network design decisions |
| [05_training_evaluation.md](docs/technical/05_training_evaluation.md) | Loss function, hyperparameters, metrics |
| [06_onnx_export.md](docs/technical/06_onnx_export.md) | Export and validation |
| [07_unity_feature_assembler.md](docs/technical/07_unity_feature_assembler.md) | C# feature construction from OVR |
| [08_unity_inference_manager.md](docs/technical/08_unity_inference_manager.md) | Sentis inference, denorm, EMA smoothing |
| [09_unity_hand_rendering.md](docs/technical/09_unity_hand_rendering.md) | Forward kinematics, bone mapping |
| [10_unity_interaction_task.md](docs/technical/10_unity_interaction_task.md) | Grabbing, task checklist, scoring |
| [11_known_issues_gaps.md](docs/technical/11_known_issues_gaps.md) | Open issues and gaps |

## Dependencies

```
torch >= 2.3.0
numpy >= 1.26.0
h5py >= 3.10.0
tqdm >= 4.66.0
onnxruntime >= 1.18.0   (use onnxruntime-silicon on Apple Silicon)
matplotlib >= 3.8.0
```

Unity side requires **Unity Sentis** (Meta XR SDK project targeting Quest 3).
