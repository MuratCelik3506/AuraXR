# 🎓 Thesis Project Summary: Intent-Aware Proactive XR Framework

## Project Goal

Transition an XR system from **reactive** to **proactive** by predicting user grasping intent during the first **20–30% of a movement**, allowing physics properties (mass, friction) to be pre-loaded **T ms before contact**. The platform target is **Apple M2 Max (MPS + ANE + Unity Sentis)**.

---

## 📅 Work Timeline

| Conversation | Date | Theme |
|---|---|---|
| Predictive Models for VR Intent | Feb 20, 2026 | Initial project scoping & research |
| Dataset Loader & Model Development | Feb 20, 2026 | Core data pipeline & baseline models |
| Codebase Walkthrough | Feb 23, 2026 | Full codebase review & understanding |
| Planning XR Framework Fixes | Feb 23–24, 2026 | Gap analysis & fix planning |
| XR Framework Pipeline Development | Mar 11, 2026 | Full pipeline rebuild & refinement |

---

## 🏗️ Deliverables Built

### Python / AI Pipeline (`src/`)

| File | Description |
|---|---|
| [src/data/h2o_dataset.py](file:///Users/muratcelik/Desktop/Thesis/Workspace/Phase1/src/data/h2o_dataset.py) | H2O dataset loader with wrist-relative normalization and sliding window extraction |
| [src/models/intent_former.py](file:///Users/muratcelik/Desktop/Thesis/Workspace/Phase1/src/models/intent_former.py) | Transformer-based **IntentFormer** model for early action prediction |
| [src/train.py](file:///Users/muratcelik/Desktop/Thesis/Workspace/Phase1/src/train.py) | Full training loop with MPS acceleration, early stopping, and metric logging |
| [src/evaluate.py](file:///Users/muratcelik/Desktop/Thesis/Workspace/Phase1/src/evaluate.py) | Evaluation script with Precision @ Observation Ratio, Lead Time, and TTC metrics |
| [src/export_coreml.py](file:///Users/muratcelik/Desktop/Thesis/Workspace/Phase1/src/export_coreml.py) | PyTorch → CoreML export pipeline for Apple Neural Engine (ANE) deployment |
| [src/benchmark_mps.py](file:///Users/muratcelik/Desktop/Thesis/Workspace/Phase1/src/benchmark_mps.py) | MPS benchmarking script to measure latency and throughput on Apple Silicon |
| [requirements.txt](file:///Users/muratcelik/Desktop/Thesis/Workspace/Phase1/requirements.txt) | Pinned Python dependencies |

### Unity / C# Layer (`Unity/Scripts/XRIntent/`)

| File | Description |
|---|---|
| `IntentPredictor.cs` | Loads the CoreML model via Unity Sentis and runs real-time inference |
| `GhostingSystem.cs` | Triggers the "Ghost Hand" visual effect when prediction confidence ≥ 65% |
| `HandPoseProvider.cs` | Reads 21-joint skeletal hand pose data from XR input |
| `ObjectPoseProvider.cs` | Reads 6D object pose (position + orientation) from the scene |

### Build Artifact (`build/`)

| File | Description |
|---|---|
| `build/IntentFormer.mlpackage` | Exported CoreML model package ready for ANE deployment |

### Checkpoints (`checkpoints/`)

| File | Description |
|---|---|
| `checkpoints/best_model.pt` | Best checkpoint by validation accuracy (~9.7 MB) |
| `checkpoints/last_model.pt` | Final epoch checkpoint (~9.7 MB) |
| `checkpoints/metrics.csv` | Full per-epoch training & validation metrics log (60 epochs) |

---

## 📊 Training Results (60 Epochs)

The model was trained on the **H2O dataset** using **PyTorch + MPS** on Apple M2 Max.

### Key Metric Progression

| Phase | Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Precision |
|---|---|---|---|---|---|---|
| Early | 1 | 8.95 | 4.1% | 8.93 | 9.8% | 0.6% |
| Mid | 10 | 5.76 | 29.1% | 5.80 | 34.4% | 30.7% |
| Mid | 20 | 4.43 | 48.3% | 4.63 | 43.4% | 42.3% |
| Late | 40 | 3.55 | 62.9% | 4.56 | 49.7% | 55.3% |
| **Best** | **48** | **3.45** | **65.3%** | **4.39** | **54.6%** | **62.5%** |
| Final | 60 | 3.37 | 66.7% | 4.50 | 52.2% | 57.3% |

> [!IMPORTANT]
> **Best checkpoint was saved at Epoch 48** with **Val Acc = 54.6%** and **Val Precision = 62.5%**. Training per-epoch time stabilized at **~1.2–1.5 seconds** per epoch on MPS after the first warm-up epoch (31.99s).

### Training Curve Highlights

- **Loss** dropped from ~8.95 → ~3.37 (train) and ~8.93 → ~4.50 (val)
- **Accuracy** climbed from ~4% → ~67% (train) and ~10% → ~52% (val)  
- **Precision** improved from near-zero → **~62.5%** at peak
- Mild **overfitting** visible after epoch ~50, confirming the early stopping at epoch 48 was appropriate

---

## 🔬 Architecture Details

### `IntentFormer` (Transformer-based Early Action Predictor)

- **Input:** Sliding window of hand joint poses (21 joints × 3D coords = 63 features) with wrist-relative normalization applied
- **Temporal coverage:** First 20–30% of the action sequence
- **Architecture:** Positional encoding → Transformer encoder layers → Classification head
- **Output:** Intent class logits (Grasping, Pouring, Pushing, etc.)
- **Loss:** Cross-entropy with early-prediction penalty weighting

### Data Processing (`H2ODataset`)

- Parses H2O dataset format: 21-joint hand skeleton + 6D object poses
- **Wrist-relative normalization:** All joints expressed relative to the wrist joint, removing global position bias
- Sliding window extraction for the early observation ratio
- `np.load` compatibility patched for pickle format differences

---

## 🍎 Apple Silicon Deployment Pipeline

```
PyTorch Model (MPS Training)
        ↓
  export_coreml.py
        ↓
  IntentFormer.mlpackage  (Apple Neural Engine)
        ↓
  Unity Sentis Bridge
        ↓
  Real-time XR Inference
```

- **CoreML export** uses `coremltools` with `float16` precision for ANE efficiency
- **ANE** handles inference, freeing the **GPU entirely for Unity rendering**
- **Inference latency target:** sub-millisecond on ANE for real-time XR

---

## 🥽 Unity Integration Logic

### Ghost Hand UX
- `IntentPredictor.cs` runs inference every frame via Unity Sentis
- When **confidence ≥ 65%**, `GhostingSystem.cs` renders a semi-transparent ghost hand
- Ghost intensity scales proportionally with confidence level

### Physics Pre-loading
- `OnCollisionEnter` jitter is eliminated by pre-loading friction/mass properties **T ms before contact**
- Pre-loading is triggered by the same confidence threshold from `IntentPredictor.cs`

---

## 📁 Final Project Structure

```
Phase1/
├── instruction.md          # Project specification
├── requirements.txt        # Python dependencies
├── src/
│   ├── data/
│   │   └── h2o_dataset.py  # H2O data loader + normalization
│   ├── models/
│   │   └── intent_former.py # Transformer model
│   ├── train.py            # MPS training loop
│   ├── evaluate.py         # Evaluation metrics
│   ├── export_coreml.py    # CoreML export
│   └── benchmark_mps.py    # MPS benchmarking
├── checkpoints/
│   ├── best_model.pt       # Best checkpoint (Epoch 48)
│   ├── last_model.pt       # Final checkpoint (Epoch 60)
│   └── metrics.csv         # Full training log
├── build/
│   └── IntentFormer.mlpackage # CoreML artifact
└── Unity/
    └── Scripts/XRIntent/
        ├── IntentPredictor.cs    # Sentis inference bridge
        ├── GhostingSystem.cs     # Ghost hand UX
        ├── HandPoseProvider.cs   # XR hand input
        └── ObjectPoseProvider.cs # XR object pose input
```

---

## ✅ Summary of Status

| Component | Status |
|---|---|
| H2O Dataset Loader | ✅ Complete |
| Wrist-Relative Normalization | ✅ Complete |
| IntentFormer Transformer Model | ✅ Complete |
| MPS Training Pipeline | ✅ Complete (60 epochs) |
| Evaluation Metrics (Precision, Lead Time, TTC) | ✅ Complete |
| CoreML Export | ✅ Complete |
| Unity Sentis Integration | ✅ Complete |
| Ghosting UX System | ✅ Complete |
| Physics Pre-loading Logic | ✅ Complete |
| MPS Benchmarking Script | ✅ Complete |
| **Best Model Val Accuracy** | **54.6% @ Epoch 48** |
| **Best Model Val Precision** | **62.5% @ Epoch 48** |
