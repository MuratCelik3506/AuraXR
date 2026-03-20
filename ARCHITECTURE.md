# Architecture & System Design — Intent-Aware XR Framework

This document provides a detailed breakdown of the logic and design decisions behind each core component of the framework.

---

## 📅 1. Data Pipeline Overview

The pipeline's core task is to turn raw skeletal coordinates into **intent signals** that predict the user's final action *before* they complete it.

### 🧩 `src/data/h2o_dataset.py`
- **Logic:** This module handles the H2O (Human-to-Object) dataset, which is a supervised learning source with 36 fine-grained manipulation labels.
- **Problem Solved:** Raw skeletal data is jittery and relative to a camera rig.
- **Transformation:** It applies **Wrist-Relative Normalization** (subtracting the wrist position from all 20 other hand joints), making the motion independent of the global room position. This allows the model to learn the *shape* and *dynamics* of a grasp.
- **Windowing:** It slices a long action (e.g., 2 seconds) into short observation windows (e.g., first 400ms) to train the model's "early prediction" capability.

### 💎 `src/data/hot3d_dataset.py`
- **Logic:** This module parses the HOT3D-Clips (Meta Aria/Quest 3) dataset, which is higher fidelity but lacks action labels.
- **Problem Solved:** HOT3D stores data in `.tar` archives (WebDataset format) and uses UmeTrack angle representations.
- **Transformation:** 
    - It extracts JSON metadata from inside tar archives without unpacking them to save I/O.
    - It implements a **Forward Kinematics (FK)** heuristic to reconstruct 3D joint positions from angle-based UmeTrack data.
    - It infers labels (`pick-up`, `observe`, `put-down`) based on the clip's temporal progression (Start, Middle, End).

### 🔗 `src/data/combined_dataset.py`
- **Logic:** A dataset wrapper that can merge H2O and HOT3D into a single training stream.
- **Fusion Modes:**
    - **`concat`:** Keeps datasets separate in the head (H2O's 36 classes vs. HOT3D's 3 classes).
    - **`shared_head`:** Maps both datasets to the same 3 coarse classes (Pick-up, Observe, Put-down) so the model learns common manipulation features across both.

---

## 🧠 2. Model & Prediction Logic

### 🚀 `src/models/intent_former.py` (IntentFormer)
- **Logic:** A lightweight, temporal self-attention model (Transformer).
- **Problem Solved:** RNNs (like LSTMs) often struggle to weight the very beginning of a sequence (the important bit) as much as the end.
- **Observation-Ratio Embedding:** This is a crucial design detail. We feed the `obs_ratio` (at what percentage of the action we currently are) into the model's positional encoding. This informs the model that a "stable" prediction is expected when the ratio increases.
- **[CLS] Token:** Like BERT, the model uses a special classification token to summarize the entire sequence into one "intent vector."

---

## 📈 3. Training & Evaluation Logic

### 📉 `src/train.py`
- **Logic:** Orchestrates the training process, specifically optimized for Apple Silicon (M-series) using the **`mps`** (Metal Performance Shaders) backend.
- **Early-Prediction Loss:** Implements a specialized loss that penalizes incorrect early predictions more heavily than late ones, forcing the model to be "fast and certain."

### 📏 `src/evaluate.py`
- **Logic:** Beyond simple accuracy, it measures clinical-ready metrics for XR.
- **Lead Time:** Calculates how many frames of "advance notice" the system gets before a collision.
- **Time-To-Contact (TTC):** Estimating the precise millisecond of future contact.

---

## 🍎 4. Deployment Logic

### 🛠️ `src/export_coreml.py`
- **Logic:** Automates the "Weight Transplant" required to move a Transformer model to the **Apple Neural Engine (ANE)**.
- **Problem Solved:** PyTorch's native Transformer layers often fallback to CPU in CoreML because they use "fused" kernels that the ANE doesn't understand.
- **Strategy:** This script disassembles the Transformer into standard matrix multiplications and layer norms, which are 100% lowerable to the ANE, ensuring the Unity game's GPU remains at 90 fps.

### ⚡ `src/benchmark_mps.py`
- **Logic:** Provides a throughput profile (fps vs. batch size) on the M-series GPU/CPU, validating that the model can run at >100 Hz in real-time.
