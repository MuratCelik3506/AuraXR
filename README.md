# Intent-Aware Proactive XR Framework

A research-driven pipeline for **Early Action Prediction** (Intent Prediction) in Mixed Reality. This framework uses a Transformer-based model (**IntentFormer**) to anticipate user grasping and manipulation intentions before physical contact occurs, enabling zero-latency proactive XR effects like "Ghost Hands" and pre-loaded physics.

---

## 🚀 Key Features

- **Early Prediction:** Achieves high confidence (65%+) within the first 20-30% of a hand trajectory.
- **Dual Dataset Support:**
    - **H2O:** Dense, 36-class manipulation labels.
    - **HOT3D:** High-fidelity egocentric tracking data from Project Aria and Meta Quest 3.
- **Optimized for Apple Silicon:** Fully utilizes **Metal Performance Shaders (MPS)** for training and **Apple Neural Engine (ANE)** for inference via CoreML.
- **Unity Integration Ready:** Includes CoreML export scripts and guidelines for real-time deployment using Unity Sentis.

---

## 🏗️ Project Structure

```bash
├── data/               # H2O and HOT3D datasets (raw/pre-processed)
├── scripts/
│   └── download_hot3d.py # CLI for HF Hub HOT3D downloading
├── src/
│   ├── data/
│   │   ├── h2o_dataset.py       # H2O skeletal data loading & logic
│   │   ├── hot3d_dataset.py     # HOT3D clip parsing & hand FK reconstruction
│   │   └── combined_dataset.py  # Multi-dataset fusion (Shared Head / Concat)
│   ├── models/
│   │   └── intent_former.py     # IntentFormer architecture (PyTorch)
│   ├── train.py                 # Core training loop (MPS optimized)
│   ├── evaluate.py              # Specialized metrics (Lead Time, TTC, p@ratio)
│   ├── export_coreml.py         # Conversion to ANE-optimized CoreML
│   └── benchmark_mps.py         # Latency profiling on M-series hardware
└── requirements.txt             # Hardware-pinned dependencies
```

---

## 🛠️ Getting Started

### 1. Installation
Ensure you have Python 3.10+ on macOS.

```bash
pip install -r requirements.txt
```

### 2. Prepare HOT3D Data
Login to HuggingFace (requires dataset access) and download a subset for testing:

```bash
huggingface-cli login
python scripts/download_hot3d.py --max_clips 10 --device Aria
```

### 3. Training
Train on the combined dataset using a **Shared Head** (3 classes) to leverage both H2O and HOT3D:

```bash
python -m src.train \
    --dataset combined \
    --fusion shared_head \
    --epochs 60
```

---

## 📊 Evaluation Metrics

The framework evaluates performance based on:
1. **Precision @ 20%:** Accuracy when only the first 20% of the movement is seen.
2. **Lead Time:** How many milliseconds before contact the prediction becomes stable.
3. **Time-To-Contact (TTC):** Estimation of remaining time before interaction.
4. **Ghosting Rate:** How frequently the proactive "Ghost Hand" UI would be triggered.

---

## 📱 CoreML Deployment

To export the model for Unity integration on the Apple Neural Engine:

```bash
python -m src.export_coreml --checkpoint checkpoints/best_model.pt
```

This script performs a **Weight Transplant** to bypass CoreML's native Transformer limitations, ensuring 100% of the model runs on the ANE.

---

## 📜 Acknowledgements
Developed as part of a Master's Thesis on Proactive User Interfaces in Virtual Reality. 
Includes data from the [H2O Dataset](https://h2odataset.github.io/) and [HOT3D Dataset](https://facebookresearch.github.io/hot3d/).
