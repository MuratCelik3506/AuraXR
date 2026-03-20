# Project Specification: Intent-Aware Proactive XR Framework

## 1. Context & Objective

The goal is to transition an XR (Extended Reality) system from a **reactive** state to a **proactive** state. Instead of waiting for a collision, the system must predict user intent during the first **20-30%** of a movement.

* **Primary Goal:** Pre-load physical properties (mass, friction) $T$ milliseconds before contact.
* **Hardware Target:** Apple M2 Max (Unified Memory Architecture).
* **Core Library/API:** Metal Performance Shaders (MPS), CoreML, and Unity Sentis.

## 2. Data Architecture (H2O Dataset)

The code must handle data structures compatible with the **H2O (Hands-to-Objects)** framework:

* **Input Features:** - **3D Hand Poses:** 21 skeletal joints.
* **6D Object Poses:** 3D position + 3D orientation.


* **Data Pre-processing:**
* Prioritize **Skeletal Data** over RGB frames to reduce I/O bottlenecks on M2 Max.
* Implement **Wrist-Relative Normalization** to ensure motion tracking is independent of the user's global position.



## 3. Model Requirements (Transformer-Based)

The prediction engine should be a lightweight Transformer designed for **Early Action Prediction**:

* **Temporal Analysis:** Process a sliding window of the initial 20-30% of motion.
* **Output:** Classification of intent (e.g., *Grasping, Pouring, Pushing*).
* **Performance Metrics:** - **Lead Time:** Gap between prediction and action completion.
* **TTC (Time-to-Contact):** Estimated time remaining before contact.
* **Precision Calculation:** $Precision = \frac{True\ Positives}{True\ Positives + False\ Positives}$.



## 4. Technical Implementation Pipeline

1. **Training:** Use PyTorch with **MPS (Metal Performance Shaders)** for GPU acceleration on Apple Silicon.
2. **Conversion:** Export the trained model to **CoreML** to utilize the **Apple Neural Engine (ANE)**, leaving the GPU free for Unity rendering.
3. **Integration:** Use **Unity Sentis** as the bridge to run the CoreML model in real-time with millisecond-level latency.

## 5. Interaction Logic (Unity/C#)

* **The "Ghosting" Effect:** - Trigger a visual "Ghost Hand" (semi-transparent) when prediction confidence exceeds **65%**.
* **Physics Pre-loading:**
* Finalize physics engine calculations (friction/mass interaction) milliseconds before the `OnCollisionEnter` event to eliminate "jitter."



## 6. Coding Instructions for Antigravity

> "When generating code, prioritize low-latency execution and memory efficiency. Focus on the mathematical transformation of skeletal joints and the asynchronous bridge between the CoreML inference engine and the Unity Physics loop."

---

### How to use this:

1. Save the content above as `Instruction.md`.
2. Upload this file along with your folder data to Antigravity.
3. Use a prompt like: *"Based on the attached Instruction.md and my folder data, generate the C# scripts for the Unity Sentis integration and the Python script for the CoreML conversion."*