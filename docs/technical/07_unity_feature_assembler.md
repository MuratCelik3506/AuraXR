# 07 — Unity Feature Assembler

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source file:** `AuraXRFeatureAssembler.cs`

---

## Purpose

`AuraXRFeatureAssembler` is the Unity-side equivalent of `build_dataset.py`'s `extract_frames()` function. It runs every frame in Unity and builds the same 11-dim feature vector that the Python training code produced.

**If the features don't match training, inference will produce wrong poses.**

---

## What It Does (Summary)

Each frame in `LateUpdate()`:
1. Read left/right controller positions and rotations from OVR
2. Read grip/trigger button values
3. Get nearest interactable object (set externally by `ProximityDetector`)
4. Build a 96-dim raw feature frame and push it into a ring buffer
5. The ring buffer holds the last 16 frames (T=16 window)

Note: The full 96-dim frame includes visual embedding placeholders (dims 32–95, all zeros). The actual ONNX model only uses 11 of these — the inference manager extracts those 11 manually.

---

## 96-Dim Frame Layout

```
Dims  0–2   : left controller position (x, y, -z)    ← Z negated for HOT3D frame
Dims  3–6   : left controller quaternion (w, x, y, -z) ← Z negated
Dim   7     : left grip trigger (0–1)
Dim   8     : left index trigger (0–1)
Dims  9–17  : same for right controller
Dims 18–20  : left nearest object centroid (x, y, -z)
Dims 21–23  : left nearest object bbox half-extents (x, y, z)
Dim  24     : left nearest object category ID (1–33, 0=unknown)
Dims 25–31  : same for right hand's nearest object
Dims 32–95  : visual embedding (64 floats, all zeros — reserved for future camera input)
```

---

## Coordinate Frame Conversion

HOT3D training data used a **right-handed Y-up coordinate system** (Z backward).  
Unity uses a **left-handed Y-up coordinate system** (Z forward).

To convert from Unity to HOT3D frame:
```csharp
// Position: negate Z
f[0] = pos.x;  f[1] = pos.y;  f[2] = -pos.z;

// Quaternion: negate Z imaginary component
f[3] = rot.w;  f[4] = rot.x;  f[5] = rot.y;  f[6] = -rot.z;
```

This conversion is critical. If omitted, the directional features will point in the wrong direction and the model will output incorrect poses.

---

## Ring Buffer

```csharp
private float[][] _buffer;      // [16 frames][96 dims]
private int _head;              // current write position
private bool _full;             // true after first complete loop

// IsReady is false until 16 frames are collected
public bool IsReady => _full;
```

The buffer fills over the first ~0.5 seconds. `AuraXRInferenceManager` checks `IsReady` before running inference.

---

## Frame Sampling Rate

Training data was captured at ~30 fps. Quest 3 renders at 72 Hz. To match the temporal distribution, only every 2nd frame is added to the buffer (`frameSampleRate = 2`), giving ~36 fps effective sampling:

```csharp
if (++_sampleCounter < frameSampleRate) return;
_sampleCounter = 0;
// ... add frame to buffer
```

The 16-frame window then covers approximately `16 / 36 ≈ 0.44 seconds`, close to the training window.

---

## Editor Simulation Mode

When running in the Unity Editor without a connected Quest headset, the script can simulate controller poses that match the HOT3D training distribution:

```csharp
public bool forceHot3dSimulation = false;  // toggle in Inspector
```

When `forceHot3dSimulation = true` (or no OVR controller connected):
- Left hand simulates a tabletop reaching motion with oscillating position/rotation
- Right hand mirrors the left
- A mustard bottle (category 13) is simulated near the left hand
- A mug (category 8) is simulated near the right hand

This lets you test inference in the Editor without wearing a headset.

---

## How the Inference Manager Uses This

`AuraXRInferenceManager` does **not** read from the ring buffer directly. Instead, it recomputes the 11-dim feature itself (from the same controller/object data) and calls the ONNX model directly. The ring buffer exists for potential future temporal models (e.g., RNN/Transformer that uses the full 16-frame window).

Current inference is frame-by-frame (no temporal history). The ring buffer architecture is designed to be extended.

---

## External Dependencies

- **OVRInput** — Meta XR SDK, provides controller input
- **ProximityDetector.cs** — sets `nearestObjectLeft`, `nearestObjectRight`, and their category IDs
- **InteractableObject.cs** — objects must be tagged so `ProximityDetector` can find them

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Is the Z-negation applied consistently for both position and quaternion? (Check lines 137–139 and 164–166)
- [ ] What is `nearestObjectCategoryLeft` set to in a real scene? Trace through ProximityDetector to confirm.
- [ ] Is `frameSampleRate=2` correct for 72 Hz? (72/2 = 36 fps → 16 frames ≈ 0.44s window)
- [ ] The visual embedding (dims 32–95) is always 0. When and how should camera/visual features be added here?
