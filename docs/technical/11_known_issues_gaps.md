# 11 — Known Issues & Gaps

**Status:** DRAFT | **Last updated:** 2026-06-07

This is the living issues tracker. Any gap found during documentation review or testing gets added here. When resolved, mark with ✅ and date.

---

## Critical (Blocks Testing on Quest 3)

| # | Issue | Where | Status |
|---|-------|-------|--------|
| C1 | ONNX models not validated on Quest 3 device (only tested in Editor + ONNX Runtime on Mac) | `AuraXRInferenceManager` + `onnx/` | ⏳ TODO |
| C2 | `BackendType.GPUCompute` on Snapdragon XR2 not confirmed working — may need `BackendType.CPU` fallback | `AuraXRInferenceManager.cs` line 543 | ⏳ TODO |
| C3 | `ReadbackAndClone()` for GPU→CPU tensor transfer may block rendering thread on Quest 3 — async version may be needed | `AuraXRInferenceManager.cs` line 454 | ⏳ TODO |

---

## Model Quality

| # | Issue | Where | Status |
|---|-------|-------|--------|
| M4 | Approach augmentation uses smoothstep blend — not validated against real pre-shape motion from the dataset | `build_dataset.py` lines 54–58 | ⏳ TODO |

---

## Unity Integration

| # | Issue | Where | Status |
|---|-------|-------|--------|
| U1 | `fingerJoints[15]` wiring in Inspector — not confirmed all 15 joints are wired for both hands | `HandRigController.cs` | ⏳ VERIFY |
| U2 | Pivot offset (16.85cm X, 3.51cm Z) was measured from one session — may vary per user or per headset fit | `AuraXRInferenceManager.cs` line 61 | ⏳ CHECK |
| U3 | Hand sign convention (`jointSignMultipliers`) is all +1 but untested with actual mesh bone orientations | `HandRigController.cs` lines 51–63 | ⏳ VERIFY |
| U4 | `ProximityDetector` range and update frequency not documented — may cause lag in nearest-object detection | `ProximityDetector.cs` | ⏳ TODO |
| U5 | Visual embedding (dims 32–95 in feature vector) is always 0 — camera input integration is a future gap | `AuraXRFeatureAssembler.cs` line 218 | ⏳ FUTURE |

---

## Pipeline Completeness

| # | Issue | Where | Status |
|---|-------|-------|--------|
| P2 | Evaluation results (MAE numbers) not filled into `05_training_evaluation.md` | `results/eval_right.json` | ⏳ TODO |
| P3 | Model quantization (Float32 → Int8/Float16) not implemented — needed for Quest 3 performance | `onnx/` | ⏳ FUTURE |
| P4 | No user study protocol documented | — | ⏳ TODO |

---

## How to Add New Issues

When you find a gap during review:
1. Add a row to the appropriate section above
2. Use these status codes: `⏳ TODO` / `⏳ VERIFY` / `⏳ UNCLEAR` / `⏳ FUTURE`
3. When resolved: replace with `✅ DONE (YYYY-MM-DD)` and add a note

---

## Resolved Issues

| # | Issue | Fixed | Notes |
|---|-------|-------|-------|
| M1 | Which architecture is deployed? | 2026-06-07 | Confirmed: `spatial_input_dim=8` in `model_meta_*.json`. |
| M2 | BopToGrip: 24/33 mismatches vs grip_categories.py | 2026-06-03 | Corrected in AuraXRInferenceManager.cs + UnityScripts/ backup. |
| M2b | BopToBbox: 33/33 mismatches vs grip_categories.py | 2026-06-03 | Same fix — both dicts now match Python ground truth exactly. |
| P1 | Frame counts unknown (XXX placeholder) | 2026-06-03 | Left: 877,985 total. Right: 1,021,853 total. |
| D1 | `quat_conjugate` / `rotate_vec` not inspected | 2026-06-07 | Both documented in 03_feature_engineering.md lines 72–73. |
| D3 | ONNX spatial_input shape (4 or 8 dims?) | 2026-06-07 | 8 dims confirmed: `[dir_world(3), dir_obj_local(3), dist(1), approach_speed(1)]`. Verified in `model.py` + `export_onnx.py`. |
