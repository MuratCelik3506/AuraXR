# 11 — Known Issues & Gaps

**Status:** DRAFT | **Last updated:** 2026-06-03

This is the living issues tracker. Any gap found during documentation review or testing gets added here. When resolved, mark with ✅ and date.

---

## Critical (Blocks Testing on Quest 3)

| # | Issue | Where | Status |
|---|-------|-------|--------|
| C1 | ONNX models not validated on Quest 3 device (only tested in Editor + ONNX Runtime on Mac) | `AuraXRInferenceManager` + `onnx/` | ⏳ TODO |
| C2 | `BackendType.GPUCompute` on Snapdragon XR2 not confirmed working — may need `BackendType.CPU` fallback | `AuraXRInferenceManager.cs` line 481 | ⏳ TODO |
| C3 | `ReadbackAndClone()` for GPU→CPU tensor transfer may block rendering thread on Quest 3 — async version may be needed | `AuraXRInferenceManager.cs` line 392 | ⏳ TODO |

---

## Model Quality

| # | Issue | Where | Status |
|---|-------|-------|--------|
| M1 | Deployed ONNX uses V1 architecture (spatial_input_dim=4). V2 (`model_v2.py`) trained but not exported. V6 checkpoint = V1 hyperparameter variant, not V2. | `checkpoints/`, `onnx/model_meta_*.json` | ✅ CLARIFIED (2026-06-03) |
| M2 | **FIXED** BopToGrip had 24/33 wrong values AND BopToBbox had 33/33 wrong values in `AuraXRInferenceManager.cs`. Both corrected to match `grip_categories.py` exactly. | `AuraXRInferenceManager.cs` lines 109–153 | ✅ FIXED (2026-06-03) |
| M3 | No ablation table comparing V1 / V2 / V5 / V6 MAE scores side-by-side | `results/` | ⏳ TODO |
| M4 | Approach augmentation uses smoothstep blend — is this validated against real pre-shape motion from the dataset? | `build_dataset.py` lines 54–58 | ⏳ TODO |

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
| P1 | Actual frame counts after filtering not documented (placeholder XXX in `01_pipeline_overview.md`) | `data/left/dataset.h5`, `data/right/dataset.h5` | ⏳ TODO |
| P2 | Evaluation results (MAE numbers) not filled into `05_training_evaluation.md` | `results/eval_right.json` | ⏳ TODO |
| P3 | Model quantization (Float32 → Int8/Float16) not implemented — needed for Quest 3 performance | `onnx/` | ⏳ FUTURE |
| P4 | No user study protocol documented | — | ⏳ TODO |

---

## Documentation Gaps Found During Review

| # | Found in doc | Gap | Status |
|---|-------------|-----|--------|
| D1 | [03_feature_engineering.md](03_feature_engineering.md) | `hot3d_utils.py` `quat_conjugate` and `rotate_vec` implementations not inspected | ⏳ TODO |
| D2 | [04_model_architecture.md](04_model_architecture.md) | V5 vs V6 differences not described — what changed? | ⏳ TODO |
| D3 | [06_onnx_export.md](06_onnx_export.md) | V6 ONNX spatial_input shape not confirmed (4 or 8 dims?) | ⏳ TODO |

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
| M1 | Which architecture is deployed? | 2026-06-03 | V1 confirmed via model_meta.json. V2 not yet exported. |
| M2 | BopToGrip: 24/33 mismatches vs grip_categories.py | 2026-06-03 | Corrected in AuraXRInferenceManager.cs + UnityScripts/ backup. |
| M2b | BopToBbox: 33/33 mismatches vs grip_categories.py | 2026-06-03 | Same fix — both dicts now match Python ground truth exactly. |
| P1 | Frame counts unknown (XXX placeholder) | 2026-06-03 | Left: 877,985 total. Right: 1,021,853 total. |
