# AuraXR Unity Demo Package

This folder is a copyable Unity package for the model-demo scene described in
`docs/F-unity-demo-scene-plan.md`.

## Contents

- `Assets/AuraXR/Scripts`: runtime components for feature assembly, retargeting,
  blending, HUD, and JSONL logging.
- `Assets/AuraXR/Editor/AuraXRDemoSceneBuilder.cs`: menu item that creates the
  demo hierarchy: `AuraXR > Build Model Demo Scene`.
- `Assets/StreamingAssets/AuraXR/model_stats.json`: HOT3D normalization stats.
- `Assets/StreamingAssets/AuraXR/objects/*.bytes`: three `(1024, 3)` point
  clouds exported as float32 little-endian XYZ.
- `Assets/StreamingAssets/AuraXR/grasp_model.onnx`: exported ONNX model.

## Import

Copy the `Assets` folder contents into the active Unity project. If the project
already has an `Assets` folder, merge these subfolders:

```text
Assets/AuraXR
Assets/StreamingAssets/AuraXR
```

## Scene Setup

1. In Unity, run `AuraXR > Build Model Demo Scene`.
2. Select `AuraXRModelRuntime`.
3. Assign `model_stats.json` to `AuraXRFeatureAssembler.modelStatsJson`.
4. Assign each object point cloud TextAsset:
   - `mug_white_pts.bytes`
   - `can_parmesan_pts.bytes`
   - `spatula_red_pts.bytes`
5. Assign the ONNX `grasp_model.onnx` to `AuraXRModelRuntime.modelAsset`.
6. Define scripting symbol `AURAXR_USE_SENTIS` only after Sentis is installed
   and the local Sentis API version matches the adapter code.
7. Disable `AuraXRModelRuntime.bypassModel` after the Sentis model asset is assigned.

Until `AURAXR_USE_SENTIS` is enabled, the runtime uses an all-zero pose fallback.
That lets the feature assembler, warm-up state, blend logic, HUD, and JSONL logger
be tested before binding Sentis.

## Runtime Contract

The model input tensors are:

```text
frame_feat    (1, 16, 13)
obj_pts       (1, 1024, 3)
contact_flag  (1, 16, 1)
prev_pose     (1, 45)
```

The output tensors are:

```text
selected_pose (1, 45)
quality_score (1, 1)
success_prob  (1, 1)
```

`AuraXRFeatureAssembler` samples features at 30 Hz, waits for a full 16-frame
window before inference, and logs `model_state = warming_up` until ready.
