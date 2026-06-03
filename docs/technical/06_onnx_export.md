# 06 — ONNX Export & Unity Sentis Loading

**Status:** DRAFT | **Last updated:** 2026-06-03

**Source files:**
- `hot3d_exploration/export_onnx.py`
- `hot3d_exploration/evaluate_onnx.py`
- `onnx/model_meta_left.json`
- `onnx/model_meta_right.json`

---

## Why ONNX?

Unity's ML inference system, **Unity Sentis** (formerly Unity Barracuda), runs ONNX models natively on the device GPU. Converting from PyTorch to ONNX is the bridge between Python training and C# runtime inference.

---

## Export: `export_onnx.py`

### Run command
```bash
python export_onnx.py --checkpoint ../checkpoints/right/ --output_dir ../onnx/
python export_onnx.py --checkpoint ../checkpoints/left/  --output_dir ../onnx/
```

### What it does (step by step)

**Step 1: Load model from checkpoint**
```python
meta = json.load(checkpoint/model_meta.json)   # architecture config
model = AuraXRModel(**meta["architecture"])
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
```

**Step 2: Create dummy inputs**
```python
spatial_dummy = torch.zeros(1, 4)   # batch=1
object_dummy  = torch.zeros(1, 7)
```

**Step 3: Export to ONNX**
```python
torch.onnx.export(
    model,
    (spatial_dummy, object_dummy),
    "auraxr_right.onnx",
    input_names=["spatial_input", "object_input"],
    output_names=["joint_angles"],
    dynamic_axes={...},   # batch dimension is dynamic
    opset_version=14,     # Unity Sentis requires opset 14+
    do_constant_folding=True,  # fuse constant ops for speed
)
```

**Step 4: Verify with ONNX Runtime**
```python
sess = ort.InferenceSession("auraxr_right.onnx")
onnx_out = sess.run(None, inputs)[0]
pytorch_out = model(spatial_dummy, object_dummy).numpy()
assert abs(onnx_out - pytorch_out).max() < 1e-5   # must match
```

**Step 5: Copy model_meta.json**
`model_meta.json` is copied from the checkpoint directory to the `onnx/` directory. Unity reads this to denormalize model outputs.

---

## Output Files

| File | Purpose |
|------|---------|
| `onnx/auraxr_right.onnx` | Right hand model |
| `onnx/auraxr_left.onnx` | Left hand model |
| `onnx/auraxr_right_v6.onnx` | Right hand V6 variant |
| `onnx/auraxr_left_v6.onnx` | Left hand V6 variant |
| `onnx/model_meta_right.json` | Normalization stats for right model |
| `onnx/model_meta_left.json` | Normalization stats for left model |

---

## model_meta.json Schema

This JSON file is critical for Unity. It contains the normalization statistics needed to denormalize model outputs back to real angle values.

```json
{
  "feature_mean":  [11 floats],   // mean for each input feature dimension
  "feature_std":   [11 floats],   // std for each input feature dimension
  "target_mean":   [22 floats],   // mean for each output joint angle
  "target_std":    [22 floats],   // std for each output joint angle
  "architecture": {
    "spatial_input_dim": 4,
    "object_input_dim":  7,
    "output_dim":        22,
    "hidden_dim":        128,
    "embedding_dim":     64
  }
}
```

**Unity denormalization formula:**
```csharp
// In AuraXRInferenceManager.cs:
float angle_i = model_output[i] * tgtStd[i] + tgtMean[i];  // radians
```

**Verified values from `onnx/model_meta_right.json` (first 5 joints):**
```
target_mean: [0.159, -0.012, 0.239, 0.092, -0.037, ...]  radians — looks correct
target_std:  [0.232,  0.217, 0.241, 0.323,  0.127, ...]
```
These are in the range 0.1–0.3 rad (6–17°), which matches typical finger joint angles.

---

## Unity Sentis: How the ONNX Model Is Loaded

**Setup in Unity:**
1. Copy `.onnx` files → `Assets/AuraXR/Models/` (or `Assets/Resources/`)
2. Copy `model_meta_*.json` → same folder
3. In `AuraXRInferenceManager.cs`, assign in Inspector:
   - `rightModelAsset` ← drag `auraxr_right.onnx`
   - `leftModelAsset` ← drag `auraxr_left.onnx`
   - `rightMetaJson` ← drag `model_meta_right.json`
   - `leftMetaJson` ← drag `model_meta_left.json`

**Runtime loading (Start method):**
```csharp
var model = ModelLoader.Load(rightModelAsset);
_workerRight = new Worker(model, BackendType.GPUCompute);
```
`BackendType.GPUCompute` uses the device GPU (Snapdragon on Quest 3).

**Running inference (Update method):**
```csharp
using var spatialTensor = new Tensor<float>(new TensorShape(1, 4), spatialInput);
using var objectTensor  = new Tensor<float>(new TensorShape(1, 7), objectInput);
worker.SetInput("spatial_input", spatialTensor);
worker.SetInput("object_input",  objectTensor);
worker.Schedule();
var outTensor = worker.PeekOutput("joint_angles") as Tensor<float>;
using var cpu = outTensor.ReadbackAndClone();   // copy GPU→CPU
```

---

## Opset Version Note

Unity Sentis supports **ONNX opset 14+**. The export uses `opset_version=14`. If you use a newer PyTorch with opset 17+ ops, you may need to downgrade or adjust the export.

---

## What to Inspect Together

When reviewing this document with the professor, check:
- [ ] Open `onnx/model_meta_right.json` — do the `target_mean` values look like real radian values (roughly 0.1–0.5 rad per finger joint)?
- [ ] Does `BackendType.GPUCompute` work on Quest 3? Or should we use `BackendType.CPU` as fallback?
- [ ] Is constant folding (`do_constant_folding=True`) safe? (It can cause issues with dynamic batch sizes — verify the ONNX verification step passes)
- [ ] V6 ONNX models exist — are they using V1 or V2 architecture? Check if spatial_input shape is [1,4] or [1,8].
