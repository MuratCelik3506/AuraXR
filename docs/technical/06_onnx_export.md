# 06 — ONNX Export & Unity Sentis Loading

**Last updated:** 2026-06-06

**Source files:**
- `src/export_onnx.py`
- `src/evaluate_onnx.py`
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
spatial_dummy = torch.zeros(1, 8)   # batch=1, 8-dim spatial
object_dummy  = torch.zeros(1, 7)
```

**Step 3: Export to ONNX**
```python
torch.onnx.export(
    model,
    (spatial_dummy, object_dummy),
    "auraxr_right.onnx",
    input_names=["spatial_input", "object_input"],
    output_names=["joint_angles", "wrist_rot_6d"],
    dynamic_axes={...},   # batch dimension is dynamic on all inputs/outputs
    opset_version=14,          # Unity Sentis requires opset 14+
    do_constant_folding=True,  # pre-computes static subgraph nodes at export time; reduces runtime ops
)
```

**Step 4: Verify with ONNX Runtime**
```python
sess = ort.InferenceSession(str(onnx_path))
inputs = {"spatial_input": spatial_dummy.numpy(), "object_input": object_dummy.numpy()}
outputs = sess.run(None, inputs)        # outputs[0]=joint_angles, outputs[1]=wrist_rot_6d

with torch.no_grad():
    pt_joints, pt_rot = model(spatial_dummy, object_dummy)   # model returns a tuple

max_diff_joints = abs(outputs[0] - pt_joints.numpy()).max()
max_diff_rot    = abs(outputs[1] - pt_rot.numpy()).max()
assert max_diff_joints < 1e-5   # must match
assert max_diff_rot    < 1e-5
```
The 1e-5 tolerance accounts for float32 rounding differences between PyTorch and ONNX Runtime's CPU backend. Note: `model()` returns a tuple `(joint_angles, wrist_rot_6d)` — never call `.numpy()` directly on the return value.

**Step 5: Copy model_meta.json**
`model_meta.json` is copied from the checkpoint directory to the `onnx/` directory. Unity reads this to denormalize model outputs.

---

## Output Files

| File | Purpose |
|------|---------|
| `onnx/auraxr_right.onnx` | Right hand model |
| `onnx/auraxr_left.onnx` | Left hand model |
| `onnx/model_meta_right.json` | Normalization stats for right model |
| `onnx/model_meta_left.json` | Normalization stats for left model |

---

## model_meta.json Schema

This JSON file is critical for Unity. It contains the normalization statistics needed to denormalize model outputs back to real angle values.

```json
{
  "feature_mean":    [15 floats],   // mean for each input feature dimension
  "feature_std":     [15 floats],   // std for each input feature dimension
  "target_mean":     [22 floats],   // mean for each UME joint angle (radians)
  "target_std":      [22 floats],   // std for each UME joint angle
  "wrist_rot_mean":  [6 floats],    // mean for 6D wrist rotation output
  "wrist_rot_std":   [6 floats],    // std for 6D wrist rotation output
  "architecture": {
    "spatial_input_dim": 8,
    "object_input_dim":  7,
    "output_dim":        22,
    "hidden_dim":        256,
    "embedding_dim":     128
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
using var spatialTensor = new Tensor<float>(new TensorShape(1, 8), spatialInput);
using var objectTensor  = new Tensor<float>(new TensorShape(1, 7), objectInput);
worker.SetInput("spatial_input", spatialTensor);
worker.SetInput("object_input",  objectTensor);
worker.Schedule();
var jointTensor = worker.PeekOutput("joint_angles") as Tensor<float>;
var wristTensor = worker.PeekOutput("wrist_rot_6d") as Tensor<float>;
using var jointsCpu = jointTensor.ReadbackAndClone();
using var wristCpu  = wristTensor.ReadbackAndClone();
```

---

## Opset Version Note

Unity Sentis supports **ONNX opset 14+**. The export uses `opset_version=14`. If you use a newer PyTorch with opset 17+ ops, you may need to downgrade or adjust the export.

