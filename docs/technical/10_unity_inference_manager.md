# Unity Inference Contract

Unity should load one ONNX model and one meta JSON per hand:

```text
onnx/auraxr_lstm_left.onnx
onnx/model_meta_lstm_left.json
onnx/auraxr_lstm_right.onnx
onnx/model_meta_lstm_right.json
```

Per-frame model inputs:

| Name | Shape |
|---|---|
| `frame_feat` | `(1, 29)` |
| `obj_embed` | `(1, 32)` |
| `h_0` | `(2, 1, 256)` |
| `c_0` | `(2, 1, 256)` |

Outputs:

| Name | Shape |
|---|---|
| `pose_pca` | `(1, 15)` |
| `wrist_rot` | `(1, 6)` |
| `contact_prob` | `(1, 1)` |
| `h_n` | `(2, 1, 256)` |
| `c_n` | `(2, 1, 256)` |

Runtime rules:

- Reset `(h, c)` when entering the approach state.
- Feed `h_n`, `c_n` back as `h_0`, `c_0` on the next frame.
- Use `model_meta_lstm_<hand>.json` to normalize features and denormalize outputs.
- Object embeddings come from the SDF grid/embedding database generated offline.
