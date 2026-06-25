# Model Architecture

The deployed model is `SDFLSTMModel` in `src/model.py`.

```text
frame_feat (B, 29) -> Linear(29,64) -> LayerNorm -> ReLU
obj_embed  (B, 32) -> concat with frame projection
concat     (B, 96) -> Linear(96,64) -> LayerNorm -> ReLU
sequence   (B,T,64) -> LSTM(64,256), 2 layers, dropout 0.25
hidden     (B,T,256) -> pose / wrist / contact heads
```

Inputs:

- `frame_feat`: 25 core features plus 4 local SDF features.
- `obj_embed`: 32-dim object embedding from `data/models/sdf_grids/`.
- `h_0`, `c_0`: recurrent state for ONNX runtime.

Outputs:

- `pose_pca`: MANO PCA pose, 15 dims.
- `wrist_rot`: 6D wrist rotation.
- `contact_prob`: scalar contact probability.
- `h_n`, `c_n`: updated recurrent state.

Checkpoints use the `lstm.*` state dict prefix.
