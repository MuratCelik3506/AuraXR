# ONNX Export

Use `src/export_onnx_lstm.py`.

```bash
.venv/bin/python3 src/export_onnx_lstm.py \
  --ckpt checkpoints/lstm_right/best.pt \
  --data_dir data/processed/hot3d_mano/right \
  --hand right \
  --out_dir onnx
```

Generated files:

```text
onnx/auraxr_lstm_right.onnx
onnx/auraxr_lstm_right.onnx.data
onnx/model_meta_lstm_right.json
```

The exporter validates output shapes with ONNX Runtime before returning.
