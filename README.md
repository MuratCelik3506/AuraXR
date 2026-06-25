# AuraXR

AuraXR predicts VR hand pose from controller/wrist motion, object proximity, and
object geometry. The active production model is a stateful SDF-conditioned LSTM.

## Active Model

| Item | Value |
|---|---|
| Model | `SDFLSTMModel` |
| Input | 29 dims per frame: 25 core features + 4 local SDF features |
| Object geometry | 32-dim SDF embedding, looked up by object ID |
| Temporal state | LSTM `(h, c)`, 2 layers, hidden size 256 |
| Output | MANO PCA pose 15, wrist rotation 6D, contact probability |
| Checkpoints | `checkpoints/lstm_left/best.pt`, `checkpoints/lstm_right/best.pt` |
| ONNX | `onnx/auraxr_lstm_left.onnx`, `onnx/auraxr_lstm_right.onnx` |

## Current Scores

| Hand | Frames | MPJPE | PA-MPJPE | MAE | PCK-AUC@50 | Contact@5mm |
|---|---:|---:|---:|---:|---:|---:|
| Left | 134,418 | 6.87 mm | 6.43 mm | 21.28 deg | 0.8625 | 27.36% |
| Right | 135,388 | 7.66 mm | 7.06 mm | 22.56 deg | 0.8469 | 32.44% |

## Directory Layout

```text
data/
  raw/                         Raw HOT3D / ARCTIC / DexYCB downloads
  processed/
    hot3d_mano/{left,right}/   dataset_mano.h5 for LSTM training/eval
    arctic_mano/{left,right}/  Optional contact-pose augmentation
    dexycb_mano/{left,right}/  Optional contact-pose augmentation
  models/
    mano/                      MANO model files
    sdf_grids/                 SDF grids, embeddings, BOP id lookup

checkpoints/
  lstm_left/best.pt
  lstm_right/best.pt

onnx/
  auraxr_lstm_left.onnx
  auraxr_lstm_left.onnx.data
  auraxr_lstm_right.onnx
  auraxr_lstm_right.onnx.data
  model_meta_lstm_left.json
  model_meta_lstm_right.json

results/
  eval_lstm_left.json
  eval_lstm_right.json
```

Large generated artifacts are ignored by git. Keep raw data, processed HDF5,
checkpoints, ONNX exports, and evaluation results in the paths above so scripts
and Unity integration resolve files consistently.

## Commands

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train one hand:

```bash
.venv/bin/python3 src/train_lstm.py \
  --hot3d_dir data/processed/hot3d_mano/right \
  --arctic_dir data/processed/arctic_mano/right \
  --dexycb_dir data/processed/dexycb_mano/right \
  --output_dir checkpoints/lstm_right \
  --hand right
```

Evaluate:

```bash
.venv/bin/python3 src/evaluate_lstm.py \
  --ckpt checkpoints/lstm_right/best.pt \
  --data data/processed/hot3d_mano/right \
  --hand right \
  --out results/eval_lstm_right.json
```

Export ONNX:

```bash
.venv/bin/python3 src/export_onnx_lstm.py \
  --ckpt checkpoints/lstm_right/best.pt \
  --data_dir data/processed/hot3d_mano/right \
  --hand right \
  --out_dir onnx
```

## Maintained Code

| File | Purpose |
|---|---|
| `src/model.py` | `SDFLSTMModel`, offline `SDFEncoder`, optional `GraspFlowModel` |
| `src/train_lstm.py` | LSTM training on HOT3D with optional ARCTIC/DexYCB augmentation |
| `src/evaluate_lstm.py` | Sequence-order evaluation with FK metrics |
| `src/export_onnx_lstm.py` | Stateful ONNX export and metadata generation |
| `src/build_dataset_mano.py` | Process raw hand/object data into `dataset_mano.h5` |
| `src/compute_sdf_embeddings.py` | Build SDF embedding lookup tables |

Old ablation paths have been removed from the active project surface. The only
supported runtime model is SDF-LSTM.
