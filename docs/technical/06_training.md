# Training

Use `src/train_lstm.py`.

Single-source HOT3D:

```bash
.venv/bin/python3 src/train_lstm.py \
  --data_dir data/processed/hot3d_mano/right \
  --output_dir checkpoints/lstm_right \
  --hand right
```

Multi-source training:

```bash
.venv/bin/python3 src/train_lstm.py \
  --hot3d_dir data/processed/hot3d_mano/right \
  --arctic_dir data/processed/arctic_mano/right \
  --dexycb_dir data/processed/dexycb_mano/right \
  --output_dir checkpoints/lstm_right \
  --hand right
```

Default hyperparameters:

| Setting | Value |
|---|---:|
| Window length | 16 frames |
| Window stride | 4 |
| Batch size | 256 |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Max epochs | 200 |
| Early stopping patience | 50 |

Checkpoint output:

```text
checkpoints/lstm_left/best.pt
checkpoints/lstm_right/best.pt
```

Training depends on:

```text
data/models/sdf_grids/sdf_embed_matrix.npy
data/models/sdf_grids/sdf_bop_ids.npy
data/processed/<source>_mano/<hand>/dataset_mano.h5
```
