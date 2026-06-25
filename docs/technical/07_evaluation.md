# Evaluation

Use `src/evaluate_lstm.py`.

```bash
.venv/bin/python3 src/evaluate_lstm.py \
  --ckpt checkpoints/lstm_right/best.pt \
  --data data/processed/hot3d_mano/right \
  --hand right \
  --out results/eval_lstm_right.json
```

Metrics:

| Metric | Meaning |
|---|---|
| `mpjpe_mm` | Mean per-joint position error after MANO FK |
| `pa_mpjpe_mm` | Procrustes-aligned MPJPE |
| `mae_deg` | Mean absolute MANO PCA pose error |
| `pck_auc_50` | PCK AUC over 0-50 mm thresholds |
| `approach_mpjpe_mm` | MPJPE where wrist-object distance is below 20 cm |
| `contact_accuracy` | Contact-labelled frames with MPJPE below 5 mm |
| `smoothness_accel_mean` | Mean absolute second-order pose difference |

Current validation scores:

| Hand | Frames | MPJPE | PA-MPJPE | MAE | PCK-AUC@50 | Contact@5mm |
|---|---:|---:|---:|---:|---:|---:|
| Left | 134,418 | 6.87 mm | 6.43 mm | 21.28 deg | 0.8625 | 27.36% |
| Right | 135,388 | 7.66 mm | 7.06 mm | 22.56 deg | 0.8469 | 32.44% |
