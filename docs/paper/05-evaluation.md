# 5. Evaluation Protocol

## 5.1. Offline Metrics

### Geodesic Rotation Error (Ana Metrik)
```
d(R_pred, R_gt) = arccos( (trace(R_pred^T R_gt) - 1) / 2 )   [derece]
```
Her eklem için hesaplanır, ortalama raporlanır. Axis-angle MAE yalnızca yardımcı metrik.

### MPJPE ve Fingertip Position Error
```
mpjpe         = mean over joints of ||FK(pred) - FK(gt)||      [mm]
fingertip_err = mean over 5 fingertips of ||pos_pred - pos_gt|| [mm]
```

### Contact Ratio
```
contact_ratio = (FK parmak uçları mesh'e < 5mm olan uç sayısı) / 5
```
Hedef: > 0.7

### Penetration Depth
```
penetration = mean(max(0, -SDF(fingertip_pos)))   [mm]
```
Hedef: < 3mm ortalama

### Joint-Limit Violation Rate
Anatomik sınırları aşan eklem yüzdesi.

### Diversity Score (CVAE için)
```
diversity = mean pairwise geodesic distance between K samples
```
K=5 örnek, aynı bilek pozu + obje için.

## 5.2. Temporal Stability Metrics

### Frame-to-Frame Geodesic Velocity (Ana Jitter Metriği)
```
jitter_vel = mean over frames of d(R_pred_t, R_pred_{t-1})
```

### Frame-to-Frame Geodesic Acceleration
```
jitter_acc = mean over frames of |d(R_t, R_{t-1}) - d(R_{t-1}, R_{t-2})|
```

### Jitter Score
```
jitter_score = max_velocity / mean_velocity   ↓ daha iyi
```

### Contact Stability
Temas oranının frame-to-frame varyansı (sürekliliği ölçer).

## 5.3. Unity Physics Evaluation

### Protokol
1. Obje sahneye yerleştirilir.
2. Model parmak pozu üretir (`argmax(success_prob)` ile seçilen aday).
3. Pose avatar/collider ele uygulanır.
4. 0.5 saniye stabilize olması beklenir.
5. Bozucu kuvvet uygulanır: `F = α × m × g` (α=1.0, rastgele yön, 0.1s süre).
6. 1 saniye gözlem.

### Başarı Kriteri
```
success = d_norm < τ_d  and  rotation_change < τ_r  and  object_not_dropped
    τ_d = 0.10  (bbox çapının %10'u)
    τ_r = 15°
```
Eşikler calibration setinden belirlenir; final test setine dokunulmaz.

### Raporlanan Metrikler
- Success rate
- Contact ratio
- Penetration depth
- Object displacement
- Failure category (penetrasyon / temassızlık / yanlış yön / jitter)

### Unity Eval Obje Setleri
| Kullanım | Obje Seti | Obje Sayısı | Çıktı |
|---|---|---|---|
| Confidence calibration | HOT3D calibration | 3 | `success_label` |
| Final temporal test | HOT3D held-out | 4 | Nihai fizik metriği |
| Static geometry test (opsiyonel) | OakInk test alt kümesi | ~20–30 | Statik model genellemesi |

## 5.4. Confidence Calibration

### quality_score (heuristic)
- Spearman korelasyonu: quality_score vs. Unity success rate
- MSE on val set

### success_prob (Unity binary)
- AUC-ROC (hedef: > 0.80)
- Expected Calibration Error (ECE)
- Reliability diagram
- Precision/recall @ threshold 0.5, 0.7, 0.9

### Aday Seçim Doğruluğu
```
oracle_rank = rank of argmax(success_prob) candidate among K
mean_oracle_rank ↓ daha iyi
```

## 5.5. Runtime Latency

| Model | Hedef |
|---|---|
| SingleFrame | < 3 ms |
| Temporal-GRU | < 5 ms |
| Temporal-TCN | < 5 ms |
| Temporal-Transformer | < 8 ms |
| CVAE-K5 | < 10 ms |

Ölçüm: PC GPU (CUDA), warm-up sonrası 100 tekrar ortalaması.

## 5.6. İstatistiksel Raporlama Protokolü
- Her öğrenilmiş varyant en az **3 random seed** ile eğitilir; `mean ± std` raporlanır.
- Unity success rate için **%95 bootstrap CI** obje seviyesinde (frame değil).
- Model karşılaştırmaları: **paired bootstrap** (aynı obje/senaryo üzerinde).
- Obje başına ve kategori başına metrik ayrıca raporlanır.
- Validation set üzerinden model seçimi; **test set tek seferlik** kullanım.
