# 4. Training

## 4.1. Datasets

### OakInk
- 1800 farklı obje, 50K+ statik grasp pozu, MANO formatı
- Modele öğrettiği şey: "Bu obje geometrisine göre geçerli final kavrama pozu nedir?"
- `T=1`, `rel_vel=0`, birleşik arayüzde `frame_feat (B,1,13)`
- Format: `hand_tsl (3,)`, `hand_pose (16,4)` wxyz quat, `obj_anno (4,4)` object-to-world
- Eklem pozisyonları kamera frame'inde — `cam_extr⁻¹` ile world frame'e taşınır
- Obje mesh: `OakBase/<category>/<instance>/part_*.ply`, metre cinsinden, object-local frame

### HOT3D
- 33 obje, egocentric (Aria + Quest 3), tam manipülasyon sekansları (pick-up → observe → put-down)
- Modele öğrettiği şey: "El bu poza zaman içinde nasıl kapanır?" (temporal geçiş kalitesi offline metrikle doğrulanamadı)
- `T>1` sliding window, `frame_feat (B,T,13)` object-relative
- Format: UmeTrack, `wrist_xform.t_xyz` + `wrist_xform.q_wxyz`, per-frame obje pozu `dynamic_objects.csv`
- Bilinen sınırlılık: thumb DOF hatası — başparmak pronasyon/supinasyon ekseni gürültülü

### Koordinat Dönüşüm Pipeline'ı
```
T_wrist_in_object = T_object_in_world⁻¹ @ T_wrist_in_world
rel_pos   = T_wrist_in_object[:3, 3]
rel_rot6d = rot_matrix_to_6d(T_wrist_in_object[:3,:3])
rel_vel   = (rel_pos[t] - rel_pos[t-1]) / Δt   # OakInk için sıfır
dist      = signed_distance_to_mesh_surface(wrist)
```

## 4.2. Data Splits

### OakInk
| Set | Oran | Yaklaşık Obje Sayısı |
|---|---|---|
| Train | %80 | ~1440 |
| Val | %10 | ~180 |
| Test | %10 | ~180 |

Kategori-stratified split; aynı obje farklı setlerde yok.

### HOT3D — 4-Seviyeli Obje Ayrımı
| Seviye | Obje Sayısı | Amaç |
|---|---|---|
| Backbone train | 22 | Model ağırlıklarını eğitmek |
| Backbone val | 4 | Erken durdurma, hiperparametre |
| Confidence calibration | 3 | Unity label → `success_prob` eğitimi |
| Final held-out test | 4 | Nihai metrik — tek seferlik kullanım |

Final held-out test objelerine hiçbir eğitim veya kalibrasyon adımında bakılmaz.

## 4.3. Training Protocol

### Aşama 1 — OakInk Statik Pre-training (Gerçekleşti)
```
Input:      frame_feat (B, 1, 13)  [T=1, rel_vel=0]
Window:     T=16 (eval'de; train'de T=1)
Loss:       L_recon + β*L_KL + L_contact + L_penetration + L_tip + L_quality
            limit_weight=1.0, contact_weight=0.3, penetration_weight=0.1
            tip_weight=0.5, quality_weight=0.1
β (kl_weight): 0.0 → warm-up → final 0.01 (50 epoch)
Optimizer:  Adam, lr=3e-4 (log'dan; docs'ta 1e-3 hedeflenmişti)
Early stop: 10 epoch (val_rec)
LR decay:   ReduceLROnPlateau(factor=0.5, patience=5)
```

**Aşama 1 eğitim seyri (train/val reconstruction loss):**

| Epoch | Train rec | Val rec | LR |
|---|---|---|---|
| 0 | 0.04013 | 0.02417 | 3e-4 |
| 10 | 0.01973 | 0.01786 | 3e-4 |
| 20 | 0.01612 | 0.01511 | 3e-4 |
| 30 | 0.01447 | 0.01418 | 3e-4 |
| 46 (best) | 0.01264 | **0.01205** | 3e-4 |

Best epoch: 46, val_total: 0.05684. Checkpoint: `checkpoints/aura_phase1_best.pt`

### Aşama 2 — HOT3D Temporal Fine-tuning (Gerçekleşti)
```
Input:      frame_feat (B, T, 13)  [T=16]
Batch:      MixedDataLoader: %70 HOT3D + %30 OakInk replay
Loss:       Aşama 1 loss + vel_weight=0.01*L_vel + acc_weight=0.04*L_acc
            (vel/acc yalnızca HOT3D batch'lerde aktif)
Optimizer:  AdamW, lr=1e-4 → 5e-5 (ReduceLROnPlateau, epoch 57'de decay)
Grad clip:  max_norm=1.0
Epoch aralığı: 47–60 (14 epoch)
Joint limits: hard clamp yok; joint_limit_loss soft gradient ile öğretir
```

**Aşama 2 eğitim seyri:**

| Epoch | Train rec | Val rec | Train vel | Train acc | LR |
|---|---|---|---|---|---|
| 47 | 0.01813 | 0.02230 | 0.00043 | 0.02775 | 1e-4 |
| 50 | 0.01759 | 0.02200 | 0.00040 | 0.03180 | 1e-4 |
| 55 | 0.01726 | 0.02142 | 0.00033 | 0.04225 | 1e-4 |
| 57 | 0.01711 | 0.02144 | 0.00029 | 0.04400 | 5e-5 |
| 60 | 0.01700 | 0.02159 | 0.00028 | 0.04854 | 5e-5 |

Best epoch: 50 (val_total: 0.06988). Checkpoint: `checkpoints/aura_phase2_best.pt`
Not: val_rec Phase 1'den yüksek (0.012 → 0.022) — HOT3D temporal distribüsyonu daha geniş/zor.

### Aşama 3 — Confidence Kalibrasyonu (Açık)
```
Loss:    BCE(success_prob, unity_label)
Backbone: frozen
Input:   HOT3D calibration objeleri (3 obje, Unity physics label)
Durum:   Unity success label pipeline henüz kurulmadı — success_prob head eğitilmedi
```
