# 11. Data Pipeline — Canonical Format ve Preprocessing

Kaynak: `docs/A-veri-hazirlama.md`, `src/preprocessing/`, `src/model/model_io.py`.

---

## 11.1. Dataset Özet Karşılaştırması

| Özellik | OakInk | HOT3D |
|---|---|---|
| Obje sayısı | ~1800 | 33 |
| Sample tipi | Statik grasp pozları | Temporal manipülasyon sekansları |
| El temsili | MANO (pose θ 48-dim, shape β 10-dim) | UmeTrack (bilek + 15 eklem rot) |
| Frame yapısı | T=1 (per-sample) | T>1 (sliding window) |
| Obje mesh | OakBase PLY (metre, object-local) | GLB (metre, object-local) |
| Thumb kalitesi | İyi | Bilinen DOF hatası (yaw ekseni hatalı) |
| Modele öğrettiği | Obje geometrisi → final poz | Zaman içinde el kapanış dinamiği |

---

## 11.2. OakInk Canonical Format

**Kaynak:** `build_oakink_canonical.py` → `data/processed/oakink_canonical/dataset.npz`

| Alan | Boyut | Açıklama |
|---|---:|---|
| `pose` | `(N, 48)` | MANO global orient(3) + finger_aa45(45) |
| `shape` | `(N, 10)` | MANO betas |
| `tsl` | `(N, 3)` | Bilek world frame translation |
| `obj_anno` | `(N, 12)` | Object-to-world: R(9, row-major) + t(3) |
| `fingertips_world` | `(N, 5, 3)` | GT parmak uçları — world frame |
| `obj_name` | `(N,)` | Obje kimliği |
| `category` | `(N,)` | Obje kategorisi |
| `obj_pts/{obj_name}.npy` | `(1024, 3)` | Canonical obje point cloud (metre) |

**Koordinat dönüşüm (wrist frame'e taşıma):**
```
pts_world = obj_pts_canonical @ R_obj.T + t_obj    # canonical → world
pts_wrist = (pts_world - wrist_tsl) @ R_wrist       # world → wrist
R_wrist = axis_angle_to_matrix(global_orient_aa)
```

**Split:** 80/10/10 sample-level random split. 8921 train / 1115 val / 1115 test.
Not: Obje bazlı değil, sample bazlı split. Final generalization testi için revize gerekebilir.

---

## 11.3. HOT3D Canonical Format

**Kaynak:** `build_hot3d_canonical_full.py` → `data/processed/hot3d_canonical/seq_*.npz`

| Alan | Boyut | Açıklama |
|---|---:|---|
| `rel_pos` | `(F, 3)` | Bilek-object relatif konum (object frame, metre) |
| `rel_rot6d` | `(F, 6)` | Bilek-object relatif 6D rotasyon |
| `rel_vel` | `(F, 3)` | Relatif bilek hızı (m/s) |
| `dist` | `(F, 1)` | Bilek-obje mesafesi (metre) |
| `finger_aa45` | `(F, 45)` | MANO parmak axis-angle |
| `fk_joints` | `(F, 16, 3)` | Gerçek MANO FK eklem pozisyonları (world frame) |
| `wrist_world_t` | `(F, 3)` | Bilek world translation (metre) |
| `wrist_world_q` | `(F, 4)` | Bilek world quaternion (wxyz) |
| `obj_world_t` | `(F, 3)` | Obje world translation |
| `obj_world_q` | `(F, 4)` | Obje world quaternion |
| `contact_flag` | `(F,)` | 3cm AABB eşiğiyle temas |
| `segment_id` | `(F,)` | Grasp segment kimliği |
| `obj_name` | `(F,)` | Obje adı |

**Genel istatistikler:** 157 sequence, 297.248 frame, 4.113 grasp segmenti.

---

## 11.4. HOT3D Grasp Segmentasyonu

**Parametreler (`model_io.py`):**

| Parametre | Değer | Açıklama |
|---|---|---|
| `APPROACH_DIST_THRESHOLD_M` | 0.15 m | Segment başlangıç mesafesi |
| `GRASP_CONTACT_THRESHOLD_M` | 0.005 m | Temas tespit eşiği |
| `GRASP_FINGER_MCP_DEG` | 20° | Min parmak kapanma açısı |
| `GRASP_VELOCITY_THRESHOLD_M_S` | 0.1 m/s | Min bilek hız eşiği |
| `TRANSITION_WINDOW_FRAMES` | 10 | Geçiş penceresi |
| AABB contact eşiği (build) | 3 cm | 3cm AABB parmak ucu temas tespiti |
| Min segment uzunluğu | 5 frame | Daha kısa segmentler atılır |
| Pre-context | 30 frame | Temas öncesi bağlam |
| Post-context | 5 frame | Temas sonrası bağlam |

**Not:** 3 cm AABB eşiği geniş; frame'lerin ~%79'unda quality_label = 0 (yaklaşım fazı dahil).

---

## 11.5. HOT3D Obje Split

**Kaynak:** `data/processed/hot3d_canonical/obj_split.json`

Gerçek uygulama: **4-seviyeli** değil, **3-seviyeli** obje-bazlı frame-level split (toplamda HOT3D train split, 33 değil bilinen 27 obje):

| Split | Objeler | Kategori örüntüsü | Frame sayısı |
|---|---|---|---|
| train | 21 obje | hook + wide + power | ~170k |
| val | keyboard, spatula_red, vase | wide + hook + power | ~38k |
| test | coffee_pot, dumbbell_5lb, whiteboard_eraser | hook + power + wide | ~28k |

Val ve test her üç grasp kategorisinden (hook, wide, power) birer obje içerir. Aynı sequence farklı split objelerine karışabilir; sequence-level leak olmadan obje-bazlı genelleme testi.

---

## 11.6. Batch Girdi Sözleşmesi (Eğitim)

| Alan | Boyut | Açıklama |
|---|---:|---|
| `frame_feat` | `(B, T, 13)` | Ana girdi: rel_pos(3)+rel_rot6d(6)+rel_vel(3)+dist(1) |
| `prev_frame_feat` | `(B, T, 13)` | 1 frame geriye (Phase 2 L_vel için) |
| `prev2_frame_feat` | `(B, T, 13)` | 2 frame geriye (Phase 2 L_acc için) |
| `target_pose` | `(B, 45)` | t anındaki GT parmak pozu |
| `prev_pose` | `(B, 45)` | t-1 GT parmak pozu |
| `prev2_pose` | `(B, 45)` | t-2 GT parmak pozu |
| `obj_pts` | `(B, 1024, 3)` | Canonical point cloud |
| `obj_pts_contact` | `(B, M, 3)` | Wrist-frame'e taşınmış noktalar (contact loss) |
| `contact_flag` | `(B, T, 1)` | Per-frame temas sinyali |
| `quality_label` | `(B, 1)` | Heuristic quality label |

---

## 11.7. Heuristic Quality Label

**Formül (`model_io.py`):**
```python
quality_raw = (
    w1 * contact_ratio          # QUALITY_W_CONTACT = 1.0
    - w2 * clip(pen / 0.01, 0, 1)   # QUALITY_W_PENETRATION = 0.3, max_pen=10mm
    - w3 * clip(dist / 0.05, 0, 1)  # QUALITY_W_DIST = 0.2, max_dist=5cm
)
quality_label = clip(quality_raw, 0.0, 1.0)
```

Contact eşiği: `CONTACT_THRESHOLD_M = 0.030` (30mm — HOT3D AABB eşiğiyle tutarlı).

**Fingertip pozisyon kaynağı (kritik):**
- HOT3D: `fk_joints[t, [3,6,9,12,15]]` (stored world frame) → wrist frame
- OakInk: `fingertips_world` (hand_j cam→world, dataset.npz'de saklı) → wrist frame
- mano_fk.py simplified FK **kullanılmaz** (bkz. §12.1 FK bug fix)

**Quality label dağılımı (fix öncesi/sonrası):**

| Dataset | Label mean (öncesi) | Label mean (sonrası) | >0 oranı (öncesi→sonrası) |
|---|---|---|---|
| HOT3D | 0.000 | 0.065 | %0 → %18 |
| OakInk | 0.000 | 0.101 | %0 → %26 |

---

## 11.8. Augmentasyon Stratejisi

Docs'ta tanımlandı (`A6`), henüz uygulanmadı:

| Augmentasyon | Yöntem | Dataset |
|---|---|---|
| Yaklaşım yönü çeşitlendirme | ±45° yaw/pitch bilek trajectory rotation | HOT3D |
| Bilek hız pertürbasyon | σ=0.02 m/s Gaussian gürültü | HOT3D |
| Obje konum jitter | ±5mm Gaussian | Her ikisi |
| Temporal flip | Sekans tersine çevir | HOT3D |
| Bilek rotasyon çeşitlendirme | ±30° döndür + FK güncelle | OakInk |
| Ölçek pertürbasyon | ±10% mesh scale | OakInk |
| Eklem gürültüsü | ±2° Gaussian | Her ikisi |

**Not:** Phase 1 ve 2 eğitimlerinde augmentasyon uygulanmadı. Tüm sonuçlar augmentasyonsuz.

---

## 11.9. Normalizasyon

OakInk stats: `data/processed/oakink_canonical/stats.json`
HOT3D stats: `data/processed/hot3d_canonical/stats.json`

- `input_mean`, `input_std`: 13-dim frame_feat normalizasyonu
- `pts_mean`, `pts_std`: 3-dim per-axis (scalar değil) point cloud normalizasyonu

Unity runtime: yalnızca HOT3D stats kullanılmalı (phase 2 model HOT3D fine-tune).
OakInk stats ile phase 2 model anlamsız inference üretir.
