# 6. Experiments and Results

## Ana Araştırma Soruları (Mevcut Durumda Cevaplanabilenler İşaretli)

1. Tek model OakInk+HOT3D'yi birleştirerek hem statik grasp kalitesini hem temporal stabiliteyi sağlayabiliyor mu? — **Kısmen: pipeline çalışıyor, baseline olmadan temporal katkı ölçülmedi**
2. Point cloud temsili, basit bounding box'a göre grasp kalitesini artırıyor mu? — **Cevaplanmadı: BBox baseline deneyi yapılmadı**
3. Temporal pencere kullanmak, tek-frame modele göre daha stabil parmak pozu üretiyor mu? — **Cevaplanmadı: SingleFrame baseline deneyi yapılmadı**
4. CVAE ile K aday üretmek, deterministik tek çıktıya göre başarı oranını artırıyor mu? — **Hayır: K=1/3/5 arasında anlamlı fark yok**
5. `quality_score` ve `success_prob` head'leri başarısız graspları önceden ayırt edebiliyor mu? — **Kısmen hayır: OakInk'te orta düzey, HOT3D'de çok zayıf; success_prob eğitilmedi**
6. Model gerçek zamanlı XR kullanımına uygun latency'de çalışıyor mu? — **Cevaplanmadı: latency benchmark yapılmadı**

---

## 6.0. Mevcut Sonuçlar — Genel Tablo

Tüm eval: `git commit 52f3ce8`, device: MPS (Apple Silicon), window=16.

### Phase 1 Checkpoint (`aura_phase1_best.pt`, epoch 46) — OakInk

| Split | Geodesic (°) | MPJPE (mm) | Fingertip (mm) | Contact ratio | Pen. (mm) | Jt. limit viol. |
|---|---|---|---|---|---|---|
| OakInk val | 9.54 | 5.64 | 11.78 | 0.140 | 0.46 | 38.6% |
| OakInk test | 9.72 | 5.73 | 12.02 | 0.131 | 0.48 | 38.2% |

**quality_score metrikleri (Phase 1):**

| Split | AUC | ECE | Spearman |
|---|---|---|---|
| OakInk val | 0.944 | 0.054 | 0.724 |
| OakInk test | 0.967 | 0.058 | 0.712 |

### Phase 2 Checkpoint (`aura_phase2_best.pt`, epoch 50) — OakInk + HOT3D

**OakInk (statik grasp, T=1):**

| Split | Geodesic (°) | MPJPE (mm) | Fingertip (mm) | Contact ratio | Pen. (mm) | AUC | Spearman |
|---|---|---|---|---|---|---|---|
| val | 9.16 | 5.48 | 11.49 | 0.139 | 0.46 | 0.949 | 0.727 |

**HOT3D temporal (K=1, T=16):**

| Split | Geodesic (°) | MPJPE (mm) | Fingertip (mm) | Contact ratio | Pen. (mm) | Jt. viol. | AUC | Spearman |
|---|---|---|---|---|---|---|---|---|
| val | 11.70 | 6.06 | 13.77 | 0.146 | 4.02 | 53.9% | 0.336 | 0.157 |
| test | 12.58 | 6.62 | 14.92 | 0.230 | 1.36 | 55.1% | 0.704 | 0.500 |

**HOT3D temporal metrikler (Phase 2, val):**

| K | Geodesic (°) | Contact ratio | Pen. (mm) | Jitter score | Jitter vel | Geodesic vel (°) | Jitter acc | Diversity |
|---|---|---|---|---|---|---|---|---|
| K=1 | 11.70 | 0.146 | 4.02 | 6.15 | 0.273 | 3.36 | 0.362 | — |
| K=3 | 11.74 | 0.146 | 4.01 | 6.15 | 0.272 | 3.35 | 0.359 | 0.0512 |
| K=5 | 11.76 | 0.146 | 4.01 | 6.14 | 0.273 | 3.36 | 0.361 | 0.0512 |

**HOT3D test vs val farkı:**
- Geodesic: test 12.58° > val 11.70° — held-out objeler için hata artıyor
- Contact ratio: test 0.230 > val 0.146 — bu fark obje şekline bağlı varyans olabilir; "held-out objeler daha iyi" yorumu yanıltıcıdır, 3 obje üzerinden genelleme yapılamaz
- Penetration: test 1.36mm < val 4.02mm — büyük olasılıkla centroid-proxy metric artifact'ı (bkz. §12.9), gerçek penetrasyon farkı değil

---

## 6.1. Gözlemler ve Analiz

### Contact Ratio Problemi
- Hedef: >0.70. Mevcut: OakInk ~0.13–0.14, HOT3D val ~0.15, HOT3D test ~0.23
- **Çok düşük**: Model parmak uçlarını obje yüzeyine yaklaştırmıyor.
- Olası nedenler:
  - `contact_penetration_loss` centroid-proxy kullanıyor (küresel olmayan objeler için yanıltıcı)
  - `L_contact`'ta 15mm hinge toleransı: MANO template FK ~9mm residual bırakıyor, hinge bu farkı absorbe ediyor ve gerçek temas sinyali zayıf
  - Contact loss ağırlığı (0.3) düşük olabilir

### Joint Limit Violation Rate Problemi
- Phase 1: ~38%, Phase 2: ~54%. Saturation rate: ~0.5–0.7% (çok düşük)
- **Yorum**: Joint limit loss soft gradient üretiyor ama violation çok fazla — model anatomik sınırları sıklıkla aşıyor. Hard clamp kaldırıldı (gradient kesmesini önlemek için) ama yeterli regularizasyon sağlanamadı.
- HOT3D'de Phase 2 sonrası violation %38→%54: temporal fine-tuning joint limit regularizasyonunu zayıflattı.

### quality_score HOT3D'de Zayıf
- OakInk val Spearman: 0.724 — heuristic label statik grasp için orta düzey korelasyon gösteriyor
- HOT3D val Spearman: 0.157 (çok zayıf) — temporal sekansda heuristic label ile quality arasındaki ilişki kopuyor
- HOT3D test Spearman: 0.500 — test setinde biraz daha iyi (object-level varyans)
- HOT3D AUC: val 0.336 (rastgele tahminden kötü kısmı), test 0.704
- **Sonuç**: quality_score heuristic'i HOT3D temporal bağlamı için yetersiz

### Diversity Score Düşük
- K=3 ve K=5: diversity ~0.051 (çok düşük)
- CVAE latent space yeterince çeşitli kavrama üretemiyor — KL ağırlığı çok düşük veya training collapse başlangıcı

### Phase 2 Training Gözlemleri
- val_rec Phase 2'de Phase 1'den yüksek (0.012 → 0.022): HOT3D temporal dağılımı OakInk'ten farklı; beklenen
- vel/acc loss çok küçük değerler (0.0003/0.047): temporal regularizasyon aktif ama katkısı sınırlı
- LR decay epoch 57'de (5e-5): 14 epoch yeterli convergence vermedi, daha uzun eğitim gerekebilir

---

## 6.2. Training Rejimi Karşılaştırması (Deney 0)

Mevcut sonuçlar Phase 1 ve Phase 2 (OakInk pretrain + HOT3D fine-tune) için mevcut.
Diğer varyantlar (HOT3D-only, mixed 70/30 vs 50/50) henüz çalıştırılmadı.

| Varyant | OakInk Geodesic (°) | HOT3D Geodesic (°) | Contact ratio | Jitter score | Durum |
|---|---|---|---|---|---|
| OakInk pretrain + HOT3D FT | 9.16 (val) | 12.58 (test) | 0.13 / 0.23 | 4.91 (test) | **Tamamlandı** |
| OakInk-only static | — | — | — | — | Bekliyor |
| HOT3D-only temporal | — | — | — | — | Bekliyor |
| Mixed 70/30 | — | — | — | — | Bekliyor |
| Mixed 50/50 | — | — | — | — | Bekliyor |

---

## 6.3. Obje Temsili Karşılaştırması (Deney 1)

Mevcut sistem PointNet-Grasp. Diğer varyantlar henüz çalıştırılmadı.

| Model | OakInk Geodesic (°) | Contact ratio | Pen. (mm) | Durum |
|---|---|---|---|---|
| PointNet-Grasp (mevcut) | 9.16 val / 9.72 test | 0.131–0.140 | 0.46–0.48 | **Tamamlandı** |
| MLP-BBox | — | — | — | Bekliyor |
| MLP-BBox+Pose | — | — | — | Bekliyor |
| PointNet+Normals | — | — | — | Bekliyor |

---

## 6.4. Temporal Encoder Karşılaştırması (Deney D4)

Mevcut: Temporal-GRU, T=16. Diğer varyantlar ve window boyutları bekliyor.

| Model | HOT3D Geodesic (°) | Contact ratio | Jitter score | Geodesic vel (°/frame) | Durum |
|---|---|---|---|---|---|
| Temporal-GRU T=16 (mevcut) | 12.58 (test) | 0.230 | 4.91 | 3.59 | **Tamamlandı** |
| SingleFrame | — | — | — | — | Bekliyor |
| Temporal-GRU T=4 | — | — | — | — | Bekliyor |
| Temporal-GRU T=8 | — | — | — | — | Bekliyor |
| Temporal-TCN | — | — | — | — | Bekliyor |
| Temporal-Transformer | — | — | — | — | Bekliyor |

---

## 6.5. Çoklu Aday Üretimi (Deney D5)

Mevcut: K=1, 3, 5 HOT3D val ve test üzerinde değerlendirildi.

| K | HOT3D val Geodesic (°) | Contact ratio | Pen. (mm) | Diversity | Jitter score |
|---|---|---|---|---|---|
| K=1 | 11.70 | 0.146 | 4.02 | — | 6.15 |
| K=3 | 11.74 | 0.146 | 4.01 | 0.0512 | 6.15 |
| K=5 | 11.76 | 0.146 | 4.01 | 0.0512 | 6.14 |

**Gözlem**: K artışı geodesic error, contact ratio veya jitter'da anlamlı değişiklik yaratmıyor. Diversity ~0.051 çok düşük — CVAE yeterli çeşitlilik üretemiyor.

---

## 6.6. Confidence Kalibrasyonu (Deney D6)

### quality_score (heuristic, Aşama 1+2)

| Dataset | Split | AUC | ECE | Spearman |
|---|---|---|---|---|
| OakInk | val | 0.944–0.949 | 0.051–0.054 | 0.724–0.727 |
| OakInk | test | 0.967 | 0.058 | 0.712 |
| HOT3D | val | 0.336 | 0.058 | 0.157 |
| HOT3D | test | 0.704 | 0.054 | 0.500 |

**Yorum**: OakInk'te quality_score Spearman ~0.72 — statik bağlamda heuristic label grasp kalitesiyle orta düzeyde ilişkili. HOT3D temporal'de Spearman 0.157 (val), 0.500 (test) — heuristic label temporal bağlamda kalite göstergesi olarak çalışmıyor.

### success_prob (Unity binary, Aşama 3)
- Durum: **Eğitilmedi** — Unity physics label pipeline henüz kurulmadı
- Mevcut success_prob değerleri: sadece sıfırdan başlatılmış head'den geliyor, anlamlı değil

---

## 6.7. Per-Object Analiz — OakInk Val (Phase 2)

HOT3D val 24 obje, per-object breakdown:

| Obje | N samples | Geodesic (°) | Fingertip (mm) | Contact ratio | Pen. (mm) |
|---|---|---|---|---|---|
| mouse_51 | 80 | 11.68 | 14.72 | 0.198 | 0.032 |
| cameras_50 | 90 | 11.60 | 12.70 | 0.358 | 0.389 |
| binoculars_42 | 80 | 11.36 | 13.38 | 0.250 | 0.866 |
| eyeglasses_40 | 73 | 10.22 | 12.03 | 0.227 | 2.555 |
| gamecontroller_43 | 81 | 10.19 | 12.17 | 0.272 | 0.351 |
| pincer_36 | 85 | 9.85 | 12.24 | **0.033** | 0.0 |
| headphones_41 | 85 | 9.56 | 11.59 | 0.304 | 0.747 |
| scissors_35 | 71 | 8.82 | 11.13 | **0.045** | 0.012 |
| can_17 | 13 | 8.52 | 11.66 | **0.000** | 0.0 |
| lightbulb_28 | 72 | 8.34 | 10.75 | **0.053** | 0.0 |
| power_drill_27 | 58 | 8.18 | 10.94 | **0.079** | 1.468 |
| knife_20 | 30 | 7.73 | 11.22 | **0.027** | 0.0 |
| fryingpan_26 | 51 | 7.66 | 10.07 | 0.102 | 0.167 |
| wrench_24 | 43 | 7.48 | 9.94 | **0.014** | 0.0 |
| bottle_16 | 6 | 7.44 | 10.61 | **0.067** | 1.363 |
| cylinder_bottle_15 | 3 | 7.20 | 10.13 | **0.000** | 0.0 |
| toothbrush_25 | 44 | 7.17 | 9.15 | **0.000** | 0.0 |
| wineglass_14 | 7 | 7.02 | 10.01 | **0.000** | 0.0 |
| screwdriver_21 | 34 | 6.64 | 9.76 | **0.000** | 0.0 |
| hammer_22 | 52 | 6.44 | 8.96 | **0.031** | 0.001 |
| teapot_13 | 4 | 6.03 | 8.57 | **0.000** | 0.0 |
| flashlight_23 | 48 | 6.02 | 8.94 | **0.004** | 0.0 |
| mug_10 | 2 | 5.97 | 9.60 | **0.000** | 0.0 |
| cup_11 | 3 | **4.53** | **7.20** | **0.000** | 0.0 |

**Gözlemler:**
- Geodesic error: büyük/karmaşık objeler (mouse, cameras) daha yüksek hata üretiyor
- Contact ratio: çoğu obje için 0.0 veya çok düşük — temas sorunu yaygın
- En iyi contact: cameras_50 (%36), headphones_41 (%30), binoculars_42 (%25)
- Penetrasyon: eyeglasses_40'ta 2.56mm yüksek; power_drill'de 1.47mm

---

## 6.8. Ablation Study (Deney D7)

Henüz çalıştırılmadı.

| Ablation | Amaç | Durum |
|---|---|---|
| Point cloud yok, sadece bbox | Geometri bilgisinin katkısı | Bekliyor |
| Normals yok | Yüzey normalinin katkısı | Bekliyor |
| Temporal yok | Zaman bilgisinin katkısı | Bekliyor |
| Previous pose yok | Smoothness için prev_pose_emb katkısı | Bekliyor |
| Confidence head yok | Aday seçiminin katkısı | Bekliyor |
| CVAE yok | Çoklu grasp üretiminin katkısı | Bekliyor |

---

## 6.9. Runtime Latency

Henüz ölçülmedi. Hedefler:

| Model | Hedef |
|---|---|
| Temporal-GRU K=1 | < 5 ms |
| Temporal-GRU K=3 | < 8 ms |
| Temporal-GRU K=5 | < 10 ms |

---

## 6.10. Unity Physics Eval

Henüz çalıştırılmadı. Bağımlılıklar:
- Unity XR Hands retargeting (MANO → XR Hands bone mapping)
- Fizik eval pipeline ve success label export
- success_prob Aşama 3 eğitimi
