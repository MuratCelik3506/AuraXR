# D. Deney Planı

Bu dosya tezin deneylerini, farklı training koşullarını ve karşılaştırmaları tanımlar. Ana veri kaynakları OakInk ve HOT3D'dir; Unity yalnızca fizik tabanlı değerlendirme ve confidence kalibrasyonu için kullanılır.

---

## D1. Ana Araştırma Soruları

1. Tek ana model, OakInk ve HOT3D'den birlikte öğrenerek hem statik grasp kalitesini hem temporal stabiliteyi sağlayabiliyor mu?
2. Obje geometrisini point cloud ile vermek, basit bounding box temsiline göre grasp kalitesini artırıyor mu?
3. Temporal pencere kullanmak, tek-frame modele göre daha stabil ve daha az jitter yapan parmak pozu üretiyor mu?
4. CVAE ile birden fazla aday grasp üretmek, tek deterministik çıktıya göre başarı oranını artırıyor mu?
5. `quality_score` ve `success_prob` output head'leri başarısız grasp'leri önceden ayırt edebiliyor mu?
6. Model gerçek zamanlı XR kullanımına uygun latency'de çalışıyor mu?

---

## Model Sayımı

Tezde tek ana AI model vardır: **Temporal Geometry-Conditioned Grasp Model**.

| Sistem Parçası | Deneylerde Nasıl Sayılır? |
|---|---|
| Object encoder | Ana modelin bileşeni |
| Temporal encoder | Ana modelin bileşeni |
| Grasp decoder | Ana modelin bileşeni |
| Quality score head | Ana modelin çıkış başlığı (heuristik, MSE) |
| Success prob head | Ana modelin çıkış başlığı (Unity binary, BCE) |
| CVAE latent sampling | Ana modelin aday üretim mekanizması |
| Approach controller | AI model değil, kural tabanlı geçiş |
| Unity evaluator | AI model değil, fizik eval ortamı |

Bu nedenle deneyler "birden fazla bağımsız AI model" karşılaştırması olarak değil, aynı ana modelin temsil, temporal yapı ve training rejimi varyantları olarak raporlanır.

---

## D2. Veri Bölünmesi

**Tek kural:** Split obje kimliğine göre yapılır. Aynı obje hiçbir zaman iki farklı set arasında paylaşılamaz.

### OakInk

| Set | Oran | Yaklaşık Obje Sayısı |
|---|---|---|
| Train | %80 | ~1440 |
| Val | %10 | ~180 |
| Test | %10 | ~180 |

Obje kategorileri (kap, araç, elektronik...) her sette temsil edilmeli; kategori-stratified split uygulanır.

### HOT3D — 4-Seviyeli Obje Ayrımı

Backbone için görülmemiş olan objeler confidence kalibrasyonunda kullanılırsa, bu objeler artık tam anlamıyla "test objesi" sayılamaz. Temiz ayrım:

| Seviye | HOT3D Obje Sayısı | Amaç |
|---|---|---|
| Backbone train | 22 | Model ağırlıklarını eğitmek |
| Backbone val | 4 | Erken durdurma, hiperparametre seçimi |
| Confidence calibration | 3 | `success_prob` head için Unity label üretimi |
| Final held-out test | 4 | Nihai metrik raporlaması — tek seferlik |

**Toplam: 33 obje.** Final held-out test objelerine hiçbir eğitim veya kalibrasyon adımında bakılmaz.

### Unity Eval Obje Seçimi

Unity üç ayrı amaçla kullanılır; her kullanım farklı bir obje seti üzerinde çalışır:

| Kullanım | Obje Seti | Obje Sayısı | Çıktı |
|---|---|---|---|
| Confidence calibration | HOT3D calibration objeler | 3 | `success_label` → `success_prob` eğitimi |
| Final temporal test | HOT3D held-out objeler | 4 | Nihai fizik metriği — tek seferlik |
| Static geometry test (isteğe bağlı) | OakInk test alt kümesi (kategori-stratified) | ~20–30 | Statik modelin fizik genellemesi |

HOT3D calibration ve held-out objeleri birbirinden kesin olarak ayrıdır. OakInk Unity testi opsiyoneldir; 180 objenin tamamı için mesh/collider/fizik kurulumu yüksek maliyetlidir.

---

## D3. Model Karşılaştırmaları

### Deney 0 — Training Rejimi

Amaç: OakInk ve HOT3D'nin ana modele hangi katkıyı verdiğini ölçmek.

| Varyant | Eğitim Verisi | Ne Ölçer? |
|---|---|---|
| OakInk-only static | OakInk statik grasp pozları | Obje-geometri -> final grasp ilişkisi |
| HOT3D-only temporal | HOT3D sliding windows | Temporal kapanış ve stabilite |
| OakInk pretrain + HOT3D temporal | Önce OakInk, sonra HOT3D | Statik grasp bilgisinin temporal modele aktarımı |
| Mixed 70/30 | %70 OakInk + %30 HOT3D | Statik kalite ve temporal stabilite dengesi |
| Mixed 50/50 | %50 OakInk + %50 HOT3D | Daha güçlü temporal ağırlığın etkisi |

Metrikler:
- Geodesic rotation error (ana metrik)
- HOT3D temporal jitter (frame-to-frame velocity)
- Contact ratio
- Penetration depth
- Unity success rate

Beklenti:

> OakInk-only iyi final poz üretir ama temporal jitter'ı çözmez. HOT3D-only temporal davranış öğrenir ama obje çeşitliliği sınırlı kalabilir. En güçlü aday OakInk pretrain + HOT3D temporal veya mixed training olmalıdır.

### Deney 1 — Obje Temsili

Amaç: Obje geometrisi temsilinin etkisini ölçmek.

| Model | Obje Girdisi | Beklenen Rol |
|---|---|---|
| MLP-BBox | bbox: width, height, depth | En basit baseline |
| MLP-BBox+Pose | bbox + obje rotasyonu + mesafe | Daha güçlü baseline |
| PointNet-Grasp | point cloud | Ana geometri modeli |
| PointNet-Grasp + normals | point cloud + yüzey normali | Normal bilgisinin katkısı |

Metrikler:
- Geodesic rotation error: `d = arccos((trace(R1ᵀR2) - 1) / 2)`
- Fingertip position error (MPJPE)
- Contact ratio
- Penetration depth
- Unity grasp success rate

---

## D4. Temporal Karşılaştırmalar

Amaç: Temporal pencerenin tek-frame modele göre katkısını ölçmek.

| Model | Girdi | Açıklama |
|---|---|---|
| SingleFrame | mevcut bilek pozu + obje | Zamansız baseline |
| Temporal-GRU | son N frame bilek/parmak pozu + obje | Hafif temporal model |
| Temporal-TCN | son N frame + causal convolution | Paralel ve hızlı temporal model |
| Temporal-Transformer | son N frame + attention | Daha güçlü ama daha ağır model |

Başlangıç pencere boyutları:
- N=4
- N=8
- N=16

Temporal metrikler:
- Frame-to-frame geodesic velocity (ana jitter metriği)
- Frame-to-frame geodesic acceleration
- Jitter score (max velocity / mean velocity)
- Contact stability (temas sürekliliği)
- Unity grasp success rate
- Inference latency

Temel hipotez:

> Temporal model, tek-frame modele göre benzer contact ratio üretirken jitter ve ani parmak sıçramalarını azaltmalıdır.

---

## D5. Çoklu Kavrama Adayı Deneyi

Amaç: Aynı obje için birden fazla geçerli kavrama üretmenin katkısını ölçmek.

| Model | Aday Sayısı | Seçim |
|---|---|---|
| Deterministic | K=1 | Tek çıktı |
| CVAE-K3 | K=3 | argmax(success_prob) |
| CVAE-K5 | K=5 | argmax(success_prob) |

Metrikler:
- Diversity score
- Best-of-K contact ratio
- Best-of-K Unity success
- Latency

Not:
- Diffusion ana sistemde kullanılmaz.
- Çoklu kavrama özelliği CVAE latent sampling ile korunur.

---

## D6. Confidence Kalibrasyonu

Amaç: `quality_score` ve `success_prob` head'lerinin ne kadar bilgilendirici olduğunu ölçmek.

### quality_score (heuristic, Aşama 1+2'de eğitilir)

Label: `w1*contact_ratio - w2*clip(penetration/max_pen, 0,1) - w3*clip(avg_dist/max_dist, 0,1)`

Metrikler:
- Spearman korelasyonu (quality_score vs Unity success rate)
- MSE on val set

### success_prob (Unity binary, Aşama 3'te eğitilir)

Label: Unity physics success/fail — **yalnızca calibration alt setinden**

Metrikler:
- AUC-ROC
- Expected Calibration Error (ECE)
- Reliability diagram
- Precision/recall at threshold 0.5, 0.7, 0.9

### Aday Seçim Doğruluğu

K aday arasında `argmax(success_prob)` ile seçilen adayın gerçekten en iyi olup olmadığı:
```
oracle_rank = rank_of_selected_among_K_candidates   # 1=en iyi, K=en kötü
mean_oracle_rank ↓ daha iyi
```

> `quality_score` ve `success_prob` karıştırılmamalıdır. Nihai "confidence" iddiası `success_prob` üzerinden raporlanır.

---

## D7. Ablation Deneyleri

| Ablation | Amaç |
|---|---|
| Point cloud yok, sadece bbox | Geometri bilgisinin katkısı |
| Normals yok | Yüzey normalinin katkısı |
| Temporal yok | Zaman bilgisinin katkısı |
| Previous pose yok | Smoothness için önceki pozun katkısı |
| Confidence head yok | Aday seçiminin katkısı |
| CVAE yok | Çoklu grasp üretiminin katkısı |

---

## D8. Unity Fizik Eval

Her test senaryosu:

1. Obje sahneye yerleştirilir.
2. Model parmak pozu üretir.
3. Pose avatar/collider ele uygulanır.
4. 0.5 saniye stabilize olması beklenir.
5. Objeye bozucu kuvvet uygulanır.
6. Obje kayması/düşmesi ölçülür.

Başarı kriteri:

```
F = α × m × g  (α=1.0, rastgele yön, 0.1s süre)
Gözlem: 1 saniye
success = d_norm < τ_d  and  rotation_change < τ_r  and  object_not_dropped
    τ_d = 0.10  (bbox çapının %10'u)
    τ_r = 15°
    Eşikler calibration setinden belirlenir, final test setine dokunulmaz
    where d_norm = |Δx| / bbox_diagonal
```

Raporlanacak metrikler:
- Success rate
- Contact ratio
- Penetration depth
- Object displacement
- Failure category

---

## D9. Runtime Latency

Ölçülecek varyantlar:

| Model | Hedef |
|---|---|
| SingleFrame | < 3 ms |
| Temporal-GRU | < 5 ms |
| Temporal-TCN | < 5 ms |
| Temporal-Transformer | < 8 ms |
| CVAE-K5 | < 10 ms |

PC tabanlı XR kullanımında ana hedef:

> Grasp inference + aday seçimi toplamı 10 ms altında kalmalıdır.

---

## D10. Input / Output Doğrulama Deneyleri

Bu deneyler model kalitesinden önce veri-model-Unity hattının doğru bağlandığını kanıtlar.

### Birleşik Arayüz Testi

Amaç: OakInk (T=1) ve HOT3D (T>1) örneklerinin aynı `frame_feat (B,T,13)` formatında modele verilebildiğini doğrulamak.

**OakInk T=1 testi:**

| Alan | Beklenen Şekil | Kontrol |
|---|---:|---|
| `frame_feat` | `(B, 1, 13)` | rel_vel sıfır, dist hesaplanmış |
| `finger_hist` | `(B, 1, 45)` | statik başlangıç pozu |
| `obj_pts` | `(B, 1024, 3)` | canonical frame |
| `pred` | `(B, 45)` | NaN/Inf yok |
| `mu`, `logvar` | `(B, 64)` | finite |
| `quality_score` | `(B, 1)` | [0,1] aralığında |

**HOT3D T=8 testi:**

| Alan | Beklenen Şekil | Kontrol |
|---|---:|---|
| `frame_feat` | `(B, 8, 13)` | object-relative, timestamp sıralı |
| `finger_hist` | `(B, 8, 45)` | aynı segment_id içinde |
| `contact_flag` | `(B, 8, 1)` | grasp fazında artmalı |
| `target_pose` | `(B, 45)` | doğru frame'e karşılık gelmeli |

Başarı kriteri (her iki test için):
- OakInk ve HOT3D batch'leri aynı `model.forward()` çağrısıyla işlenir
- `rel_pos` ve `rel_rot6d` object-frame'de doğru dönüşüm uygulanmış
- `dist` yüzey mesafesi pozitif (el dışarıda) veya negatif (el içinde) doğru işaretli

### Unity Retarget Testi

Amaç: Modelin `finger_aa45` çıktısının Unity avatar rig'e doğru sırayla uygulanabildiğini doğrulamak.

Kontroller:
- MANO sıra: index, middle, ring, pinky, thumb
- Axis-angle → quaternion dönüşümü doğru
- XR rig local bone eksenleri ters/yanlış değil
- Parmaklar anatomik olmayan yönde bükülmüyor
- Tek statik pose ile avatar elde görsel kontrol yapılır

### Unity Eval Log Testi

Amaç: Unity fizik eval çıktısının confidence training/evaluation için yeterli alanları ürettiğini doğrulamak.

Beklenen log alanları:

| Alan | Açıklama |
|---|---|
| `object_id` | Test objesi |
| `test_direction` | Üstten/yandan/önden gibi eval yönü |
| `window_size` | Temporal pencere boyutu |
| `candidate_count` | CVAE aday sayısı |
| `success` | Fizik başarı etiketi |
| `quality_score` | Seçilen adayın heuristik kalite skoru |
| `success_prob` | Seçilen adayın Unity başarı olasılığı |
| `selected_candidate_index` | argmax(success_prob) ile seçilen aday indeksi |
| `candidate_quality_scores` | Tüm K adayın kalite skorları |
| `candidate_success_probs` | Tüm K adayın başarı olasılıkları |
| `contact_ratio` | Temas oranı |
| `penetration_mm` | Ortalama penetrasyon |
| `displacement_cm` | Obje kayması |
| `latency_ms` | Inference süresi |
| `failure_category` | Başarısızlık türü |

---

## D11. İstatistiksel Raporlama Protokolü

HOT3D test seti yalnızca 4 held-out obje içerir; aynı objenin frame'leri birbirleriyle korelasyonludur, bu yüzden frame-level ortalama bağımsız örnek sayısını temsil etmez.

**Raporlama kuralları:**

- Her öğrenilmiş varyant **en az 3 random seed** ile eğitilir; sonuçlar `mean ± std` olarak verilir
- Unity success rate için **%95 bootstrap confidence interval** obje seviyesinde hesaplanır (frame seviyesinde değil)
- Model karşılaştırmaları aynı obje/senaryolar üzerinde yapıldığı için **paired bootstrap** kullanılır
- **Obje başına metrik** ve **kategori başına metrik** ayrıca raporlanır (macro average yanı sıra)
- Validation set üzerinden en iyi model seçilir; **test set yalnızca nihai değerlendirmede bir kez** kullanılır
- HOT3D için mümkünse object-level cross-validation (4-fold) uygulanır

---

## D12. Minimum Savunulabilir Deney Seti

Zaman kısıtı varsa bağımlılık sırasıyla en az şu deneyler yapılmalıdır:

1. **Birleşik arayüz testi** — OakInk T=1 ve HOT3D T=8 aynı modelden geçiyor
2. **BBox baseline vs PointNet-Grasp** — geometri bilgisinin katkısı (geodesic error + contact ratio)
3. **SingleFrame vs Temporal-GRU (T=8)** — temporal bilginin jitter ve stabiliteye etkisi
4. **Deterministic vs CVAE-K3** — çoklu aday üretiminin Unity success'e katkısı
5. **quality_score kalibrasyonu** — heuristic label ile spearman korelasyonu
6. **Unity success_prob kalibrasyonu** — AUC-ROC, ECE, reliability diagram
7. **Latency raporu** — GRU+self-attention+CVAE-K3 toplam < 10ms hedefi

Bu zincir tezin üç ana iddiasını savunur: geometri koşullu, temporal, çoklu aday.
