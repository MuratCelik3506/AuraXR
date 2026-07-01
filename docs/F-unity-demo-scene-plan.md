# F. Unity Demo Scene Model Test Planı

Bu doküman, açık olan Unity Demo scene üzerinde AuraXR grasp modelini test edebilmek için kurulacak sahne ve bileşen planıdır. Bu aşamada amaç tam fizik benchmark değil; modelin gerçek sahnede doğru input sözleşmesiyle çalıştığını, 45 boyutlu MANO parmak çıktısının rig'e uygulanabildiğini ve offline eval sonuçlarıyla tutarlı diagnostic üretildiğini görmek.

---

## F0. Mevcut Durum Özeti

### Model

Ana model `TemporalGeometryConditionedGraspModel`.

Girdi sözleşmesi:

| Alan | Boyut | Anlam |
|---|---:|---|
| `frame_feat` | `(1, T, 13)` | Object-relative bilek hareketi: `rel_pos(3), rel_rot6d(6), rel_vel(3), dist(1)` |
| `contact_flag` | `(1, T, 1)` | Temas / yaklaşma sinyali |
| `prev_pose` | `(1, 45)` | Önceki MANO finger axis-angle pozu |
| `obj_pts` | `(1, 1024, 3)` | Normalize edilmiş canonical obje point cloud |

Çıktı sözleşmesi:

| Alan | Boyut | Kullanım |
|---|---:|---|
| `selected_pose` | `(1, 45)` | Avatar/hand rig'e uygulanacak MANO parmak pozu |
| `quality_score` | `(1, 1)` | Heuristik kalite skoru; success ile karıştırılmamalı |
| `success_prob` | `(1, 1)` | Unity physics label ile kalibre edilecek başarı olasılığı |

Model export yolu:

```bash
python3 src/export/export_onnx.py \
  --checkpoint checkpoints/aura_phase2_best.pt \
  --out checkpoints/grasp_model.onnx \
  --window 16 \
  --k 1
```

Notlar:

- `--checkpoint checkpoints/aura_phase2_best.pt` açıkça verilmelidir; exporter'ın default fallback'i bu demo için kullanılmamalı.
- Mevcut `export_onnx.py` çıktı node'ları `selected_pose`, `quality_score`, `success_prob` olarak sabitliyor. Unity tarafında Sentis bu isimlerle okuma yapmalı.
- İlk demo `k=1` ile başlamalı. K>1 daha sonra denenirse `quality_score` head'in modelde fusion context için shared üretildiği ve adaylara broadcast edildiği unutulmamalı; aday ayrımı pratikte `success_prob` üzerinden gelir.

### Son Eval Bulguları

En güncel model dosyası `checkpoints/aura_phase2_best.pt`. Son genel overnight koşusu `2026-07-01 02:34:05` tamamlandı; ek OakInk val ölçümü `2026-07-01 09:40:24` dosyasında yer alıyor.

| Model / Split | Geodesic | MPJPE | Fingertip | Contact | Penetration | Quality |
|---|---:|---:|---:|---:|---:|---|
| Phase 1 OakInk val | 9.54 deg | 5.64 mm | 11.78 mm | 0.140 | 0.46 mm | AUC 0.944 |
| Phase 1 OakInk test | 9.72 deg | 5.73 mm | 12.02 mm | 0.131 | 0.48 mm | AUC 0.967 |
| Phase 2 HOT3D val | 11.70 deg | 6.06 mm | 13.77 mm | 0.146 | 4.02 mm | AUC 0.336 |
| Phase 2 HOT3D test | 12.58 deg | 6.62 mm | 14.92 mm | 0.230 | 1.36 mm | AUC 0.704 |
| Phase 2 OakInk val | 9.20 deg | 5.50 mm | 11.53 mm | 0.141 | 0.43 mm | AUC 0.949 |

Yorum:

- OakInk statik grasp kalitesi iyi ve stabil görünüyor; demo sahnede statik obje testleri anlamlı ilk hedef.
- HOT3D temporal model çalışıyor fakat penetration ve quality calibration özellikle val split'te zayıf. Demo sahnede `quality_score` yalnızca diagnostic olarak gösterilmeli, karar mekanizması tek başına buna bağlanmamalı.
- `success_prob` head henüz Unity physics success label ile kalibre edilmediği için UI'da "predicted success" olarak gösterilebilir ama bilimsel başarı metriği gibi raporlanmamalı.
- K=3/K=5 adaylı eval, HOT3D val üzerinde anlamlı iyileşme üretmemiş. İlk Unity demo K=1 ile kurulmalı; aday seçimi daha sonra eklenmeli.

---

## F1. Demo Scene Hedefi

Scene adı önerisi:

```text
AuraXR_ModelDemo.unity
```

Birinci hedef:

1. Seçili objeyi sahnede göster.
2. El/bilek pozunu obje frame'ine göre `frame_feat` ring buffer'a çevir.
3. ONNX modelden `selected_pose` al.
4. 45 boyutlu MANO axis-angle çıktısını avatar finger rig'e retarget et.
5. Mesafeye göre tracked pose ile model pose arasında smooth blend yap.
6. `quality_score`, `success_prob`, latency ve input diagnostics göster/logla.

Bu scene, `docs/C-unity-entegrasyonu.md` içindeki tam eval scene'in küçük ve etkileşimli karşılığıdır.

---

## F2. Scene Hiyerarşisi

Önerilen minimum hiyerarşi:

```text
AuraXR_ModelDemo
├── XR Origin / CameraRig
├── DemoTable
├── DemoObjectRoot
│   ├── ObjectSlot_Mug
│   ├── ObjectSlot_Box
│   └── ObjectSlot_Tool
├── HandRoot
│   ├── TrackedWristAnchor
│   └── PredictedHandRig
├── AuraXRModelRuntime
├── AuraXRFeatureAssembler
├── AuraXRHandRetargeter
├── AuraXRBlendController
├── AuraXRDemoHUD
├── AuraXRDemoLogger
└── [Debug] TestTools          ← build'e dahil edilmez; geliştirme sırasında aktif
    ├── AuraXRPlaybackMode
    ├── FeatureDriftMonitor
    ├── PerJointAngleMagnitudeBar
    ├── ReferencePoseOverlay
    ├── ObjectSwitcher
    ├── StatsOverrideTool
    └── PrevPoseInjector
```

### Demo Object Set

İlk kurulum için 3 obje yeterli:

| Obje | Amaç |
|---|---|
| Mug / cup | OakInk benzeri statik grasp için ana sanity check |
| Box / can | Basit power grasp |
| Tool-like elongated object | HOT3D tarzı yön ve temas hassasiyetini görmek |

Her obje için gerekli asset:

```text
object_id
mesh prefab
convex collider
canonical point cloud: 1024 x 3 .bytes/.json/.asset
normalization stats reference
bbox diagonal
```

Point cloud kaynağı Python preprocessing çıktılarındaki `.npy` dosyalarıdır. Unity doğrudan `.npy` okumak yerine build-time export edilmiş JSON/binary asset kullanmalı. Kaynak `.npy` dosyası 1024'ten farklı sayıda nokta içeriyorsa tam 1024 noktaya resampling Python tarafında, asset üretimi sırasında yapılmalı; Unity runtime'da point cloud resampling yapılmamalı.

---

## F3. Gerekli Unity Bileşenleri

### 1. `AuraXRModelRuntime`

Sorumluluk:

- `checkpoints/grasp_model.onnx` modelini yükler.
- Sentis input tensor'larını oluşturur.
- Output node'ları okur: `selected_pose`, `quality_score`, `success_prob`.
- Latency ölçer.

Input isimleri:

```text
frame_feat    shape (1, 16, 13)
obj_pts       shape (1, 1024, 3)
contact_flag shape (1, 16, 1)
prev_pose     shape (1, 45)
```

Output isimleri:

```text
selected_pose
quality_score
success_prob
```

İlk sürüm ayarları:

| Parametre | Değer |
|---|---:|
| window | 16 |
| n_points | 1024 |
| k | 1 |
| target FPS | 30-90 |
| inference budget | < 5 ms hedef, gerçek değer HUD'a yazılacak |

`prev_pose` başlangıç değeri:

- Ring buffer ilk kez dolarken ve önceki model pozu yokken `prev_pose` sıfır vektör olmalı: 45 float, MANO rest/canonical finger pose.
- İlk inference sonrasında `prev_pose`, bir önceki frame'in uygulanmış veya seçilmiş `selected_pose` değeriyle güncellenmeli.
- Rastgele veya uninitialized pose kesinlikle kullanılmamalı; self-attention token girdisine doğrudan giriyor.

### 2. `AuraXRFeatureAssembler`

Sorumluluk:

- Bilek world pose ve obje world pose bilgisinden object-relative feature üretir.
- Son 16 frame'i ring buffer'da tutar.
- `rel_vel` değerini frame delta ile hesaplar.
- `dist` değerini obje collider veya closest-point üzerinden hesaplar.
- `contact_flag` üretir ve ONNX'e `(1, T, 1)` olarak reshape eder.

Feature sırası kesinlikle:

```text
rel_pos(3), rel_rot6d(6), rel_vel(3), dist(1)
```

Kritik nokta:

- `checkpoints/aura_phase2_best.pt` HOT3D temporal fine-tuning sonrası modeldir. Unity runtime için normalization kaynağı `data/processed/hot3d_canonical/stats.json` olmalı.
- `data/processed/oakink_canonical/stats.json` bu demo runtime'da kullanılmamalı; OakInk stats ile HOT3D-trained model sessizce anlamsız inference üretebilir.
- `input_mean` ve `input_std` 13 elemanlı vektördür; `frame_feat[i] = (raw_feat[i] - input_mean[i]) / input_std[i]`.
- `pts_mean` ve `pts_std` 3 elemanlı per-axis vektördür; `obj_pts[j, axis] = (raw_pts[j, axis] - pts_mean[axis]) / pts_std[axis]`. Tek scalar std kullanılmamalı.
- Birimler metre olmalı. `dist` metre cinsinden normalize edilmeli; HUD'da cm gösterilecekse sadece display aşamasında çevrilmeli.
- `contact_flag` binary olmalı: `1.0` veya `0.0`.

Object-relative feature tanımı:

```text
R_rel   = R_object_world^-1 * R_wrist_world
rel_pos = R_object_world^-1 * (wrist_world_t - object_world_t)
rot6d   = first two columns of R_rel, flattened as:
          [R_rel[0,0], R_rel[1,0], R_rel[2,0],
           R_rel[0,1], R_rel[1,1], R_rel[2,1]]
rel_vel = (rel_pos_t - rel_pos_t-1) / delta_time
dist    = nearest wrist/object surface distance in meters
```

Unity `Matrix4x4` erişiminde row/column ayrımı test edilmeli. Python tarafındaki `axis_angle_to_rot6d` ilk iki rotasyon kolonunu kullanır; Unity'de satırları almak hatalı feature üretir.

`rel_vel` başlangıç kuralı:

- İlk frame için `rel_vel = (0, 0, 0)`.
- Ring buffer dolana kadar her yeni frame'de bir önceki `rel_pos` ile 1-frame finite difference kullanılır.
- `delta_time` Unity frame zamanı olmalı; sıfıra bölmeyi engellemek için küçük epsilon guard eklenmeli.
- Pencere içinde her frame'in kendi `rel_vel` değeri saklanmalı, sadece son frame'in hızı tüm pencereye kopyalanmamalı.

`contact_flag` kuralı:

- Başlangıç eşiği `contact_threshold_m = 0.02` olmalı; bu değer Blend Controller'daki `grasp active` 2 cm eşiğiyle aynı.
- `contact_flag_t = 1.0` if `dist_t <= contact_threshold_m`, aksi halde `0.0`.
- Collider gerçek teması daha erken bildirirse `contact_flag_t = 1.0` yapılabilir; final kural `dist_t <= 0.02 OR physics_contact`.
- Ring buffer içinde her frame'in kendi `contact_flag` değeri tutulmalı; sadece son frame değeri tüm pencereye kopyalanmamalı.

Ring buffer warm-up kuralı:

- İlk demo davranışı: 16 frame dolana kadar model inference çalıştırılmamalı.
- HUD `active_window_fill = 0..16` ve `model_state = warming_up` göstermeli.
- Buffer dolana kadar hand rig tracked/default pose'ta kalmalı, `prev_pose = zeros(45)` korunmalı.
- 16 frame dolduktan sonra inference başlar ve `model_state = running` olur.
- Alternatif zero-padding HOT3D dataset'te bazı shifted window'lar için kullanılıyor; Unity demo ilk sürümünde zero-padded partial window kullanılmamalı. Bu, başlangıçta daha tahmin edilebilir debugging sağlar.

### 3. `AuraXRHandRetargeter`

Sorumluluk:

- `selected_pose[45]` axis-angle değerlerini quaternion'a çevirir.
- MANO joint sırasını avatar rig kemiklerine açık tabloyla bağlar.
- Per-joint rest rotation ve axis correction uygular.

MANO sıra:

```text
0,1,2       index MCP
3,4,5       index PIP
6,7,8       index DIP
9,10,11     middle MCP
12,13,14    middle PIP
15,16,17    middle DIP
18,19,20    ring MCP
21,22,23    ring PIP
24,25,26    ring DIP
27,28,29    pinky MCP
30,31,32    pinky PIP
33,34,35    pinky DIP
36,37,38    thumb CMC
39,40,41    thumb MCP
42,43,44    thumb IP
```

İlk test için hedef:

- Model output pozu avatar üzerinde anatomik olarak ters dönmeden görünmeli.
- Her eklem için axis correction sahnede inspector üzerinden ayarlanabilir olmalı.

### 4. `AuraXRBlendController`

Sorumluluk:

- Tracked hand pose ile predicted hand pose arasında mesafe bazlı geçiş uygular.

Başlangıç eşikleri:

| Faz | Koşul |
|---|---|
| free motion | distance > 10 cm |
| transition | 2 cm < distance <= 10 cm |
| grasp active | distance <= 2 cm veya contact |

Blend:

```text
t = smoothstep((10cm - distance) / 8cm)
jointRotation = Slerp(trackedRotation, predictedRotation, t)
```

Histerezis:

| Geçiş | Koşul |
|---|---|
| free -> transition | 5 frame boyunca distance < 10 cm |
| transition -> grasp | 3 frame boyunca distance < 2 cm |
| grasp -> transition | distance > 5 cm |

### 5. `AuraXRDemoHUD`

HUD yalnızca diagnostic göstermeli:

```text
object_id
distance_cm
blend_weight
quality_score
success_prob
latency_ms
contact_flag
active_window_fill
```

`quality_score` ve `success_prob` ayrı satırlar olmalı. Bu iki değer UI veya log formatında birleştirilmemeli.

K>1 notu:

- Mevcut modelde `quality_score` aday başına farklı hesaplanmıyor; context-level değer adaylara broadcast ediliyor.
- K adaylı demo eklendiğinde HUD, aday seçimini `success_prob` üzerinden açıklamalı; `quality_score` aday çeşitliliği göstergesi gibi yorumlanmamalı.

### 6. `AuraXRDemoLogger`

Her deneme için JSON Lines önerisi:

```json
{
  "timestamp": "...",
  "object_id": "mug_white",
  "window_size": 16,
  "active_window_fill": 16,
  "distance_cm": 3.4,
  "blend_weight": 0.72,
  "contact_flag": 1,
  "quality_score": 0.81,
  "success_prob": 0.57,
  "latency_ms": 4.3,
  "last_frame_feat": [13 floats],
  "full_window_frame_feat": [[13 floats] x 16],
  "selected_pose": [45 floats],
  "joint_axis_angle_magnitudes": [15 floats],
  "event": "grasp_start"
}
```

Alan açıklamaları:

- `last_frame_feat`: Normalize edilmiş son frame input'u. `rel_pos(3)`, `rel_rot6d(6)`, `rel_vel(3)`, `dist(1)` sırasıyla.
- `full_window_frame_feat`: Modele giren tam 16-frame penceresi. Offline Python ile aynı sekansı tekrar çalıştırmak ve Unity/Python çıktılarını karşılaştırmak için.
- `joint_axis_angle_magnitudes`: `selected_pose` içindeki 15 eklem için `norm(axis_angle[j*3:(j+1)*3])` değerleri (radyan). Anatomik saçmalık tespiti için.
- `event`: Yalnızca durum geçişlerinde yazılır: `"grasp_start"`, `"grasp_end"`, `"object_switch"`, `"warmup_complete"`. Her frame'de yazılmaz.

Bu log fizik başarı metriği değildir; scene-level integration debug kaydıdır.

#### Log yoğunluğu

Loglar üç seviyede tutulmalı; runtime'da Inspector'dan seçilebilir:

| Seviye | İçerik | Kullanım |
|---|---|---|
| `minimal` | timestamp, object_id, quality_score, success_prob, latency_ms, event | Hızlı sanity check |
| `diagnostic` | `minimal` + last_frame_feat, selected_pose, joint_magnitudes, blend_weight, distance_cm | Standart debug oturumu |
| `full` | `diagnostic` + full_window_frame_feat | Python karşılaştırması ve offline replay |

`full` mod her frame yazarsa disk hızlı dolar; yalnızca `grasp_start` anında veya manuel trigger ile anlık snapshot alınması önerilir.

#### Zorunlu log noktaları

Aşağıdaki olaylarda her log seviyesinde kayıt oluşturulmalı:

- Model ilk kez inference yaptığında (`warmup_complete`)
- `grasp_start` / `grasp_end` geçişlerinde
- Obje değiştirildiğinde (`object_switch`)
- Latency `10 ms` üzerine çıktığında (`latency_spike`)
- Herhangi bir eklemde `joint_axis_angle_magnitude > π` (`pose_anomaly`)

---

## F3b. Test Kolaylığı Bileşenleri

Bu bileşenler zorunlu değildir; ancak geliştirme ve debug sürecini önemli ölçüde hızlandırır.

### 7. `AuraXRPlaybackMode`

Gerçek XR donanımı olmadan tekrarlanabilir test yapabilmek için:

- HOT3D veya kayıtlı bir Unity oturumundan alınan `frame_feat` sekansını `.json` asset olarak sahneye ekle.
- Inspector'daki `PlaybackMode` toggle açıkken `AuraXRFeatureAssembler` tracked el yerine bu dosyadan okur.
- Aynı girdiyle her çalıştırmada aynı model çıktısı beklenir; retarget ve blend doğruluğu tekrarlanabilir şekilde test edilir.
- Kayıtlı sekans `full_window_frame_feat` JSONL kaydından doğrudan üretilebilir.

### 8. `FreezeFrame`

- Spacebar ile `AuraXRFeatureAssembler` dondurulur, son inference çıktısı sabit tutulur.
- Retarget sonucunu XR içinde rahatça incelemek için; her seferinde elinizi hareket ettirme zorunluluğu kalkar.

### 9. `FeatureDriftMonitor`

HUD'a ek olarak veya ayrı bir debug panel olarak:

- `rel_pos`, `rel_vel`, `dist` değerlerinin normalize edilmiş karşılıklarını gerçek zamanlı grafik olarak göster.
- Normalize değer `[-3, 3]` aralığının dışına çıkıyorsa kırmızı uyarı: normalization hatası veya birim sorununun anlık göstergesi.
- `contact_flag` geçmişini son 16 frame için bar olarak göster.

### 10. `PerJointAngleMagnitudeBar`

- `selected_pose[45]`'ten hesaplanan 15 eklem `norm(axis_angle)` değerini renk kodlu bar olarak HUD'da göster.
- `0` → mavi (rest), `π/2` → sarı, `>π` → kırmızı (anatomik sınır aşımı uyarısı).
- Hangi eklemin saçma değer ürettiği sayısal karşılaştırma olmadan görsel olarak ayırt edilir.

### 11. `ReferencePoseOverlay`

- Inspector'dan veya `.json` dosyasından yüklenmiş bilinen iyi bir `selected_pose[45]` değerini wire-frame olarak avatar üzerine bindirerek göster.
- Retarget hatası sayısal analiz gerekmeden görsel olarak tespit edilebilir hale gelir.
- Referans poz HOT3D veya OakInk eval çıktısından alınabilir.

### 12. `ObjectSwitcher`

- `1/2/3` tuşlarıyla `mug / box / tool` geçişi.
- Her geçişte point cloud, normalization stats referansı, `object_id` ve mesh otomatik değişmeli.
- F5 manuel test protokolünü tek tuşla tekrarlamak için; geçiş anında `object_switch` eventi loglanmalı.

### 13. `StatsOverrideTool`

- Inspector'dan normalization stats setini runtime'da `hot3d ↔ oakink` olarak değiştir.
- İki stats setinin model çıktısını nasıl etkilediğini sahne içinde gözlemlemek için.
- F6 §3'teki yanlış stats riski bu araçla sayısal olarak gösterilebilir.

### 14. `PrevPoseInjector`

- Inspector butonu ile `prev_pose`'u sıfırla veya önceden kaydedilmiş bir pozla doldur.
- Warm-up beklenmeden belirli bir başlangıç koşulunu test etmek için.
- Kaydedilmiş poz `selected_pose` JSONL çıktısından kopyalanabilir.

---

## F4. Kurulum Sırası

### Aşama 1 - Asset ve Model Hazırlığı

1. `checkpoints/aura_phase2_best.pt` üzerinden ONNX export al.
2. ONNX'i Unity `StreamingAssets` altına koy.
3. `data/processed/hot3d_canonical/stats.json` içindeki `input_mean`, `input_std`, `pts_mean`, `pts_std` değerlerini Unity-readable asset'e çevir.
4. Demo objeleri için `obj_pts` point cloud asset'lerini üret; kaynak nokta sayısı 1024 değilse Python preprocessing/export scriptiyle tam `(1024, 3)` resample et.
5. Hızlı doğrulama için Sentis'ten önce Python inference server seçilirse `src/unity/unity_contract.py` içindeki `UnityInferRequest` / `UnityInferResponse` şemaları kullanılmalı; yeni JSON şeması icat edilmemeli.

Başarı kriteri:

- Unity modeli yükler.
- Dummy input ile `selected_pose[45]`, `quality_score`, `success_prob` okunur.
- Dummy input shape'leri kesin doğrulanır: `frame_feat(1,16,13)`, `obj_pts(1,1024,3)`, `contact_flag(1,16,1)`, `prev_pose(1,45)`.

### Aşama 2 - Scene Wiring

1. Demo table ve 3 obje slotu kur.
2. Her obje prefab'ına `object_id`, mesh, collider ve point cloud reference ekle.
3. `TrackedWristAnchor` ile obje arasından `frame_feat` üret.
4. Ring buffer'ın 16 frame dolma durumunu HUD'da göster.

Başarı kriteri:

- Bilek hareket ettikçe `rel_pos`, `rel_vel`, `dist`, `contact_flag` stabil değişir.
- Unity feature order Python sözleşmesiyle birebir eşleşir.
- `contact_flag` tek değer olarak değil, pencere boyunca `(T,1)` dizi olarak tutulur ve ONNX'e `(1,T,1)` verilir.
- `rel_rot6d` için Unity'de üretilen ilk iki kolon değeri Python referans hesapla karşılaştırılır.
- İlk 15 frame boyunca inference bekler; 16. frame sonrası `model_state = running` olur.
- `AuraXRPlaybackMode` ile kaydedilmiş bir sekans beslendiğinde Unity çıktısı Python offline inference ile numerik olarak yakın (`selected_pose` farkı `< 1e-3`) gelmelidir.

### Aşama 3 - Retarget ve Blend

1. `selected_pose` axis-angle değerlerini quaternion'a dönüştür.
2. MANO -> rig bone mapping tablosunu inspector'da doldur.
3. Rest pose correction ayarla.
4. Mesafe bazlı smooth blend'i devreye al.

Başarı kriteri:

- Obje uzaktayken kullanıcı/tracked hand baskın.
- Objeye yaklaşırken model pozu kademeli baskın hale gelir.
- Parmaklarda ani flip, ters kapanma veya jitter görünmez.

### Aşama 4 - Demo Diagnostics

1. HUD değerlerini ekle.
2. JSONL logging ekle.
3. 3 obje için kısa manuel test protokolü uygula.

Başarı kriteri:

- Her objede inference latency ölçülür.
- Kalite ve başarı olasılığı ayrı loglanır.
- Demo kaydı offline eval ile karşılaştırılabilecek kadar izlenebilir olur.

---

## F5. Manuel Test Protokolü

Her obje için:

1. Obje seç.
2. El/bileği 20 cm uzakta tut; model blend ağırlığı 0 olmalı.
3. 10 cm eşiğine yaklaş; transition başlamalı.
4. 2 cm / contact eşiğinde model pose baskın olmalı.
5. 3 farklı yönden yaklaş: üstten, yandan, önden.
6. Her yaklaşım için HUD ve JSONL değerlerini kontrol et.

Minimum kabul:

| Kontrol | Beklenen |
|---|---|
| Tensor shape | Hatasız |
| Latency | Ölçülüyor; hedef < 5 ms |
| Retarget | Anatomik olarak makul parmak kapanışı |
| Blend | Ani sıçrama yok |
| Logging | `quality_score` ve `success_prob` ayrı |
| Object switch | Point cloud ve mesh aynı `object_id` ile değişiyor |
| Playback karşılaştırması | Unity `selected_pose` ≈ Python offline çıktısı |
| Feature drift | `FeatureDriftMonitor`'da normalize değerler `[-3, 3]` içinde |
| Joint magnitude | `PerJointAngleMagnitudeBar`'da `>π` olan eklem yok |
| Log event coverage | `grasp_start`, `warmup_complete`, `object_switch` eventleri JSONL'de görünüyor |

---

## F6. Riskler ve Dikkat Noktaları

1. Unity proje dosyaları bu workspace'te görünmüyor.
   - Git durumunda eski `Unity/` ve `unity/` C# dosyaları silinmiş görünüyor.
   - Uygulama aşamasında açık Unity project path'i doğrulanmalı.

2. Coordinate mismatch en büyük risk.
   - Python model object-relative feature bekliyor.
   - Unity'de world -> object dönüşümü, rot6d kolon sırası ve metre birimi doğrulanmalı.
   - `rot6d` satır-major flatten değildir; `R_rel` matrisinin ilk iki kolonu sırayla alınır.

3. Normalization eksik olursa model çıktısı anlamsızlaşır.
   - Phase 2 demo için yalnızca `data/processed/hot3d_canonical/stats.json` kullanılmalı.
   - `pts_std` scalar değil, 3 eksenli vektördür.

4. Retarget doğrudan `Quaternion.Euler` ile yapılmamalı.
   - Model axis-angle radyan üretir; önce axis-angle -> quaternion dönüşümü gerekir.

5. `success_prob` henüz gerçek Unity success label ile kalibre edilmiş final metrik değildir.
   - Demo HUD'da gösterilebilir, fakat tez sonucu olarak yorumlanmamalı.

6. HOT3D val penetration yüksek.
   - Demo sırasında görsel penetration ve collider penetration ayrı izlenmeli.

7. Unity frame rate ile HOT3D temporal istatistiği farklı olabilir.
   - HOT3D temporal training pratikte 30 FPS dağılımına göre `rel_vel` görür.
   - Unity 72/90 FPS'te 1-frame finite difference kullanırsa normalize edilmiş `rel_vel` büyüklüğü eğitim dağılımından daha küçük kalabilir.
   - İlk demo için en güvenli seçenek feature sampling'i 30 Hz'e sabitlemek veya `rel_vel` hesabında `delta_time` ile m/s üretip HUD'da dağılımı izlemektir.
   - Eğer görsel output hızdan aşırı etkilenirse geçici fallback olarak 30 Hz feature update loop kullanılmalı; render FPS ayrı kalabilir.

---

## F7. Uygulama Başlamadan Önce Beklenen Kararlar

Devam etmeden önce netleşmesi gerekenler:

1. Açık Unity projesinin gerçek klasörü bu repo içinde mi, yoksa ayrı bir workspace'te mi?
2. Demo scene mevcut bir `.unity` dosyası üzerinde mi düzenlenecek, yoksa `AuraXR_ModelDemo.unity` yeni scene olarak mı oluşturulacak?
3. İlk runtime yolu Sentis/ONNX mi olacak, yoksa hızlı doğrulama için Python inference server mı kullanılacak?
4. Kullanılacak hand rig hazır mı, yoksa ilk test için basit debug bone rig mi oluşturulacak?

Önerilen başlangıç kararı:

- Yeni `AuraXR_ModelDemo.unity` scene.
- İlk runtime: Sentis/ONNX.
- İlk model: `checkpoints/aura_phase2_best.pt` export, `window=16`, `k=1`.
- İlk obje seti: mug/cup, box/can, elongated tool.
- İlk hedef: görsel retarget + diagnostic logging; fizik success eval sonraki aşama.

---

## F8. Uygulama Durumu

Başlatılan uygulama artefactleri:

```text
checkpoints/grasp_model.onnx
UnityDemo/README.md
UnityDemo/Assets/AuraXR/Scripts/AuraXRDemoTypes.cs
UnityDemo/Assets/AuraXR/Scripts/AuraXRFeatureAssembler.cs
UnityDemo/Assets/AuraXR/Scripts/AuraXRModelRuntime.cs
UnityDemo/Assets/AuraXR/Scripts/AuraXRHandRetargeter.cs
UnityDemo/Assets/AuraXR/Scripts/AuraXRBlendController.cs
UnityDemo/Assets/AuraXR/Scripts/AuraXRDemoHUD.cs
UnityDemo/Assets/AuraXR/Scripts/AuraXRDemoLogger.cs
UnityDemo/Assets/AuraXR/Editor/AuraXRDemoSceneBuilder.cs
UnityDemo/Assets/StreamingAssets/AuraXR/grasp_model.onnx
UnityDemo/Assets/StreamingAssets/AuraXR/model_stats.json
UnityDemo/Assets/StreamingAssets/AuraXR/objects_manifest.json
UnityDemo/Assets/StreamingAssets/AuraXR/objects/*.bytes
src/unity/export_demo_assets.py
```

Doğrulananlar:

- `checkpoints/aura_phase2_best.pt` ONNX'e export edildi.
- ONNX Runtime smoke test geçti: output node'ları `selected_pose`, `quality_score`, `success_prob`; shape'ler `(1,45)`, `(1,1)`, `(1,1)`.
- HOT3D stats ve üç demo obje point cloud'u Unity-readable StreamingAssets formatına export edildi.
- `src/unity/export_demo_assets.py` Python compile check'ten geçti.
- Aktif Unity projesi bulundu: `/Users/muratcelik/Desktop/Thesis/Unity/AURAXR2`, aktif scene `Assets/Demo.unity`.
- `AuraXR_Stepwise_Demo` hiyerarşisi kuruldu: environment, XR rig, gerçek mesh objeler ve ayrı model componentleri.
- XR Interaction Toolkit `XR Origin Hands (XR Rig)` prefab'i sahneye eklendi.
- Controller visual'ları ve desktop/simulator fallback hareket component'i eklendi.
- Model-driven görünür virtual right hand rig oluşturuldu ve `AuraXRHandRetargeter` içindeki 15 MANO bone slotuna bağlandı.
- Placeholder objeler yerine gerçek mesh/prefab kullanıldı: `mug_white.obj`, `bowl.obj`, XRI `Pot.prefab`.
- `AuraXRFeatureAssembler` aktif obje olarak `mug_white` ve wrist kaynağı olarak right controller transform'una bağlandı.

Unity Editor notu:

- Unity MCP bağlantısı tekrar açıldıktan sonra scene otomatik değiştirildi ve kaydedildi.
- Save sırasında mevcut OpenXR package warning'i görülüyor; AuraXR compile/runtime wiring hatası değildir.
- Unity InferenceEngine paketi mevcut: `Unity.InferenceEngine 2.6.1`.
- İlk generic `grasp_model.onnx` import edilemedi; sebep Unity InferenceEngine'in ONNX `GRU` operator'ını desteklememesi ve generic export içinde sampling/candidate-selection dinamiklerinin bulunmasıydı.
- `src/export/export_unity_onnx.py` eklendi. Bu exporter Unity için deterministik, sabit shape'li ONNX üretir:
  - CVAE runtime randomness kaldırıldı (`z=zeros`).
  - Candidate selection / advanced indexing kaldırıldı.
  - GRU 16 frame için elle unroll edildi; ONNX graph içinde `GRU` op kalmadı.
- `checkpoints/grasp_model_unity.onnx` üretildi ve Unity projesinde `Assets/AuraXR/Models/grasp_model_unity.onnx` olarak `Unity.InferenceEngine.ModelAsset` import edildi.
- Unity InferenceEngine CPU smoke test geçti: `selected_pose`, `quality_score`, `success_prob` outputları okundu.
- Scene runtime artık gerçek model assetine bağlı: `AuraXRModelRuntime.bypassModel=false`, backend `CPU`.

Kullanım durumu:

- Scene artık MetaXR/desktop içinde görülebilir ve hareket test edilebilir.
- `WASD` rig hareketi, `Q/E` rig aşağı/yukarı, `I/K/J/L` right wrist/controller hareketi, `U/O` right wrist/controller yukarı/aşağı için eklendi.
- Feature assembly, warm-up, HUD/logging, object-distance/contact ve virtual hand retarget wiring test edilebilir.
- Model pose uygulaması artık Unity InferenceEngine üzerinden gerçek `grasp_model_unity.onnx` outputu kullanır.
- GPUCompute backend sonra denenebilir; ilk doğrulanmış backend CPU'dur.
