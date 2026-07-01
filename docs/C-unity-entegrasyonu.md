# C. Unity Entegrasyonu & Simülasyon Eval

Model çıktıları gerçek XR deneyimine dönüşmeden önce Unity tarafında iki işlem gerçekleşir: (1) tahmin edilen parmak açıları Unity XR Hands iskeletine uygulanır, (2) geçiş penceresi yönetimi ile model çıktıları blend edilir. Simülasyon tabanlı eval de bu ortamda çalışır.

Unity bu tezde ayrı bir AI model değildir. Unity'nin rolü runtime uygulama, fizik tabanlı değerlendirme ve confidence kalibrasyonu için başarı/başarısızlık ölçümüdür.

Sistem **PC üzerinde çalışır**; Quest yalnızca Air Link veya Link Cable üzerinden display/controller olarak bağlanır. Compute kısıtı yoktur.

---

## C1. XR Hands Runtime Bağlantısı

### XR Hands Mimarisi (Unity)

Unity XR Hands paketi, `XRHand` nesnesi üzerinden per-frame eklem verisi sağlar. Her eklem `XRHandJoint` olarak erişilir:

```csharp
XRHand hand = xrHandSubsystem.rightHand;
XRHandJoint wrist = hand.GetJoint(XRHandJointID.Wrist);
wrist.TryGetPose(out Pose wristPose);
```

AI modeli çıktısını uygulamak için bu varsayılan eklem verisi **override** edilmesi gerekir. Unity XR Hands, native override mekanizması sunmaz — bunun yerine:

### Override Yöntemi

**Yöntem A — Ayrı Avatar Rig (önerilen):**
- XR Hands verisi yalnızca controller pozisyonu/rotasyonu için kullanılır (bilek)
- Görsel avatar ayrı bir rig üzerinde çalışır (`SkinnedMeshRenderer` + kemik hierarchy)
- AI çıktısı doğrudan avatar kemiklerine uygulanır, fizik/algılama katmanı ayrı tutulur

**Yöntem B — XR Hands Retargeting:**
- `XRHandSkeletonDriver` bileşeni override edilerek AI çıktısı inject edilir
- Daha karmaşık ama XR Hands collision'ları da etkilenir

Thesis kapsamında Yöntem A tercih edilir: görsel doğruluk öncelikli, fizik simülasyonu ayrı eval sahnesinde yapılır.

### Eklem Rotasyonu Uygulama

Model çıktısı: 15 eklem × 3 axis-angle değeridir (`finger_aa45`). Unity tarafında bu değerler doğrudan `Quaternion.Euler` gibi uygulanmamalıdır; önce axis-angle → quaternion dönüşümü yapılmalı, sonra avatar rig'in local bone eksenlerine retarget edilmelidir.

```csharp
// Axis-angle → quaternion (near-zero norm guard gerekli)
Quaternion AxisAngleToQuaternion(Vector3 axisAngle)
{
    float angle = axisAngle.magnitude;
    if (angle < 1e-8f) return Quaternion.identity;
    return Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axisAngle / angle);
}

// Per-frame update
for (int i = 0; i < 15; i++)
{
    Transform joint = fingerJoints[i];
    Vector3 axisAngle = modelOutput.jointAxisAngle[i]; // radyan
    Quaternion q = AxisAngleToQuaternion(axisAngle);
    joint.localRotation = RetargetManoToRig(i, q);
}
```

**Model çıkış sırası (MANO `finger_aa45`):**
```
modelOutput[0..2]    → Index MCP
modelOutput[3..5]    → Index PIP
modelOutput[6..8]    → Index DIP
modelOutput[9..11]   → Middle MCP
modelOutput[12..14]  → Middle PIP
modelOutput[15..17]  → Middle DIP
modelOutput[18..20]  → Ring MCP
modelOutput[21..23]  → Ring PIP
modelOutput[24..26]  → Ring DIP
modelOutput[27..29]  → Little/Pinky MCP
modelOutput[30..32]  → Little/Pinky PIP
modelOutput[33..35]  → Little/Pinky DIP
modelOutput[36..38]  → Thumb CMC
modelOutput[39..41]  → Thumb MCP
modelOutput[42..44]  → Thumb IP
```

XR Hands'te parmak başına daha fazla/başka isimli eklem bulunduğu için bu sıra XR Hands sırasına birebir eşit değildir. Retarget katmanı MANO eklemlerini avatar rig kemiklerine açık bir tabloyla bağlamalıdır. Metacarpal eklemleri için varsayılan rotasyon tutulabilir veya sabit bir spread uygulanabilir.

### Runtime Model Girdi/Çıktı Sözleşmesi

Unity runtime için iki seviye arayüz vardır.

**MVP / mevcut statik model arayüzü:**

| Alan | Boyut | Unity'deki Kaynak |
|---|---:|---|
| `wrist_feat` | `(1, 6)` | Bilek konumu + global orientation axis-angle |
| `obj_pts` | `(1, 1024, 3)` | Seçili objenin cache'lenmiş point cloud'u |

Bu arayüz mevcut `src/grasp_model.py` ile uyumludur.

**Temporal hedef arayüzü:**

| Alan | Boyut | Unity'deki Kaynak |
|---|---:|---|
| `frame_feat` | `(1, T, 13)` | Son T frame bilek-obje relatif poz/rot/hız/mesafe |
| `finger_hist` | `(1, T, 45)` | Son T frame tracked veya önceki model parmak pozu |
| `obj_pts` | `(1, 1024, 3)` | Seçili objenin cache'lenmiş point cloud'u |
| `contact_flag` | `(1, T, 1)` | Collider/mesafe tabanlı temas sinyali |

Model çıktısı:

| Alan | Boyut | Kullanım |
|---|---:|---|
| `candidate_poses` | `(1, K, 45)` | CVAE ile üretilen K aday parmak pozu |
| `candidate_quality_scores` | `(1, K)` | Heuristik kalite skoru (MSE ile eğitilir) |
| `candidate_success_probs` | `(1, K)` | Unity binary başarı olasılığı (BCE, Aşama 3) |
| `selected_pose` | `(1, 45)` | Unity avatar rig'e uygulanacak poz |
| `selected_quality_score` | `(1, 1)` | Log/eval için kalite skoru |
| `selected_success_prob` | `(1, 1)` | UI/log/eval için başarı olasılığı |

İlk uygulama K=1 ile başlayabilir. CVAE aday seçimi eklendiğinde K=3 veya K=5 denenir.

### Model Çalıştırma (ONNX Runtime)

PyTorch modeli ONNX'e export edilir ve Unity'de **Sentis** (Unity'nin ML inference paketi) ile çalıştırılır.

```python
# Python — export
torch.onnx.export(model, dummy_input, "grasp_model.onnx", opset_version=17)
```

```csharp
// Unity — Sentis 2.x API (sürüme göre değişebilir)
var model  = ModelLoader.Load(Application.streamingAssetsPath + "/grasp_model.onnx");
var worker = new Worker(model, BackendType.GPUCompute);
worker.Schedule(inputTensor);
var output = worker.PeekOutput("pred") as Tensor<float>;
```

**Sürüm sabitlemesi (tez boyunca tutarlı olması gerekir):**

| Bileşen | Versiyon |
|---|---|
| Unity | TBD |
| XR Hands | TBD |
| Sentis | TBD |

Sentis API'si sürümler arasında breaking change içerebilir; kullanılan versiyon yukarıdaki tabloya yazılmalıdır.

Inference her Update() döngüsünde (~30–90 Hz) çağrılır. Latency bütçesi < 5ms model inference için.

---

## C2. Kullanıcı Eli → Grasp Düzeltmesi Geçiş Penceresi (Blending)

Ayrı bir Approach Model kullanılmaz. Kullanıcının takip edilen el/controller hareketinden Temporal Geometry-Conditioned Grasp Model'in parmak düzeltmesine geçiş ani değil, yumuşak (smooth) olmalıdır. Kullanıcı animasyonun "tutulduğunu" fark etmemeli.

### Geçiş Tetikleyicileri

```
free_motion = True   →  el–obje mesafesi > 10cm
transition  = True   →  2cm < mesafe ≤ 10cm
grasp_active = True  →  mesafe ≤ 2cm  VEYA  temas algılandı
```

### Blending Formülü

```
t = smoothstep(0, 1, (threshold_far - distance) / (threshold_far - threshold_near))
```

Parmak rotasyonları quaternion olarak blend edilir:
```csharp
// Axis-angle → quaternion dönüşümü yapıldıktan sonra:
joint.localRotation = Quaternion.Slerp(trackedRotation, predictedRotation, t);
```

Axis-angle vektörlerini doğrudan `lerp` etmek π sınırlarında artifact üretebilir; her eklem için `Slerp` kullanılmalıdır.

`smoothstep` lineer interpolasyona göre daha doğal (başlangıç ve bitişte hız sıfıra yaklaşır):
```
smoothstep(t) = 3t² - 2t³
```

### Model Activation

- Ana model transition başladığında inference üretmeye başlar
- Blend ağırlığı düşükken kullanıcı/el takip pozu baskındır
- Mesafe azaldıkça modelin parmak düzeltmesi baskın hale gelir

### Jitter Önleme (Geçiş Sırasında)

Bilek mesafesi gürültülüdür (Quest controller tracking ~1-2mm). Mesafe eşiğini geçip geri dönebilir → histerezis:
```
free → transition:      mesafe < 10cm için en az 5 frame
transition → grasp:     mesafe < 2cm için en az 3 frame
grasp → transition:     mesafe > 5cm  (hysteresis gap: 3cm)
```

---

## C3. Unity Fizik Simülasyonu Tabanlı Eval

Bu eval ortamı: (a) modelin gerçekten işe yarayıp yaramadığını ölçer, (b) güven skoru etiketleri için veri üretir.

### Eval Sahnesi Yapısı

```
EvalScene
├── HandRig (avatar kemikleri, collision ile)
├── ObjectSpawner (test objeleri)
├── ForceApplier (bozucu kuvvet)
├── EvalLogger (JSON çıktısı)
└── EvalOrchestrator (senaryoları sırayla çalıştırır)
```

### Senaryo Protokolü

Her test senaryosu:
1. Obje spawn edilir (belirlenen konumda)
2. Model inference çalışır → parmak açıları uygulanır
3. **0.5 saniye beklenir** (grasp stabilize olsun)
4. Bozucu kuvvet uygulanır: **F = α × m × g**, rastgele yönde, 0.1s süre
   - α = 1.0 (obje ağırlığının 1 katı) — hafif obje ve ağır obje aynı zorlukta
5. **1 saniye izlenir:**
   - `d_norm = |Δx| / bbox_diagonal < τ_d` AND `rotation_change < τ_r` AND `object_not_dropped` → **BAŞARILI**
   - Başlangıç eşikleri: **τ_d = 0.10** (bbox çapının %10'u), **τ_r = 15°** — calibration setinden fine-tune edilebilir, final test setinde sabit
   - Aksi → **BAŞARISIZ**

Normalize edilmiş yer değiştirme (`d_norm`) sabit eşik (5cm) yerine obje boyutuna göre ölçeklenir; büyük bir tabure ile küçük bir makas için aynı 5cm eşiğini kullanmak adil değildir.

### Friction Sensitivity Testi

Tüm test matrisi için değil, seçilmiş 5 obje üzerinde:

```
μ = 0.4, 0.6, 0.8
```

ile success rate karşılaştırılır. Sonuçlar μ=0.6 raporlama değeri ile sunulur.

### Collision Ayarları

- El parmaklarına `CapsuleCollider` (her parmak segmenti için)
- Objeye `MeshCollider` (convex) + `Rigidbody` (mass: per-object, annotation'dan)
- Baseline friction: μ=0.6

### Başarısızlık Analizi

Her başarısız senaryo kategorize edilir:

| Kategori | Tespit Yöntemi |
|---|---|
| **Penetrasyon** | Parmak kolliderleri mesh içinde |
| **Temassızlık** | Hiç temas olmadan kuvvet uygulandı |
| **Yanlış yön** | Bilek yönelimi GT'den >30° sapma |
| **Jitter** | Grasp uygulama sırasında eklem açısı std > eşik |

### Test Matrisi

| | Üstten | Yandan | Önden | Alttan |
|---|---|---|---|---|
| Kap/bardak | | | | |
| Araç (makas, kalem) | | | | |
| Elektronik (telefon) | | | | |
| Yuvarlak (top) | | | | |
| ...görülmemiş obje... | | | | |

Her hücre: başarı oranı (% başarılı / toplam deneme)

### Çıktı Format

```json
{
  "object_id": "mug_001",
  "test_direction": "top",
  "window_size": 16,
  "candidate_count": 3,
  "success": true,
  "quality_score": 0.82,
  "success_prob": 0.74,
  "contact_ratio": 0.8,
  "penetration_mm": 1.2,
  "displacement_cm": 0.8,
  "latency_ms": 4.7,
  "failure_category": null
}
```

---

## C4. Inference Latency Ölçümü

### PC GPU Ölçümü

```python
import time, torch

# Warm-up
for _ in range(10):
    model(dummy_input)

# Ölçüm
times = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model(dummy_input)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

print(f"Mean: {mean(times)*1000:.2f}ms, P95: {percentile(times,95)*1000:.2f}ms")
```

**Hedef:** Ana model < 5ms (PC GPU'da), CVAE ile K aday kullanılıyorsa toplam aday seçimi < 10ms

### Quest Streaming Latency (Air Link / Link Cable)

**Motion-to-photon gecikme bileşenleri:**
```
Controller tracking → PC (Air Link: ~7ms)
PC: Model inference (< 5ms)
PC: Unity render (< 11ms @ 90 Hz)
PC → Quest display (Air Link: ~7ms)
─────────────────────────────────
Toplam hedef: < 30ms
```

**Ölçüm yöntemi:**
- Quest'in controller pozisyonuna bilinen bir hareket uygulanır
- Kamera ile display'den alınan görüntü arasındaki gecikme yüksek hızlı kamera veya Unity'nin `Time.deltaTime` loglamasıyla ölçülür
- Air Link ve Link Cable karşılaştırılır

### Bottleneck Analizi

Eğer toplam latency > 30ms:
1. Ana model ONNX quantization (FP32 → FP16)
2. Batch size = 1 için özel CUDA kernel optimizasyonu
3. CVAE aday sayısını K=5'ten K=1-3 aralığına düşürme
4. Point cloud N boyutunu 1024'ten 512'ye düşürme
