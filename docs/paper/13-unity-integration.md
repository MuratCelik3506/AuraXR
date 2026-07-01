# 13. Unity Entegrasyonu ve Demo Scene

Kaynak: `docs/C-unity-entegrasyonu.md`, `docs/F-unity-demo-scene-plan.md`, `UnityDemo/`.

---

## 13.1. Sistem Mimarisi (Unity Tarafı)

Unity bu tezde ayrı bir AI modeli değildir. Rolleri:
1. Runtime uygulama: model çıktılarını XR avatar eline retarget et
2. Fizik tabanlı eval: success/fail ölçümü (henüz tamamlanmadı)
3. Confidence kalibrasyonu: success_label üretimi (henüz tamamlanmadı)

**Platform:** PC (GPU), Quest yalnızca Air Link / Link Cable ile display.

---

## 13.2. XR Hands Override Yöntemi

Unity XR Hands native override mekanizması sunmaz. Seçilen yöntem:

**Yöntem A — Ayrı Avatar Rig (uygulandı):**
- XR Hands verisi yalnızca bilek konumu/yönelimi için kullanılır
- Görsel avatar ayrı rig üzerinde (`SkinnedMeshRenderer` + bone hierarchy)
- AI çıktısı doğrudan avatar kemiklerine; fizik/algılama ayrı

---

## 13.3. MANO → XR Hands Retarget

Model çıktısı `finger_aa45` (45-dim, 15 eklem × 3 axis-angle):

```
0,1,2   → Index MCP        9,10,11  → Middle MCP
3,4,5   → Index PIP        12,13,14 → Middle PIP
6,7,8   → Index DIP        15,16,17 → Middle DIP
18,19,20 → Ring MCP        27,28,29 → Pinky MCP
21,22,23 → Ring PIP        30,31,32 → Pinky PIP
24,25,26 → Ring DIP        33,34,35 → Pinky DIP
36,37,38 → Thumb CMC       39,40,41 → Thumb MCP
42,43,44 → Thumb IP
```

**Axis-angle → Quaternion dönüşümü (near-zero guard şart):**
```csharp
Quaternion AxisAngleToQuaternion(Vector3 axisAngle) {
    float angle = axisAngle.magnitude;
    if (angle < 1e-8f) return Quaternion.identity;
    return Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axisAngle / angle);
}
```

**Dikkat:** XR Hands'te her parmakta metacarpal eklemi de var (26 eklem), MANO'da yok (21 nokta). Metacarpal için ya sabit spread ya da bilek-proximal arası interpolasyon.

---

## 13.4. Object-Relative Feature Hesabı (Unity Runtime)

**Feature sırası (kesin):** `rel_pos(3), rel_rot6d(6), rel_vel(3), dist(1)` — 13 dim

**Matematiksel tanım:**
```
R_rel   = R_object_world^-1 * R_wrist_world
rel_pos = R_object_world^-1 * (wrist_world_t - object_world_t)
rot6d   = [R_rel col0, R_rel col1] flattened
          = [R_rel[0,0], R_rel[1,0], R_rel[2,0], R_rel[0,1], R_rel[1,1], R_rel[2,1]]
rel_vel = (rel_pos_t - rel_pos_t-1) / delta_time
dist    = nearest surface/collider mesafe (metre)
```

**Kritik notlar:**
- Unity `Matrix4x4` erişiminde row/column ayrımı test edilmeli — Python'da kolonlar alınır, Unity'de satır almak hatalı feature üretir
- Normalizasyon: yalnızca `hot3d_canonical/stats.json` kullanılmalı
- `pts_std` 3-eksenli vektör, scalar değil
- Birimler: metre. Display'de cm göster, hesaplamada metre tut

**Ring buffer warm-up:** İlk 16 frame dolana kadar inference çalıştırılmaz. `prev_pose = zeros(45)`.

**rel_vel başlangıcı:** İlk frame `rel_vel = (0,0,0)`. Her frame için 1-frame finite diff, `delta_time` epsilon guard'lı.

**contact_flag:** `dist <= 0.02m OR physics_contact → 1.0`, aksi 0.0. Her frame için tutulur, pencere boyunca aynı değer kopyalanmaz.

---

## 13.5. Tracked → Model Pose Geçiş (Blending)

**Faz eşikleri:**

| Faz | Koşul |
|---|---|
| free motion | mesafe > 10 cm |
| transition | 2 cm < mesafe ≤ 10 cm |
| grasp active | mesafe ≤ 2 cm veya temas |

**Blend formülü:**
```
t = smoothstep(0, 1, (10cm - distance) / 8cm)
joint.localRotation = Quaternion.Slerp(trackedRotation, predictedRotation, t)
```
`smoothstep(t) = 3t² − 2t³` — lineer interpolasyondan daha doğal (uç noktalarda hız sıfırlanır).

**Histerezis (tracker jitter önlemi):**

| Geçiş | Koşul |
|---|---|
| free → transition | 5 frame boyunca < 10 cm |
| transition → grasp | 3 frame boyunca < 2 cm |
| grasp → transition | > 5 cm (3 cm hysteresis gap) |

---

## 13.6. ONNX Export — İki Versiyon

**1. Genel export (`export_onnx.py`):**
- Tam model, GRU op içerir, K aday, sampling içerir
- Python ONNX Runtime ile çalışır
- Unity InferenceEngine 2.6.1 ile **çalışmaz** (GRU op desteklenmiyor)

**2. Unity export (`export_unity_onnx.py`):**
- GRU 16 frame için manuel unroll (ONNX'te GRU op yok)
- CVAE z=zeros (deterministik, K=1, sampling yok)
- Candidate selection kaldırıldı
- Sabit shape, statik graph
- Unity InferenceEngine CPU backend üzerinde **çalışıyor**
- Çıktı: `checkpoints/grasp_model_unity.onnx`

**Kısıtlama:** Unity versiyonu K=1, deterministik mean pose. Araştırma K>1 için Python inference server gerekir.

---

## 13.7. Unity Demo Scene — Mevcut Durum

**Proje:** `/Users/muratcelik/Desktop/Thesis/Unity/AURAXR2`, scene: `Assets/Demo.unity`

**Kurulan bileşenler:**

| Bileşen | Sorumluluk | Durum |
|---|---|---|
| `AuraXRModelRuntime` | ONNX yükleme, Sentis inference, latency ölçümü | ✓ |
| `AuraXRFeatureAssembler` | Ring buffer, object-relative feature, normalizasyon | ✓ |
| `AuraXRHandRetargeter` | axis-angle → quat, MANO → rig bone mapping | ✓ |
| `AuraXRBlendController` | Mesafe bazlı tracked/predicted blend | ✓ |
| `AuraXRDemoHUD` | quality_score, success_prob, latency, distance, blend diagnostics | ✓ |
| `AuraXRDemoLogger` | JSONL event logging | ✓ |
| `AuraXRDemoSceneBuilder` | Editor helper | ✓ |

**Demo obje seti:** mug_white, bowl, Pot (XRI prefab)

**Keyboard debug:** WASD rig hareketi, Q/E aşağı/yukarı, I/K/J/L right wrist hareketi, U/O yukarı/aşağı

**Model bağlantısı:** `AuraXRModelRuntime.bypassModel=false`, backend CPU, `grasp_model_unity.onnx`

**Henüz tamamlanmayanlar:**
- Fizik eval scene (Unity success label üretimi)
- GPUCompute backend testi
- Quest Air Link / Link Cable latency testi

---

## 13.8. Unity Fizik Eval Protokolü (Planlandı)

**Senaryo akışı:**
1. Obje spawn edilir
2. Model inference → parmak açıları uygulanır
3. 0.5s stabilize beklenir
4. Bozucu kuvvet: `F = α × m × g`, α=1.0, rastgele yön, 0.1s
5. 1s gözlem

**Başarı kriteri:**
```
success = d_norm < τ_d  AND  rotation_change < τ_r  AND  object_not_dropped
τ_d = 0.10  (bbox çapının %10'u)       # UNITY_DISPLACEMENT_TAU = 0.10
τ_r = 15°                              # UNITY_ROTATION_TAU_DEG = 15.0
α = 1.0                                # UNITY_FORCE_ALPHA = 1.0
```

Eşikler calibration setinden belirlenir; final test setinde sabit tutulur.

**Failure categories:** penetrasyon / temassızlık / yanlış yön / jitter

**Friction sensitivity:** μ = 0.4, 0.6, 0.8 (seçilmiş 5 obje, baseline: μ=0.6)

---

## 13.9. Motion-to-Photon Latency Hedefi

```
Controller tracking → PC (Air Link: ~7ms)
PC: Model inference                (< 5ms hedef)
PC: Unity render                   (< 11ms @ 90 Hz)
PC → Quest display (Air Link: ~7ms)
─────────────────────────────────────────────────
Toplam hedef: < 30ms
```

**HOT3D training frame rate farkı:** HOT3D 30 FPS üzerinden eğitildi. Unity 72/90 FPS'te `rel_vel` değerleri eğitim distribüsyonundan küçük kalabilir. İlk demo için feature sampling 30 Hz'e sabitlenmesi veya `rel_vel` dağılımının HUD'dan izlenmesi önerilir.
