# AuraXR Unity — Kurulum Kılavuzu

> Bu kılavuz Unity projesini sıfırdan kurmak için tüm adımları içerir. Proje zaten yapılandırılmış durumdadır; bu belge referans ve sorun giderme amacıyla tutulmaktadır.

---

## KRİTİK KURULUM SIRASI

```
1.  Unity projesi oluştur + platformu Android'e geçir
2.  Paketleri yükle (Sentis, Oculus, XRI, TMP)
3.  Tüm betikleri + model dosyalarını Assets'e kopyala
4.  Özel katmanları oluştur (herhangi bir sahne nesnesi eklemeden önce)
5.  Katman Çarpışma Matrisini ayarla
6.  OVRInteractionComprehensive prefab'ını ekle
7.  OVRCameraRig'e CharacterController ekle
8.  OVRCameraRig'e ThumbstickLocomotion ekle
9.  Sahne ortamını oluştur (Floor, Walls, Table) — hemen Environment katmanını ata
10. Bottle ve Cup ekle — hemen Interactable katmanını ata
11. GameManager ekle + tüm betik bileşenlerini ekle
12. GameManager'a AuraXRAutoWire ekle — tüm Inspector referanslarını otomatik bağlar
13. LeftHandRig ve RightHandRig oluştur — HandRig katmanını ata
14. Materyaller: el = Fade rendering, Bottle/Cup = Standard + Emission AÇIK
15. TaskCanvas oluştur (World Space)
16. Play'e bas ve Console'u doğrula

NOT: AuraXRAutoWire tüm Inspector çapraz referanslarını Awake'te otomatik bağlar.
     Manuel Inspector ataması gerekmez — betik çalıştıktan sonra kendini siler.
```

---

# FAZ 1: PROJE KURULUMU

## Bölüm 1.1: Unity Projesi Oluştur

1. **Unity Hub** aç → **New Project**
2. Şablon: **3D (Built-in RP)**
3. İsim: `AuraXR_HandPose_V1`
4. Konum: `/Users/muratcelik/Desktop/Thesis/Unity_Projects/AuraXR_HandPose_V1`
5. Unity sürümü: **2023.2.20f1**
6. **Create Project** (2–3 dk bekle)

### Platformu Android'e Geçir

1. **File → Build Settings**
2. **Android** seç → **Switch Platform** (1–2 dk bekle)

### Paketleri Yükle

**Window → Package Manager → + → Add package from git URL**

Sırayla ekle:
1. `com.unity.sentis`
2. `com.unity.xr.plugin.oculus`
3. `com.unity.xr.interaction.toolkit`
4. `com.unity.textmesh-pro`

TMP yüklendikten sonra: **Window → TextMesh Pro → Import TMP Essentials**

### XR Plugin Yapılandır

1. **Edit → Project Settings → XR Plugin Management → Android sekmesi**
2. **Oculus** ✓ işaretle → yeniden derlemeyi bekle

### Klasör Yapısı Oluştur

**Project** sekmesi → boş alana sağ tıkla → her biri için **Create Folder**:

```
Assets/AuraXR/Scripts
Assets/AuraXR/Models
Assets/AuraXR/Prefabs
Assets/AuraXR/Rigs
Assets/AuraXR/Materials
Assets/AuraXR/UI
Assets/AuraXR/Data
```

---

## Bölüm 1.2: Model Dosyalarını İçe Aktar

`/Users/muratcelik/Desktop/Thesis/Workspace/V3/data/` konumundan kopyala:
- `intentformer.onnx`
- `intentformer_meta.json`

`Assets/AuraXR/Models/` içine yapıştır.

**Project** sekmesinde → `intentformer_meta.json`'a tıkla → Inspector **TextAsset** olduğunu doğrular → **Apply**.

---

# FAZ 2: BETIKLER & ÇIKARIM PIPELINE'I

## Bölüm 2.1: Tüm Betikleri Kopyala

`UnityScripts/` içindeki tüm dosyaları `Assets/AuraXR/Scripts/` konumuna kopyala:

```
AuraXRMetaLoader.cs
AuraXRFeatureAssembler.cs
AuraXRInferenceManager.cs
HandRigController.cs
InteractableObject.cs
ProximityDetector.cs
HandVisibilityController.cs
ThumbstickLocomotion.cs
GraspIndicator.cs
VirtualHandGrab.cs
HandProximityVisibility.cs
SessionDataLogger.cs
ScenarioKitchenTask.cs
HapticFeedbackManager.cs
UITaskDisplay.cs
SoundManager.cs
ConditionManager.cs
```

Doğrula: **Console**'da kırmızı ikon yok.

---

## Bölüm 2.2: GameManager Oluştur

1. **Hierarchy** → sağ tıkla → **Create Empty** → isim: `GameManager`
2. `GameManager` seç → aşağıdakilerin her biri için **Add Component**:

```
AuraXRMetaLoader
AuraXRFeatureAssembler
AuraXRInferenceManager
ProximityDetector
HandProximityVisibility
VirtualHandGrab
SessionDataLogger
ScenarioKitchenTask
HapticFeedbackManager
SoundManager
ConditionManager
```

### Inspector Atamaları

| Bileşen | Alan | Değer |
|---------|------|-------|
| AuraXRMetaLoader | Meta Json | `intentformer_meta.json` |
| AuraXRInferenceManager | Model Asset | `intentformer.onnx` |
| AuraXRInferenceManager | Meta Loader | `GameManager` |
| AuraXRInferenceManager | Feature Assembler | `GameManager` |
| AuraXRInferenceManager | Inference Every N Frames | `2` |
| AuraXRFeatureAssembler | Left Controller Transform | `LeftControllerAnchor` ← Faz 6.1'den sonra ata |
| AuraXRFeatureAssembler | Right Controller Transform | `RightControllerAnchor` ← Faz 6.1'den sonra ata |
| ProximityDetector | Feature Assembler | `GameManager` |
| ProximityDetector | Search Radius | `2.0` |
| VirtualHandGrab | Left Hand Wrist | `LeftHandRig` ← Faz 4'ten sonra ata |
| VirtualHandGrab | Right Hand Wrist | `RightHandRig` ← Faz 4'ten sonra ata |
| VirtualHandGrab | Grab Radius | `0.15` |
| VirtualHandGrab | Grip Threshold | `0.7` |
| VirtualHandGrab | Throw Multiplier | `1.5` |
| SessionDataLogger | Inference Manager | `GameManager` |
| SessionDataLogger | Feature Assembler | `GameManager` |
| SessionDataLogger | Enable Logging | ✓ |

---

## Bölüm 2.3: Play Mode Testi

**Play** (▶) tuşuna bas. Console şunu göstermeli:
```
[AuraXR] Meta loaded. Feature=96  Target=78  T=16
[AuraXR] Model loaded. Input: [1,16,96]  Output: [1,78]
```

Göstermiyorsa:
- `metaJson not assigned` → `intentformer_meta.json`'ı AuraXRMetaLoader'a sürükle
- `modelAsset not assigned` → `intentformer.onnx`'i AuraXRInferenceManager'a sürükle
- `NullReferenceException` → bir bileşen referansı eksik

Play modunu durdur.

---

# FAZ 3: CONTROLLER TAKİBİ

## Bölüm 3.1: Controller Transform'larını Bağla

Faz 6.1'de `OVRInteractionComprehensive` eklendikten sonra buraya geri dön:

1. `GameManager` seç → Inspector'da **AuraXRFeatureAssembler**
2. **Left Controller Transform** → `LeftControllerAnchor` sürükle (OVRCameraRig → TrackingSpace)
3. **Right Controller Transform** → `RightControllerAnchor` sürükle (OVRCameraRig → TrackingSpace)

### El Görsellerini Devre Dışı Bırak

1. `OVRLeftHandVisual` seç → GameObject'i devre dışı bırak
2. `OVRRightHandVisual` seç → GameObject'i devre dışı bırak

### Locomotor'ı Devre Dışı Bırak

`Locomotor` seç → GameObject'i devre dışı bırak

---

# FAZ 4: EL RİGLEME

## Bölüm 4.1: MANO El Modeli İçe Aktar

> **Not:** MANO indirmesi `.pkl` dosyaları (Python parametrik modeli) sağlar, FBX değil. pkl dosyasını Unity'ye aktarmaya **çalışma**. Bunun yerine Oculus SDK'ya zaten dahil olan OVR el prefab'larını kullan — bunlar zaten `Assets/AuraXR/Rigs/` içinde `OVRCustomHandPrefab_L` ve `OVRCustomHandPrefab_R` olarak mevcuttur. Dönüştürme veya içe aktarma gerekmez.

pkl dosyaları yalnızca Python eğitim pipeline'ı (HOT3D veri seti anotasyonları) tarafından kullanılır. Unity görsel mesh'i OVR el modelini kullanır.

---

## Bölüm 4.2: El Prefab'ları Oluştur

### OVR Bone → MANO Eklem Eşleştirmesi (Sol El)

OVRCustomHandPrefab bu kemik isimlendirmesini kullanır. `fingerJoints` dizisi için MANO sırasına eşle:

| Element | MANO eklemi | OVR kemiği |
|---------|------------|------------|
| 0 | Thumb proximal | `b_l_thumb1` (b_l_thumb0'ın çocuğu) |
| 1 | Thumb middle | `b_l_thumb2` |
| 2 | Thumb distal | `b_l_thumb3` |
| 3 | Index proximal | `b_l_index1` |
| 4 | Index middle | `b_l_index2` |
| 5 | Index distal | `b_l_index3` |
| 6 | Middle proximal | `b_l_middle1` |
| 7 | Middle middle | `b_l_middle2` |
| 8 | Middle distal | `b_l_middle3` |
| 9 | Ring proximal | `b_l_ring1` |
| 10 | Ring middle | `b_l_ring2` |
| 11 | Ring distal | `b_l_ring3` |
| 12 | Pinky proximal | `b_l_pinky1` (b_l_pinky0'ın çocuğu) |
| 13 | Pinky middle | `b_l_pinky2` |
| 14 | Pinky distal | `b_l_pinky3` |

> `b_l_thumb0`, `b_l_pinky0`, `b_l_forearm_stub` ve tüm `_marker` kemiklerini **atla**.
> Sağ el için `b_l_` yerine `b_r_` kullan.

### LeftHandRig

1. **Hierarchy** → sağ tıkla → **Create Empty** → isim: `LeftHandRig`
2. **Layer** → `HandRig` seç
3. `Assets/AuraXR/Rigs/` içindeki `OVRCustomHandPrefab_L`'yi `LeftHandRig`'e alt nesne olarak sürükle
4. `LeftHandRig` seç → **Add Component → HandRigController**
5. **HandRigController** Inspector'ında:
   - **Inference Manager** → `GameManager` sürükle
   - **Is Left Hand** → ✓
   - **Finger Joints** → Size: `15`, yukarıdaki tabloya göre 15 kemiği sürükle
6. `LeftHandRig`'e sağ tıkla → **Prefab → Create Prefab** → `Assets/AuraXR/Prefabs/` konumuna kaydet

### RightHandRig

Yukarıdaki adımları `OVRCustomHandPrefab_R` ile tekrarla, isim `RightHandRig`, **Is Left Hand** → ☐, `b_r_` kemik isimlerini kullan.

### InferenceManager'a Bağla

**GameManager** → **AuraXRInferenceManager** bileşeni seç:
- **Virtual Hand Left** → `LeftHandRig` sürükle
- **Virtual Hand Right** → `RightHandRig` sürükle

---

# FAZ 5: NESNE BETİKLERİ

Betikler Bölüm 2.1'de zaten kopyalandı ve ProximityDetector Bölüm 2.2'de GameManager'a zaten eklendi.

`InteractableObject.cs` ve `GraspIndicator.cs` Faz 6.3'te doğrudan sahne nesnelerine eklenir.

---

# FAZ 6: SAHNE KURULUMU

## Bölüm 6.1: OVRInteractionComprehensive Ekle

1. **Project** sekmesi → `OVRInteractionComprehensive` ara → **Hierarchy**'ye sürükle
2. İkinci bir `OVRManager` **ekleme**

```
OVRCameraRig
└── TrackingSpace
     ├── CenterEyeAnchor
     ├── LeftControllerAnchor           ← FeatureAssembler.leftControllerTransform
     │    └── OVRLeftControllerVisual
     ├── RightControllerAnchor          ← FeatureAssembler.rightControllerTransform
     │    └── OVRRightControllerVisual
     ├── LeftHandAnchor
     ├── RightHandAnchor
     └── OVRInteractionComprehensive
          ├── OVRLeftHandVisual          ← devre dışı bırak
          ├── OVRRightHandVisual         ← devre dışı bırak
          └── Locomotor                  ← devre dışı bırak
```

Şimdi **Bölüm 3.1**'e geri dön ve LeftControllerAnchor / RightControllerAnchor'ı ata.

---

## Bölüm 6.2: Katman & Çarpışma Matrisi

### Özel Katmanlar Oluştur

**Edit → Project Settings → Tags and Layers**

| Slot | İsim |
|------|------|
| 6 | `Environment` |
| 7 | `Player` |
| 8 | `Interactable` |
| 9 | `HandRig` |

### Katman Çarpışma Matrisi

**Edit → Project Settings → Physics** → alttaki **Layer Collision Matrix**:

| Çift | Çarpışsın mı? |
|------|--------------|
| `Player` ↔ `Environment` | ✓ AÇIK |
| `Interactable` ↔ `Environment` | ✓ AÇIK |
| `HandRig` ↔ `Environment` | KAPALI |
| `HandRig` ↔ `Interactable` | KAPALI |
| `Player` ↔ `Interactable` | KAPALI |
| `Player` ↔ `HandRig` | KAPALI |

### Physics Solver

Hâlâ **Edit → Project Settings → Physics** içindeyken:
- **Default Solver Iterations**: `12`
- **Default Solver Velocity Iterations**: `4`

> Aşağıdaki ortamı oluşturmadan önce **step2_unity.md**'yi tamamla (CharacterController + ThumbstickLocomotion).

---

## Bölüm 6.3: VR Mutfak Ortamı

### Floor (Zemin)

1. **Hierarchy** → sağ tıkla → **3D Object → Plane** → isim: `Floor`
2. Konum `(0, 0, 0)`, Ölçek `(0.4, 1, 0.4)`
3. **Layer** → `Environment`
4. **MeshCollider** mevcut olduğunu doğrula, **Is Trigger = KAPALI**
5. `FloorMaterial` oluştur: Standard, Albedo RGB (180, 160, 130), Metallic 0, Smoothness 0.2
6. `Floor`'a uygula

### Walls (Duvarlar)

1. **Hierarchy** → sağ tıkla → **Create Empty** → isim: `Walls`
2. `Walls`'a sağ tıkla → her biri için **3D Object → Cube**:

| İsim | Konum | Ölçek |
|------|-------|-------|
| `WallBack` | (0, 1.25, 2.0) | (4, 2.5, 0.1) |
| `WallLeft` | (-2.0, 1.25, 0) | (0.1, 2.5, 4) |
| `WallRight` | (2.0, 1.25, 0) | (0.1, 2.5, 4) |

Her duvar: **Layer** → `Environment`, BoxCollider **Is Trigger = KAPALI**

`WallMaterial` oluştur: Standard, Albedo RGB (220, 215, 200), Smoothness 0.05. Üçüne de uygula.

### Table (Masa)

1. **Hierarchy** → sağ tıkla → **Create Empty** → isim: `Table`, konum `(0, 0, 1.4)`
2. `Table`'a sağ tıkla → **3D Object → Cube** → isim: `TableTop`
   - Yerel konum `(0, 0.75, 0)`, Ölçek `(1.1, 0.04, 0.65)`
   - **Layer** → `Environment`, BoxCollider **Is Trigger = KAPALI**
   - `TableMaterial` oluştur: Standard, Albedo RGB (160, 100, 60), Smoothness 0.4, Metallic 0
3. `Table`'a sağ tıkla → her ayak için **3D Object → Cube**:

| İsim | Yerel Konum | Ölçek |
|------|-------------|-------|
| `LegFrontLeft` | (-0.50, 0.365, 0.29) | (0.05, 0.73, 0.05) |
| `LegFrontRight` | (0.50, 0.365, 0.29) | (0.05, 0.73, 0.05) |
| `LegBackLeft` | (-0.50, 0.365, -0.29) | (0.05, 0.73, 0.05) |
| `LegBackRight` | (0.50, 0.365, -0.29) | (0.05, 0.73, 0.05) |

Her ayak: **Layer** → `Environment`, BoxCollider **Is Trigger = KAPALI**, `TableMaterial` uygula.

### Bottle (Şişe)

1. **Hierarchy** → sağ tıkla → **3D Object → Cylinder** → isim: `Bottle`
2. Konum `(0.15, 0.925, 1.4)`, Ölçek `(0.045, 0.155, 0.045)`
3. **Layer** → `Interactable`, CapsuleCollider **Is Trigger = KAPALI**
4. `BottleMaterial` oluştur: Standard, Albedo RGB (80, 160, 90), Metallic 0, Smoothness 0.8
   - **Emission**: onay kutusunu işaretle, rengi siyah bırak
5. **Add Component → InteractableObject**: Category Id `1`, Name `bottle`
6. **Add Component → Rigidbody**:
   - Mass `0.3`, Drag `1.0`, Angular Drag `0.5`
   - Use Gravity ✓, Is Kinematic ☐
   - Interpolate: `Interpolate`, Collision Detection: `Continuous Dynamic`
   - Constraints → Freeze Rotation: X ✓, Z ✓
7. **Add Component → GraspIndicator**:
   - Left Hand Rig → `LeftHandRig` ← Faz 4'ten sonra ata
   - Right Hand Rig → `RightHandRig` ← Faz 4'ten sonra ata
   - Highlight Distance `0.15`

### Cup (Bardak)

1. **Hierarchy** → sağ tıkla → **3D Object → Cylinder** → isim: `Cup`
2. Konum `(-0.18, 0.855, 1.4)`, Ölçek `(0.055, 0.085, 0.055)`
3. **Layer** → `Interactable`, CapsuleCollider **Is Trigger = KAPALI**
4. `CupMaterial` oluştur: Standard, Albedo RGB (240, 235, 220), Smoothness 0.6
   - **Emission**: onay kutusunu işaretle, rengi siyah bırak
5. **Add Component → InteractableObject**: Category Id `3`, Name `cup`
6. **Add Component → Rigidbody**: Bottle ile aynı ayarlar
7. **Add Component → GraspIndicator**: Bottle ile aynı ayarlar

### Plate (Tabak — yalnızca dekorasyon)

1. **Hierarchy** → sağ tıkla → **3D Object → Cylinder** → isim: `Plate`
2. Konum `(-0.18, 0.775, 1.4)`, Ölçek `(0.14, 0.005, 0.14)`
3. Katman: Default, bileşen yok

### Aydınlatma

Varsa varsayılan Directional Light'ı sil.

1. **Hierarchy** → sağ tıkla → **Light → Directional Light** → isim: `SunLight`
2. Rotasyon `(55, -30, 0)`
3. Intensity `1.1`, Color RGB (255, 248, 235), Shadow Type **Soft Shadows**, Shadow Strength `0.7`

**Window → Rendering → Lighting → Environment**:
- Source: Color, Ambient Color RGB (80, 90, 110), Intensity Multiplier `0.6`

### Final Sahne Hierarchy'si

```
Scene
├── GameManager
├── OVRCameraRig  (Layer: Player, CharacterController, ThumbstickLocomotion)
│    └── TrackingSpace
│         ├── CenterEyeAnchor
│         ├── LeftControllerAnchor   ← FeatureAssembler.leftControllerTransform
│         │    └── OVRLeftControllerVisual
│         ├── RightControllerAnchor  ← FeatureAssembler.rightControllerTransform
│         │    └── OVRRightControllerVisual
│         ├── LeftHandAnchor
│         ├── RightHandAnchor
│         └── OVRInteractionComprehensive
│              ├── OVRLeftHandVisual  (devre dışı)
│              ├── OVRRightHandVisual (devre dışı)
│              └── Locomotor          (devre dışı)
├── LeftHandRig   (Layer: HandRig)
├── RightHandRig  (Layer: HandRig)
├── SunLight
├── Floor
├── Walls
│    ├── WallBack
│    ├── WallLeft
│    └── WallRight
├── Table
│    ├── TableTop
│    ├── LegFrontLeft / LegFrontRight
│    └── LegBackLeft  / LegBackRight
├── Bottle  (Layer: Interactable, InteractableObject, Rigidbody, GraspIndicator)
├── Cup     (Layer: Interactable, InteractableObject, Rigidbody, GraspIndicator)
├── Plate   (dekorasyon)
└── TaskCanvas (World Space, konum 0/1.8/1.5)
     ├── InstructionText (TMP)
     └── TimerText (TMP)
```

---

## Bölüm 6.4: Tam Pipeline'ı Test Et

**Play** (▶) tuşuna bas. Console şunu göstermeli:
```
[AuraXR] Meta loaded. Feature=96  Target=78  T=16
[AuraXR] Model loaded. Input: [1,16,96]  Output: [1,78]
```

Play modunu durdur.

---

# FAZ 7: EL MATERYALİ

1. **Project** → `Assets/AuraXR/Materials/` üzerine sağ tıkla → **Create → Material** → isim: `HandSkinMaterial`
2. Inspector: Shader `Standard`, **Rendering Mode: Fade**, Albedo RGB (230, 180, 140), Metallic 0, Smoothness 0.6
3. Hierarchy'deki el mesh'ine sürükle

---

# FAZ 8: VERİ KAYIT

SessionDataLogger Bölüm 2.2'de zaten eklendi. Inspector'ı doğrula:
- Inference Manager → `GameManager`
- Feature Assembler → `GameManager`
- Enable Logging → ✓

---

# FAZ 8B: UX SENARYOLARI

## Bölüm 8B.1: Mutfak Senaryo Durum Makinesi

`ScenarioKitchenTask` zaten GameManager'da. Inspector'ı bağla:

| Alan | Değer |
|------|-------|
| Bottle | `Bottle` |
| Cup | `Cup` |
| Feature Assembler | `GameManager` |
| UI Display | `TaskCanvas` ← önce 8B.4'te oluştur |
| Sound Manager | `GameManager` |
| Grip Threshold | `0.15` |
| Grip Input Threshold | `0.7` |
| Auto Start | ✓ |

---

## Bölüm 8B.2: Haptic Geri Bildirim

`HapticFeedbackManager` zaten GameManager'da. Inspector'ı bağla:
- Feature Assembler → `GameManager`
- Haptic Trigger Distance → `0.12`

---

## Bölüm 8B.3: Kavrama Göstergesi

Bottle ve Cup üzerinde Bölüm 6.3'ten zaten mevcut. Her iki nesnenin şunlara sahip olduğunu doğrula:
- Left Hand Rig → `LeftHandRig`
- Right Hand Rig → `RightHandRig`
- Highlight Distance → `0.15`

---

## Bölüm 8B.4: UI Görev Ekranı

1. **Hierarchy** → sağ tıkla → **UI → Canvas** → isim: `TaskCanvas`
2. Render Mode: **World Space**
3. Konum `(0, 1.8, 1.5)`, Ölçek `(0.005, 0.005, 0.005)`
4. `TaskCanvas`'a sağ tıkla → **UI → TextMeshPro - Text** → isim: `InstructionText`
   - Yazı tipi boyutu `0.08`, Hizalama: Center/Middle, Renk: beyaz
5. `TaskCanvas`'a sağ tıkla → **UI → TextMeshPro - Text** → isim: `TimerText`
   - InstructionText'in üzerine yerleştir, Yazı tipi boyutu `0.06`, Renk: açık gri
6. `TaskCanvas` seç → **Add Component → UITaskDisplay**
   - Instruction Text → `InstructionText`
   - Timer Text → `TimerText`

---

## Bölüm 8B.5: Sound Manager

`SoundManager` zaten GameManager'da. Inspector'da AudioClip'leri ata (henüz yoksa boş bırak):
- Pickup Clip, Pour Clip, Place Clip, Complete Clip

---

## Bölüm 8B.6: Koşul Yöneticisi

`ConditionManager` zaten GameManager'da. Inspector'ı bağla:

| Alan | Değer |
|------|-------|
| Left Hand Rig | `LeftHandRig` |
| Right Hand Rig | `RightHandRig` |
| Left Controller Model | `OVRLeftControllerVisual` |
| Right Controller Model | `OVRRightControllerVisual` |
| Inference Manager | `GameManager` |
| Debug Condition | `VirtualHands` |

Her oturumdan önce koşulu ayarla:
```bash
adb shell am broadcast -a com.aura.setcondition --ei condition 0   # A: VirtualHands
adb shell am broadcast -a com.aura.setcondition --ei condition 1   # B: Controller
adb shell am broadcast -a com.aura.setcondition --ei condition 2   # C: StaticPose
```

Latin kare sırası (3 koşul, 20 katılımcı):
- Grup 1: A→B→C
- Grup 2: B→C→A
- Grup 3: C→A→B

---

## Bölüm 8B.7: Final GameManager Bileşen Listesi

| Bileşen | Zorunlu Alanlar |
|---------|----------------|
| AuraXRMetaLoader | Meta Json |
| AuraXRFeatureAssembler | Sol/Sağ Controller Transform |
| AuraXRInferenceManager | Model Asset, Meta Loader, Feature Assembler, Virtual Hand Sol/Sağ |
| ProximityDetector | Feature Assembler, Search Radius |
| HandProximityVisibility | Sol/Sağ Controller, Sol/Sağ Hand Rig |
| VirtualHandGrab | Sol/Sağ Hand Wrist |
| ScenarioKitchenTask | Bottle, Cup, Feature Assembler, UI Display, Sound Manager, Auto Start ✓ |
| HapticFeedbackManager | Feature Assembler |
| SoundManager | (ses klipleri isteğe bağlı) |
| ConditionManager | Sol/Sağ Hand Rig, Sol/Sağ Controller Model, Inference Manager |
| SessionDataLogger | Inference Manager, Feature Assembler, Enable Logging ✓ |

---

# FAZ 9: DAĞITIM

## Bölüm 9.1: Build Ayarları

1. **File → Build Settings** → **Android** seçili olduğunu doğrula
2. **Player Settings**:
   - Package Name: `com.yourname.auraxr`
   - Minimum API Level: `24`, Target API Level: `33`
   - Graphics APIs: Vulkan
   - XR Plug-in Management → Android: Oculus ✓
3. **Edit → Project Settings → Quality** → Level: **High**
   - MSAA: 4x, Anisotropic Filtering: Per Texture

## Bölüm 9.2: Build ve Dağıt

```bash
adb devices    # cihazın listelendiğini doğrula
```

1. **File → Build Settings → Build** → klasör: `AuraXR_Build`
2. Yükle:
```bash
adb install -r AuraXR_Build/app-release.apk
```

Log akışı:
```bash
adb logcat -s "AuraXR"
```

---

# FAZ 12: FİNAL KONTROL LİSTESİ

**Betikler & Referanslar**
- [ ] `LeftControllerAnchor` → AuraXRFeatureAssembler.leftControllerTransform
- [ ] `RightControllerAnchor` → AuraXRFeatureAssembler.rightControllerTransform
- [ ] `LeftHandRig` / `RightHandRig` → AuraXRInferenceManager.virtualHandLeft / virtualHandRight
- [ ] `OVRLeftControllerVisual` → ConditionManager.leftControllerModel
- [ ] `OVRRightControllerVisual` → ConditionManager.rightControllerModel
- [ ] `OVRLeftHandVisual` ve `OVRRightHandVisual` devre dışı
- [ ] `Locomotor` devre dışı

**Katmanlar & Fizik**
- [ ] 4 özel katman oluşturuldu (Environment/Player/Interactable/HandRig)
- [ ] Katman Çarpışma Matrisi ayarlandı (Player↔Environment AÇIK, HandRig↔Environment KAPALI)
- [ ] Floor, Walls, Table, ayaklar üzerinde Is Trigger = KAPALI
- [ ] OVRCameraRig'de CharacterController (step2 Blok D.4)
- [ ] ThumbstickLocomotion: Camera Transform = CenterEyeAnchor, Head Collision Layers = Environment
- [ ] Bottle ve Cup: Rigidbody + Continuous Dynamic

**Materyaller**
- [ ] HandSkinMaterial Rendering Mode = **Fade**
- [ ] BottleMaterial ve CupMaterial'de Emission onay kutusu AÇIK

**UX**
- [ ] ScenarioKitchenTask'ta autoStart ✓
- [ ] GraspIndicator: Bottle ve Cup üzerinde leftHandRig / rightHandRig atanmış
- [ ] TaskCanvas Render Mode = World Space

**Console (Play modu)**
- [ ] Sıfır kırmızı hata
- [ ] `[AuraXR] Meta loaded. Feature=96  Target=78  T=16`
- [ ] `[AuraXR] Model loaded. Input: [1,16,96]  Output: [1,78]`

**Build**
- [ ] `adb devices` Quest 3'ü listeler
- [ ] APK yüklenir ve başlar
- [ ] `adb logcat -s "AuraXR"` cihazda çıkarımı doğrular

---

# EK A: DOSYA YAPISI

```
Assets/
├── AuraXR/
│   ├── Scripts/     (tüm .cs dosyaları)
│   ├── Models/      (intentformer.onnx, intentformer_meta.json)
│   ├── Prefabs/     (LeftHandRig.prefab, RightHandRig.prefab)
│   ├── Rigs/        (el FBX)
│   ├── Materials/   (HandSkinMaterial, FloorMaterial, WallMaterial, TableMaterial, BottleMaterial, CupMaterial)
│   ├── UI/
│   └── Data/
└── TextMesh Pro/
```

---

# EK B: Temel Formüller

**Çıktı Düzeni (78 float):**
```
[0–14]   mano_pose_h0      Sol el eklem açıları (eklem başına 1 DoF)
[15–24]  mano_betas_h0     Sol el şekli (v1'de göz ardı edilir)
[25–27]  wrist_t_h0        Sol bilek konumu (metre)
[28–31]  wrist_q_h0        Sol bilek rotasyonu (kuaterniyon w,x,y,z)
[32–34]  delta_t_h0        Controller→bilek ofseti (öteleme)
[35–38]  delta_q_h0        Controller→bilek ofseti (rotasyon)
[39–77]  sağ el için aynı düzen
```

**Sanal Bilek Konumlandırma:**
```csharp
anchor.position = controller.position + pose.DeltaPosition;
anchor.rotation = controller.rotation * pose.DeltaRotation;
```

**Özellik Normalizasyonu:**
```
normalized[i] = (raw[i] - feature_mean[i]) / feature_std[i]
denormalized[i] = normalized[i] * target_std[i] + target_mean[i]
```
